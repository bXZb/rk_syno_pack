package rknn

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Optional runtime binding to librknnrt.so. CGO is not used so the server
// still builds as a static aarch64 binary; the .so is dlopened on the NAS.

type Context uint64

const (
	TensorFloat32 = 0
	TensorFloat16 = 1
	TensorInt8    = 2
	TensorUINT8   = 3
	TensorNCHW    = 0
	TensorNHWC    = 1
)

// Layout matches rknn_api.h on aarch64/x86_64 (24-byte structs).
type Input struct {
	Index       uint32
	_           uint32
	Buf         unsafe.Pointer
	Size        uint32
	PassThrough uint8
	Type        uint8
	Fmt         uint8
	_pad        uint8
}

type Output struct {
	WantFloat  uint8
	IsPrealloc uint8
	_          uint16
	Index      uint32
	Buf        unsafe.Pointer
	Size       uint32
	_          uint32
}

type Runtime struct {
	mu      sync.Mutex
	init    func(*Context, unsafe.Pointer, uint32, uint32, unsafe.Pointer) int32
	destroy func(Context) int32
	inSet   func(Context, uint32, unsafe.Pointer) int32
	run     func(Context, unsafe.Pointer) int32
	outGet  func(Context, uint32, unsafe.Pointer, unsafe.Pointer) int32
	outRel  func(Context, uint32, unsafe.Pointer) int32
}

func Open(libPaths ...string) (*Runtime, error) {
	candidates := append([]string{}, libPaths...)
	candidates = append(candidates,
		"librknnrt.so",
		"./lib_arm64/librknnrt.so",
		"/usr/lib/librknnrt.so",
	)
	var last error
	for _, p := range candidates {
		if p == "" {
			continue
		}
		if !filepath.IsAbs(p) {
			if _, err := os.Stat(p); err != nil {
				continue
			}
		}
		lib, err := purego.Dlopen(p, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			last = err
			continue
		}
		rt := &Runtime{}
		purego.RegisterLibFunc(&rt.init, lib, "rknn_init")
		purego.RegisterLibFunc(&rt.destroy, lib, "rknn_destroy")
		purego.RegisterLibFunc(&rt.inSet, lib, "rknn_inputs_set")
		purego.RegisterLibFunc(&rt.run, lib, "rknn_run")
		purego.RegisterLibFunc(&rt.outGet, lib, "rknn_outputs_get")
		purego.RegisterLibFunc(&rt.outRel, lib, "rknn_outputs_release")
		if rt.init == nil || rt.run == nil {
			last = fmt.Errorf("librknnrt missing symbols in %s", p)
			continue
		}
		return rt, nil
	}
	if last == nil {
		last = fmt.Errorf("librknnrt.so not found (GOARCH=%s)", runtime.GOARCH)
	}
	return nil, last
}

func (rt *Runtime) Load(modelPath string) (Context, error) {
	raw, err := os.ReadFile(modelPath)
	if err != nil {
		return 0, err
	}
	var ctx Context
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rc := rt.init(&ctx, unsafe.Pointer(&raw[0]), uint32(len(raw)), 0, nil)
	runtime.KeepAlive(raw)
	if rc != 0 {
		return 0, fmt.Errorf("rknn_init %s: %d", modelPath, rc)
	}
	return ctx, nil
}

func (rt *Runtime) Destroy(ctx Context) {
	if ctx == 0 || rt.destroy == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.destroy(ctx)
}

func (rt *Runtime) InferFloat32(ctx Context, input []float32, nOut int) ([][]float32, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("empty input")
	}
	in := Input{
		Index: 0,
		Buf:   unsafe.Pointer(&input[0]),
		Size:  uint32(len(input) * 4),
		Type:  TensorFloat32,
		Fmt:   TensorNCHW,
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rc := rt.inSet(ctx, 1, unsafe.Pointer(&in)); rc != 0 {
		return nil, fmt.Errorf("rknn_inputs_set: %d", rc)
	}
	runtime.KeepAlive(input)
	if rc := rt.run(ctx, nil); rc != 0 {
		return nil, fmt.Errorf("rknn_run: %d", rc)
	}
	outs := make([]Output, nOut)
	for i := range outs {
		outs[i].WantFloat = 1
		outs[i].Index = uint32(i)
	}
	if rc := rt.outGet(ctx, uint32(nOut), unsafe.Pointer(&outs[0]), nil); rc != 0 {
		return nil, fmt.Errorf("rknn_outputs_get: %d", rc)
	}
	got := make([][]float32, nOut)
	for i, o := range outs {
		n := int(o.Size) / 4
		if n < 0 {
			n = 0
		}
		cp := make([]float32, n)
		if o.Buf != nil && n > 0 {
			src := unsafe.Slice((*float32)(o.Buf), n)
			copy(cp, src)
		}
		got[i] = cp
	}
	rt.outRel(ctx, uint32(nOut), unsafe.Pointer(&outs[0]))
	return got, nil
}
