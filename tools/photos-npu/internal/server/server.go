package server

import (
	"bufio"
	"context"
	"encoding/json"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/bXZb/rk_syno_pack/tools/photos-npu/internal/detect"
	"github.com/bXZb/rk_syno_pack/tools/photos-npu/internal/npupb"
	"github.com/bXZb/rk_syno_pack/tools/photos-npu/internal/preprocess"
	"github.com/bXZb/rk_syno_pack/tools/photos-npu/internal/rknn"
)

type netCfg struct {
	Reorder      bool      `json:"reorder"`
	RGBMean      []float32 `json:"rgb_mean"`
	RGBScale     float32   `json:"rgb_scale"`
	NetSize      []int     `json:"net_size"`
	ResultThres  float32   `json:"result_thres"`
	IOUThres     float32   `json:"iou_thres"`
	FeatureSize  any       `json:"feature_size"`
}

type modelConf struct {
	Detection netCfg `json:"detection"`
	Feature   netCfg `json:"feature"`
	Concept   netCfg `json:"concept"`
}

type Model struct {
	npupb.UnimplementedModelServer
	dir        string
	conf       modelConf
	labels     []string
	thresholds []float32

	rknnOnce sync.Once
	rt       *rknn.Runtime
	rtErr    error
	ctxMu    sync.Mutex
	concept  rknn.Context
	detect   rknn.Context
	feature  rknn.Context
}

func New(dir string) (*Model, error) {
	m := &Model{dir: dir}
	if raw, err := os.ReadFile(filepath.Join(dir, "npu_model_conf.json")); err == nil {
		if err := json.Unmarshal(raw, &m.conf); err != nil {
			return nil, err
		}
	} else {
		m.conf = defaultConf()
	}
	m.labels = readLines(firstExisting(
		filepath.Join(dir, "asset", "labels.txt"),
		filepath.Join(dir, "labels.txt"),
	))
	m.thresholds = readFloats(firstExisting(
		filepath.Join(dir, "asset", "thresholds.txt"),
		filepath.Join(dir, "thresholds.txt"),
	))
	return m, nil
}

func defaultConf() modelConf {
	return modelConf{
		Detection: netCfg{RGBMean: []float32{127.5, 127.5, 127.5}, RGBScale: 0.0078125, NetSize: []int{512, 512, 3}, ResultThres: 0.5, IOUThres: 0.5},
		Feature:   netCfg{RGBMean: []float32{127.5, 127.5, 127.5}, RGBScale: 0.0078125, NetSize: []int{112, 112, 3}},
		Concept:   netCfg{RGBMean: []float32{127.5, 127.5, 127.5}, RGBScale: 0.00784313725, NetSize: []int{395, 395, 3}},
	}
}

func (m *Model) Close() {
	m.ctxMu.Lock()
	defer m.ctxMu.Unlock()
	if m.rt == nil {
		return
	}
	m.rt.Destroy(m.concept)
	m.rt.Destroy(m.detect)
	m.rt.Destroy(m.feature)
}

func (m *Model) ExecCmd(_ context.Context, req *npupb.CmdRequest) (*npupb.CmdReply, error) {
	name := strings.ToLower(strings.TrimSpace(req.GetName()))
	log.Printf("ExecCmd %q", name)
	switch name {
	case "status", "ping", "ready", "":
		return &npupb.CmdReply{Result: "ready"}, nil
	default:
		return &npupb.CmdReply{Result: "ok"}, nil
	}
}

func (m *Model) ConceptDetectBuffer(_ context.Context, req *npupb.BufferRequest) (*npupb.ConceptDetectReply, error) {
	reply := &npupb.ConceptDetectReply{ReplyMap: map[string]*npupb.ConceptInfos{}}
	scores, err := m.runNet("concept", req.GetBuffer(), m.conf.Concept, 1)
	if err != nil {
		log.Printf("concept infer: %v", err)
		return reply, nil
	}
	if len(scores) == 0 {
		return reply, nil
	}
	out := scores[0]
	for i, label := range m.labels {
		if i >= len(out) {
			break
		}
		s := out[i]
		if s < 0 || s > 1 {
			s = sigmoid(s)
		}
		th := float32(0)
		if i < len(m.thresholds) {
			th = m.thresholds[i]
		}
		if th > 0 && s < th {
			continue
		}
		if s <= 0 {
			continue
		}
		reply.ReplyMap[label] = &npupb.ConceptInfos{Info: []float32{s}}
	}
	return reply, nil
}

func (m *Model) FaceDetectBuffer(_ context.Context, req *npupb.BufferRequest) (*npupb.FaceDetectReply, error) {
	rep := &npupb.FaceDetectReply{}
	img, err := preprocess.DecodeImage(req.GetBuffer())
	if err != nil {
		log.Printf("face detect decode: %v", err)
		return rep, nil
	}
	outs, err := m.runNet("detection", req.GetBuffer(), m.conf.Detection, 1)
	if err != nil {
		log.Printf("face detect infer: %v", err)
		return rep, nil
	}
	if len(outs) == 0 {
		return rep, nil
	}
	b := img.Bounds()
	faces := detect.Decode128x15(outs[0], b.Dx(), b.Dy(), m.conf.Detection.ResultThres, m.conf.Detection.IOUThres)
	for _, f := range faces {
		rep.FaceInfo = append(rep.FaceInfo, &npupb.FaceInfo{
			Bbox: &npupb.FaceRect{
				X1: f.X1, Y1: f.Y1, X2: f.X2, Y2: f.Y2, Confidance: f.Score,
			},
			Landmarks: &npupb.FaceLandmarks{X: f.LX, Y: f.LY},
		})
	}
	return rep, nil
}

func (m *Model) FaceFeatureBuffer(_ context.Context, req *npupb.BufferRequest) (*npupb.FaceFeatureReply, error) {
	rep := &npupb.FaceFeatureReply{}
	outs, err := m.runNet("feature", req.GetBuffer(), m.conf.Feature, 1)
	if err != nil {
		log.Printf("face feature infer: %v", err)
		return rep, nil
	}
	if len(outs) == 0 || len(outs[0]) == 0 {
		return rep, nil
	}
	feat := outs[0]
	rep.FaceFeature = feat
	var sum float32
	for _, v := range feat {
		sum += v * v
	}
	rep.FeatureScore = float32(math.Sqrt(float64(sum)))
	return rep, nil
}

func (m *Model) runNet(kind string, jpeg []byte, cfg netCfg, nOut int) ([][]float32, error) {
	if len(jpeg) == 0 {
		return nil, nil
	}
	img, err := preprocess.DecodeImage(jpeg)
	if err != nil {
		return nil, err
	}
	w, h := 395, 395
	if len(cfg.NetSize) >= 2 && cfg.NetSize[0] > 0 {
		w, h = cfg.NetSize[0], cfg.NetSize[1]
	}
	mean := float32(127.5)
	if len(cfg.RGBMean) > 0 {
		mean = cfg.RGBMean[0]
	}
	scale := cfg.RGBScale
	if scale == 0 {
		scale = 1.0 / 127.5
	}
	tensor := preprocess.NCHWFloat32(img, preprocess.Config{Width: w, Height: h, Mean: mean, Scale: scale})

	m.ensureRKNN()
	if m.rt == nil {
		return nil, m.rtErr
	}
	ctx, err := m.ctxFor(kind)
	if err != nil || ctx == 0 {
		return nil, err
	}
	return m.rt.InferFloat32(ctx, tensor, nOut)
}

func (m *Model) ensureRKNN() {
	m.rknnOnce.Do(func() {
		m.rt, m.rtErr = rknn.Open(
			filepath.Join(m.dir, "lib_arm64", "librknnrt.so"),
			filepath.Join(m.dir, "librknnrt.so"),
		)
		if m.rtErr != nil {
			log.Printf("rknn runtime: %v (RPC still served; inference empty)", m.rtErr)
		}
	})
}

func (m *Model) ctxFor(kind string) (rknn.Context, error) {
	m.ctxMu.Lock()
	defer m.ctxMu.Unlock()
	var slot *rknn.Context
	var names []string
	switch kind {
	case "concept":
		slot = &m.concept
		names = []string{"asset/network/concept_network.rknn", "asset/network/concept.rknn", "concept_fp16.rknn", "concept.rknn"}
	case "detection":
		slot = &m.detect
		names = []string{"asset/network/detection_network.rknn", "asset/network/detection.rknn", "detection.rknn"}
	case "feature":
		slot = &m.feature
		names = []string{"asset/network/feature_network.rknn", "asset/network/feature.rknn", "feature.rknn"}
	default:
		return 0, nil
	}
	if *slot != 0 {
		return *slot, nil
	}
	if m.rt == nil {
		return 0, m.rtErr
	}
	for _, rel := range names {
		p := filepath.Join(m.dir, rel)
		if _, err := os.Stat(p); err != nil {
			continue
		}
		ctx, err := m.rt.Load(p)
		if err != nil {
			return 0, err
		}
		*slot = ctx
		log.Printf("loaded %s model %s", kind, p)
		return ctx, nil
	}
	return 0, nil
}

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

func firstExisting(paths ...string) string {
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func readLines(path string) []string {
	if path == "" {
		return nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		s := strings.TrimSpace(sc.Text())
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func readFloats(path string) []float32 {
	var out []float32
	for _, s := range readLines(path) {
		v, err := strconv.ParseFloat(s, 32)
		if err == nil {
			out = append(out, float32(v))
		}
	}
	return out
}
