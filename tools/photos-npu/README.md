# OEC RK3566 Photos `npu_server`

Replacement gRPC `npu.Model` server plus a packer that decrypts the official
`rtd1619b` Synology Photos SPK and emits an unsigned sideload package.

## Build / pack

```bash
# host tests
cd tools/photos-npu && go test ./...

# aarch64 binary + sideload SPK
scripts/pack-photos-spk.sh

# include converted models / librknnrt
scripts/pack-photos-spk.sh --rknn-dir /path/to/rknn-bits
```

Install on the DS124 box:

```bash
synopkg install SynologyPhotos-rtd1619b-1.9.1-10928-oec-npu.spk
```

`postinst` writes `SkipMemoryCheck` (DS124 1GB concept gate) and creates
`/dev/galcore` so `IsSupportedNpuNetwork()` succeeds. The real accelerator is
still `/dev/rknpu`.

## Protocol

Reconstructed from the official `npu_server` FileDescriptorProtos:

- `unix:///run/synofoto/npu-photo.sock`
- `/npu.Model/ExecCmd`
- `/npu.Model/ConceptDetectBuffer`
- `/npu.Model/FaceDetectBuffer`
- `/npu.Model/FaceFeatureBuffer`

Tensor sizes stay in `npu/npu_model_conf.json`. Concept labels are 335 classes.
Face detection output is treated as `128x128x15` (score + box + 5 landmarks).

Without `librknnrt.so` and `.rknn` files the socket still comes up; inference
RPCs return empty results.
