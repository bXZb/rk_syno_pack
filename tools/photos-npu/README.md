# OEC RK3566 Photos `npu_server`

Replacement gRPC `npu.Model` server plus a packer that decrypts the official
`rtd1619b` Synology Photos SPK and emits an unsigned sideload package with:

- aarch64 `npu_server` (official Vivante binary removed)
- `lib_arm64/librknnrt.so` (RKNN-Toolkit2 2.3.2 / driver 0.9.8)
- `asset/network/{concept,detection,feature}_network.rknn`

The GitHub Actions workflow **Pack Synology Photos OEC NPU SPK** runs the
full download → decrypt → convert → pack → upload path. The artifact is
installable with Package Center sideload / `synopkg`.

## CI

`.github/workflows/pack-photos-spk.yml` (workflow_dispatch, or push/PR on
the Photos NPU paths):

1. `go test` the server
2. Download official `SynologyPhotos-x86_64` + `rtd1619b` SPKs
3. Decrypt with SynoXtract
4. Convert official concept TFLite → `concept_network.rknn`
5. Convert public RetinaFace (UltraFace fallback) + MobileFaceNet → face RKNN
6. Fetch aarch64 `librknnrt.so` from `airockchip/rknn-toolkit2` v2.3.2
7. Repack an unsigned `SynologyPhotos-rtd1619b-*-oec-npu.spk` and upload it

Download the `synology-photos-oec-npu-spk` artifact from the run.

## Local pack

```bash
# host tests
cd tools/photos-npu && go test ./...

# CPython 3.10: convert models + fetch librknnrt
scripts/prepare-photos-rknn.sh --install-deps --out-dir build/photos-rknn

# aarch64 binary + sideload SPK (fails if runtime/models are missing)
scripts/pack-photos-spk.sh --rknn-dir build/photos-rknn --require-rknn
```

Install on the DS124 box:

```bash
synopkg install SynologyPhotos-rtd1619b-1.9.1-10928-oec-npu.spk
```

`postinst` writes `SkipMemoryCheck` (DS124 1GB concept gate) and creates
`/dev/galcore` so `IsSupportedNpuNetwork()` succeeds. The real accelerator is
still `/dev/rknpu`.

Official Photos remains Synology's package (`package=SynologyPhotos`,
`arch=rtd1619b`). This repo does not vendor the SPK; CI downloads it.

## Models

| Slot | Source | Notes |
| --- | --- | --- |
| concept | Official x86 `model_float16.tflite` | 395×395, 335 labels |
| detection | airockchip RetinaFace mobile320 (UltraFace RFB-320 fallback) | Official Vivante `.nb` is not convertible |
| feature | sirius-ai MobileFaceNet 128-d (Qualcomm ONNX fallback) | 112×112, matches NPU `feature_size` |
| runtime | `rknn-toolkit2` v2.3.2 `librknnrt.so` | aarch64, glibc 2.36 / bookworm |

Face embeddings are consistent with the packed MobileFaceNet, not with the
x86 OpenVINO 256-d path.

## Protocol

Reconstructed from the official `npu_server` FileDescriptorProtos:

- `unix:///run/synofoto/npu-photo.sock`
- `/npu.Model/ExecCmd`
- `/npu.Model/ConceptDetectBuffer`
- `/npu.Model/FaceDetectBuffer`
- `/npu.Model/FaceFeatureBuffer`

Tensor sizes stay in `npu/npu_model_conf.json`. Concept labels are 335 classes.
Face detection accepts official `128x128x15`, RetinaFace `(loc,conf,landms)`,
or UltraFace `(boxes,scores)`.
