#!/usr/bin/env bash
# Download official Photos x86 models + aarch64 librknnrt.so + public face
# graphs, then convert them to RK3566 RKNN for scripts/pack-photos-spk.sh.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare-photos-rknn.sh [options]

Options:
  --out-dir DIR          Output directory (default build/photos-rknn)
  --photos-version VER   Official Photos version (default 1.9.1-10928)
  --x86-spk PATH         Pre-downloaded x86_64 Photos SPK
  --platform NAME        RKNN target_platform (default rk3566)
  --install-deps         Install Python conversion deps (CPython 3.8-3.12)
  --skip-convert         Only download/extract; do not run rknn-toolkit2
  --skip-concept|--skip-detection|--skip-feature
  -h, --help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SYNOXTRACT="${PROJECT_DIR}/tools/SynoXtract/synoxtract"

OUT_DIR="${PHOTOS_RKNN_OUT:-${PROJECT_DIR}/build/photos-rknn}"
PHOTOS_VERSION="${PHOTOS_VERSION:-1.9.1-10928}"
X86_SPK=""
PLATFORM="rk3566"
INSTALL_DEPS=0
SKIP_CONVERT=0
SKIP_CONCEPT=0
SKIP_DETECTION=0
SKIP_FEATURE=0

RKNN_TOOLKIT_VER="2.3.2"
LIBRKNNRT_URL="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v${RKNN_TOOLKIT_VER}/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
LIBRKNNRT_GIT="https://github.com/airockchip/rknn-toolkit2.git"

RETINA_URLS=(
  "https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/RetinaFace/RetinaFace_mobile320.onnx"
)
ULTRA_URLS=(
  "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx"
  "https://github.com/onnx/models/raw/main/vision/body_analysis/ultraface/models/version-RFB-320.onnx"
  "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx"
)
FEATURE_PB_URLS=(
  "https://raw.githubusercontent.com/sirius-ai/MobileFaceNet_TF/master/arch/pretrained_model/MobileFaceNet_9925_9680.pb"
)
FEATURE_ONNX_URLS=(
  "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mobile_facenet/releases/v0.64.0/mobile_facenet-onnx-float.zip"
)

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --photos-version) PHOTOS_VERSION="$2"; shift 2 ;;
    --x86-spk) X86_SPK="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    --skip-convert) SKIP_CONVERT=1; shift ;;
    --skip-concept) SKIP_CONCEPT=1; shift ;;
    --skip-detection) SKIP_DETECTION=1; shift ;;
    --skip-feature) SKIP_FEATURE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[photos-rknn] %s\n' "$*"; }
die() { printf '[photos-rknn] error: %s\n' "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

download() {
  local dest="$1"; shift
  local url
  mkdir -p "$(dirname "${dest}")"
  for url in "$@"; do
    log "download ${url}"
    if curl -L --fail --retry 3 --retry-delay 2 -o "${dest}.part" "${url}"; then
      mv -f "${dest}.part" "${dest}"
      return 0
    fi
    rm -f "${dest}.part"
  done
  return 1
}

is_lfs_pointer() {
  [ -f "$1" ] && head -c 20 "$1" | grep -q 'version https://git-lfs'
}

ensure_synoxtract() {
  if [ -x "${SYNOXTRACT}" ]; then
    return 0
  fi
  need_cmd g++
  log "building SynoXtract"
  g++ -O2 -std=c++17 -o "${SYNOXTRACT}" \
    "${PROJECT_DIR}/tools/SynoXtract/main.cpp" -lsodium
  chmod +x "${SYNOXTRACT}"
}

fetch_librknnrt() {
  local dest="${OUT_DIR}/librknnrt.so"
  if [ -f "${dest}" ] && [ "$(stat -c%s "${dest}")" -gt 1000000 ]; then
    log "using cached librknnrt.so"
    return 0
  fi
  if download "${dest}" "${LIBRKNNRT_URL}" && ! is_lfs_pointer "${dest}"; then
    :
  else
    log "raw librknnrt failed; sparse-checkout v${RKNN_TOOLKIT_VER}"
    local repo="${WORK}/rknn-toolkit2"
    rm -rf "${repo}"
    git clone --depth 1 --branch "v${RKNN_TOOLKIT_VER}" --filter=blob:none \
      "${LIBRKNNRT_GIT}" "${repo}"
    git -C "${repo}" sparse-checkout set rknpu2/runtime/Linux/librknn_api/aarch64
    cp -f "${repo}/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" "${dest}"
  fi
  [ -f "${dest}" ] || die "librknnrt.so missing"
  if is_lfs_pointer "${dest}"; then
    die "librknnrt.so is a git-lfs pointer"
  fi
  local sz
  sz="$(stat -c%s "${dest}")"
  [ "${sz}" -gt 1000000 ] || die "librknnrt.so too small (${sz})"
  if command -v file >/dev/null 2>&1; then
    file -b "${dest}" | grep -Eqi 'ELF|aarch64|ARM' || \
      log "warning: unexpected librknnrt type: $(file -b "${dest}")"
  fi
  log "librknnrt.so ${sz} bytes"
}

extract_concept() {
  local x86="${X86_SPK}"
  if [ -z "${x86}" ]; then
    x86="${WORK}/SynologyPhotos-x86_64-${PHOTOS_VERSION}.spk"
    if [ ! -f "${x86}" ]; then
      download "${x86}" \
        "https://global.synologydownload.com/download/Package/spk/SynologyPhotos/${PHOTOS_VERSION}/SynologyPhotos-x86_64-${PHOTOS_VERSION}.spk" \
        || die "x86 Photos SPK download failed"
    fi
  fi
  [ -f "${x86}" ] || die "x86 SPK not found: ${x86}"
  ensure_synoxtract
  local extract="${WORK}/x86x"
  local pkg="${WORK}/x86"
  rm -rf "${extract}" "${pkg}"
  mkdir -p "${extract}" "${pkg}"
  log "decrypting $(basename "${x86}")"
  "${SYNOXTRACT}" -i "${x86}" -d "${extract}"
  [ -f "${extract}/package.tgz" ] || die "synoxtract did not produce package.tgz"
  local fmt
  fmt="$(file -b "${extract}/package.tgz")"
  case "${fmt}" in
    *XZ*) tar -xJf "${extract}/package.tgz" -C "${pkg}" ;;
    *gzip*) tar -xzf "${extract}/package.tgz" -C "${pkg}" ;;
    *bzip2*) tar -xjf "${extract}/package.tgz" -C "${pkg}" ;;
    *) tar -xf "${extract}/package.tgz" -C "${pkg}" ;;
  esac
  local tflite="${pkg}/models/concept/detector/model_float16.tflite"
  [ -f "${tflite}" ] || die "concept TFLite missing after extract"
  mkdir -p "${OUT_DIR}"
  cp -f "${tflite}" "${OUT_DIR}/model_float16.tflite"
  cp -f "${pkg}/models/concept/detector/labels.txt" "${OUT_DIR}/labels.txt"
  cp -f "${pkg}/models/concept/detector/thresholds.txt" "${OUT_DIR}/thresholds.txt"
  CONCEPT_TFLITE="${OUT_DIR}/model_float16.tflite"
  log "concept tflite $(stat -c%s "${CONCEPT_TFLITE}") bytes"
}

fetch_face_models() {
  DETECTION_ONNX=""
  FEATURE_ONNX=""
  FEATURE_PB=""
  if [ "${SKIP_DETECTION}" != 1 ]; then
    if download "${WORK}/RetinaFace_mobile320.onnx" "${RETINA_URLS[@]}" \
      && [ "$(stat -c%s "${WORK}/RetinaFace_mobile320.onnx")" -gt 100000 ]; then
      DETECTION_ONNX="${WORK}/RetinaFace_mobile320.onnx"
      log "detection=RetinaFace_mobile320.onnx"
    elif download "${WORK}/version-RFB-320.onnx" "${ULTRA_URLS[@]}" \
      && [ "$(stat -c%s "${WORK}/version-RFB-320.onnx")" -gt 50000 ]; then
      DETECTION_ONNX="${WORK}/version-RFB-320.onnx"
      log "detection=version-RFB-320.onnx (UltraFace fallback)"
    else
      die "could not download a face-detection ONNX"
    fi
  fi
  if [ "${SKIP_FEATURE}" != 1 ]; then
    if download "${WORK}/MobileFaceNet_9925_9680.pb" "${FEATURE_PB_URLS[@]}" \
      && [ "$(stat -c%s "${WORK}/MobileFaceNet_9925_9680.pb")" -gt 1000000 ]; then
      FEATURE_PB="${WORK}/MobileFaceNet_9925_9680.pb"
      log "feature=MobileFaceNet_9925_9680.pb"
    fi
    if download "${WORK}/mobile_facenet-onnx-float.zip" "${FEATURE_ONNX_URLS[@]}"; then
      mkdir -p "${WORK}/mfn"
      unzip -o -q "${WORK}/mobile_facenet-onnx-float.zip" -d "${WORK}/mfn"
      FEATURE_ONNX="$(find "${WORK}/mfn" -name '*.onnx' | head -n1 || true)"
      if [ -n "${FEATURE_ONNX}" ]; then
        log "feature-onnx=$(basename "${FEATURE_ONNX}")"
      fi
    fi
    if [ -z "${FEATURE_PB}" ] && [ -z "${FEATURE_ONNX}" ]; then
      die "could not download MobileFaceNet"
    fi
  fi
}

install_python_deps() {
  need_cmd python3
  python3 - <<'PY'
import sys
if sys.version_info < (3, 8) or sys.version_info >= (3, 13):
    raise SystemExit(f"rknn-toolkit2 2.3.2 supports CPython 3.8-3.12, got {sys.version}")
print("python", sys.version)
PY
  # Sequential pins match .github/workflows/convert-concept.yml.
  # A single pip -r of requirements-convert.txt fails the resolver
  # (tensorflow-cpu 2.15 / tf2onnx / onnxruntime / protobuf).
  pip install --quiet "numpy==1.26.4" psutil "ruamel.yaml" scipy tqdm opencv-python fast-histogram
  pip install --quiet "onnx==1.16.1" "onnxruntime==1.16.3" "protobuf==4.25.4"
  pip install --quiet "tensorflow-cpu==2.15.1" "tf2onnx==1.16.1"
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet "protobuf==4.25.4" "numpy==1.26.4"
  mkdir -p "${WORK}/whl"
  pip download "rknn-toolkit2==${RKNN_TOOLKIT_VER}" --no-deps -d "${WORK}/whl"
  WHL="$(ls "${WORK}/whl"/rknn_toolkit2-*.whl | head -n1)"
  [ -n "${WHL}" ] || die "rknn-toolkit2 wheel not downloaded"
  pip install --quiet --no-deps "${WHL}"
  python3 -c "from rknn.api import RKNN; print('rknn api ok', '${WHL}')"
}

run_convert() {
  local args=(
    python3 "${PROJECT_DIR}/scripts/convert-photos-rknn.py"
    --out-dir "${OUT_DIR}"
    --platform "${PLATFORM}"
  )
  if [ "${SKIP_CONCEPT}" = 1 ]; then
    args+=(--skip-concept)
  else
    args+=(--concept-tflite "${CONCEPT_TFLITE}")
  fi
  if [ "${SKIP_DETECTION}" = 1 ]; then
    args+=(--skip-detection)
  else
    args+=(--detection-onnx "${DETECTION_ONNX}")
  fi
  if [ "${SKIP_FEATURE}" = 1 ]; then
    args+=(--skip-feature)
  else
    if [ -n "${FEATURE_ONNX}" ]; then
      args+=(--feature-onnx "${FEATURE_ONNX}")
    fi
    if [ -n "${FEATURE_PB}" ]; then
      args+=(--feature-pb "${FEATURE_PB}")
    fi
  fi
  log "converting to RKNN (${PLATFORM})"
  "${args[@]}"
}

need_cmd curl
need_cmd tar
need_cmd file
mkdir -p "${OUT_DIR}"
WORK="${PHOTOS_RKNN_WORK:-${OUT_DIR}/_work}"
mkdir -p "${WORK}"

CONCEPT_TFLITE=""
DETECTION_ONNX=""
FEATURE_ONNX=""
FEATURE_PB=""

fetch_librknnrt
if [ "${SKIP_CONCEPT}" != 1 ]; then
  extract_concept
fi
fetch_face_models

if [ "${INSTALL_DEPS}" = 1 ]; then
  install_python_deps
fi

if [ "${SKIP_CONVERT}" = 1 ]; then
  log "skip convert; bits in ${OUT_DIR} / ${WORK}"
  exit 0
fi

run_convert

[ -f "${OUT_DIR}/librknnrt.so" ] || die "librknnrt.so not in ${OUT_DIR}"
if [ "${SKIP_CONCEPT}" != 1 ]; then
  [ -f "${OUT_DIR}/concept_network.rknn" ] || die "concept_network.rknn missing"
fi
if [ "${SKIP_DETECTION}" != 1 ]; then
  [ -f "${OUT_DIR}/detection_network.rknn" ] || die "detection_network.rknn missing"
fi
if [ "${SKIP_FEATURE}" != 1 ]; then
  [ -f "${OUT_DIR}/feature_network.rknn" ] || die "feature_network.rknn missing"
fi

{
  echo "photos_version=${PHOTOS_VERSION}"
  echo "platform=${PLATFORM}"
  echo "rknn_toolkit=${RKNN_TOOLKIT_VER}"
  echo "librknnrt=$(stat -c%s "${OUT_DIR}/librknnrt.so")"
  ls -1 "${OUT_DIR}"/*.rknn 2>/dev/null | while read -r f; do
    echo "$(basename "$f")=$(stat -c%s "$f")"
  done
} >"${OUT_DIR}/MANIFEST.txt"

log "done: ${OUT_DIR}"
cat "${OUT_DIR}/MANIFEST.txt"
