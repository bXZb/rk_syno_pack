#!/usr/bin/env bash
# Build the OEC RK3566 Photos NPU overlay bundle: npu payload (self-built
# npu_server + converted RKNN models) + v7 hook + on-box apply script.
# The bundle replaces files inside an OFFICIAL signed SynologyPhotos install
# (the unsigned-SPK route fails with synopkg error 319).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NPU_DIR="${PROJECT_DIR}/tools/photos-npu"
OVERLAY_DIR="${NPU_DIR}/overlay"
WORK="${PHOTOS_OVERLAY_WORK:-${PROJECT_DIR}/build/photos-overlay}"
OUT="${1:-${PROJECT_DIR}/output/dsm/photos-npu-overlay.tar.gz}"

log() { printf '[photos-overlay] %s\n' "$*"; }
die() { printf '[photos-overlay] error: %s\n' "$*" >&2; exit 1; }

NPU_SERVER="${WORK}/npu_server.aarch64"
if [ ! -f "${NPU_SERVER}" ]; then
  command -v go >/dev/null 2>&1 || die "missing go (or prebuild ${NPU_SERVER})"
  log "building npu_server (linux/arm64)"
  mkdir -p "${WORK}"
  ( cd "${NPU_DIR}" && \
    GOTOOLCHAIN=local GOOS=linux GOARCH=arm64 CGO_ENABLED=0 \
    go build -trimpath -ldflags="-s -w" -o "${NPU_SERVER}" ./cmd/npu_server )
fi

RKNN_DIR="${RKNN_DIR:?set RKNN_DIR to the prepare-photos-rknn.sh output dir}"
for f in librknnrt.so concept_network.rknn detection_network.rknn feature_network.rknn; do
  case "${f}" in
    *.rknn) [ -f "${RKNN_DIR}/${f}" ] || die "missing ${RKNN_DIR}/${f}" ;;
    *) [ -f "${RKNN_DIR}/${f}" ] || die "missing ${RKNN_DIR}/${f}" ;;
  esac
done

STAGE="${WORK}/bundle"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/npu/asset/network" "${STAGE}/npu/lib_arm64"

install -m 0755 "${NPU_SERVER}"                          "${STAGE}/npu/npu_server"
install -m 0755 "${OVERLAY_DIR}/run_server.sh"           "${STAGE}/npu/run_server.sh"
install -m 0644 "${OVERLAY_DIR}/npu_model_conf.json"     "${STAGE}/npu/npu_model_conf.json"
install -m 0644 "${OVERLAY_DIR}/99-oec-galcore.rules"    "${STAGE}/npu/99-oec-galcore.rules"
printf 'rk3566 npu_server\n' >"${STAGE}/npu/OEC_NPU"
install -m 0644 "${OVERLAY_DIR}/asset/labels.txt"        "${STAGE}/npu/asset/labels.txt"
install -m 0644 "${OVERLAY_DIR}/asset/thresholds.txt"    "${STAGE}/npu/asset/thresholds.txt"
for m in concept_network.rknn detection_network.rknn feature_network.rknn; do
  install -m 0644 "${RKNN_DIR}/${m}" "${STAGE}/npu/asset/network/${m}"
done
printf 'oec-rknn\n' >"${STAGE}/npu/asset/network/VERSION"
install -m 0755 "${RKNN_DIR}/librknnrt.so" "${STAGE}/npu/lib_arm64/librknnrt.so"

install -m 0755 "${OVERLAY_DIR}/oec-enable-npu.sh" "${STAGE}/oec-enable-npu.sh"
install -m 0755 "${OVERLAY_DIR}/apply-oec-npu.sh"  "${STAGE}/apply-oec-npu.sh"

cat >"${STAGE}/README.txt" <<'EOF'
OEC RK3566 Synology Photos NPU overlay
======================================
Prereq: official (signed) SynologyPhotos installed and running.

Install (as root, from this directory):
    sudo sh apply-oec-npu.sh

What it does: swaps the npu payload (npu_server + RKNN models + librknnrt),
installs the oec-enable-npu.sh hook into package scripts (called from postinst
and start-stop-status), lowers the official person/child concept thresholds
(2.0 = impossible) to 0.5 so person photos get a theme, regenerates concept
associations, and restarts npu_server unconfined from /usr/local/oec-npu.

Re-run after any SynologyPhotos package upgrade.
EOF

mkdir -p "$(dirname "${OUT}")"
OUT="$(cd "$(dirname "${OUT}")" && pwd)/$(basename "${OUT}")"
( cd "${STAGE}" && tar -czf "${OUT}" . )
log "bundle: ${OUT} ($(du -h "${OUT}" | cut -f1))"
tar -tzf "${OUT}"
