#!/usr/bin/env bash
# Decrypt the official rtd1619b Synology Photos SPK, swap in the RK3566
# npu_server, and emit an unsigned sideload SPK.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/pack-photos-spk.sh [options]

Options:
  --spk PATH           Official SynologyPhotos-rtd1619b SPK (downloaded if omitted)
  --npu-server PATH    Prebuilt aarch64 npu_server (built if omitted)
  --rknn-dir DIR       Extra .rknn / librknnrt.so files copied into npu/
  --photos-version VER SPK version in the download URL (default 1.9.1-10928)
  --skip-build         Do not compile npu_server; require --npu-server
  --output PATH        Output SPK path
  -h, --help

The result is a classic (unencrypted) tar SPK installable with:
  synopkg install SynologyPhotos-rtd1619b-*-oec-npu.spk
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NPU_DIR="${PROJECT_DIR}/tools/photos-npu"
SYNOXTRACT="${PROJECT_DIR}/tools/SynoXtract/synoxtract"
WORK="${PHOTOS_SPK_WORK:-${PROJECT_DIR}/build/photos-spk}"
PHOTOS_VERSION="${PHOTOS_VERSION:-1.9.1-10928}"

SPK_FILE=""
NPU_SERVER=""
RKNN_DIR=""
SKIP_BUILD=0
OUTPUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --spk) SPK_FILE="$2"; shift 2 ;;
    --npu-server) NPU_SERVER="$2"; shift 2 ;;
    --rknn-dir) RKNN_DIR="$2"; shift 2 ;;
    --photos-version) PHOTOS_VERSION="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[photos-spk] %s\n' "$*"; }
die() { printf '[photos-spk] error: %s\n' "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

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

ensure_npu_server() {
  if [ -n "${NPU_SERVER}" ]; then
    [ -f "${NPU_SERVER}" ] || die "npu_server not found: ${NPU_SERVER}"
    return 0
  fi
  NPU_SERVER="${WORK}/npu_server.aarch64"
  if [ "${SKIP_BUILD}" = 1 ]; then
    [ -f "${NPU_SERVER}" ] || die "--skip-build set but ${NPU_SERVER} is missing"
    return 0
  fi
  need_cmd go
  log "building npu_server (linux/arm64)"
  mkdir -p "${WORK}"
  (
    cd "${NPU_DIR}"
    GOTOOLCHAIN=local GOOS=linux GOARCH=arm64 CGO_ENABLED=0 \
      go build -trimpath -ldflags="-s -w" -o "${NPU_SERVER}" ./cmd/npu_server
  )
}

write_hook() {
  local dest="$1"
  cat >"${dest}" <<'HOOK'
#!/bin/sh
# Called from Photos postinst / start-stop-status on OEC RK3566.
# 1) Bypass the DS124 1GB concept memory gate.
# 2) Create /dev/galcore so IsSupportedNpuNetwork() succeeds.
set -eu
PKG_VAR="/var/packages/SynologyPhotos/var"
PKG_NPU="/var/packages/SynologyPhotos/target/npu"
mkdir -p "${PKG_VAR}"
touch "${PKG_VAR}/SkipMemoryCheck"
chmod 644 "${PKG_VAR}/SkipMemoryCheck" || true

if [ ! -e /dev/galcore ]; then
  if [ -e /dev/rknpu ]; then
    ln -sf /dev/rknpu /dev/galcore
  else
    # stat64() only checks existence
    touch /dev/galcore
    chmod 666 /dev/galcore || true
  fi
fi

if [ -f "${PKG_NPU}/99-oec-galcore.rules" ]; then
  mkdir -p /usr/lib/udev/rules.d
  cp -f "${PKG_NPU}/99-oec-galcore.rules" /usr/lib/udev/rules.d/99-oec-galcore.rules || true
fi
HOOK
  chmod 0755 "${dest}"
}

patch_scripts() {
  local scripts="$1"
  write_hook "${scripts}/oec-enable-npu.sh"

  if ! grep -q oec-enable-npu "${scripts}/postinst"; then
    printf '\n# OEC RK3566 NPU bridge\n/bin/sh /var/packages/SynologyPhotos/scripts/oec-enable-npu.sh || true\n' \
      >>"${scripts}/postinst"
  fi

  if grep -q oec-enable-npu "${scripts}/start-stop-status"; then
    return 0
  fi
  # Insert the hook at the start of the "start)" arm.
  awk '
    $0 ~ /^  start\)$/ { print; print "    /bin/sh /var/packages/SynologyPhotos/scripts/oec-enable-npu.sh || true"; next }
    { print }
  ' "${scripts}/start-stop-status" >"${scripts}/start-stop-status.oec"
  mv -f "${scripts}/start-stop-status.oec" "${scripts}/start-stop-status"
  chmod 0755 "${scripts}/start-stop-status"
}

overlay_npu() {
  local npu="$1"
  rm -rf "${npu}/lib_arm64"
  rm -f "${npu}/npu_server" "${npu}/asset/network/"*.nb
  mkdir -p "${npu}/lib_arm64" "${npu}/asset/network"

  install -m 0755 "${NPU_SERVER}" "${npu}/npu_server"
  install -m 0755 "${NPU_DIR}/overlay/run_server.sh" "${npu}/run_server.sh"
  install -m 0644 "${NPU_DIR}/overlay/99-oec-galcore.rules" "${npu}/99-oec-galcore.rules"
  install -m 0644 "${NPU_DIR}/overlay/asset/labels.txt" "${npu}/asset/labels.txt"
  install -m 0644 "${NPU_DIR}/overlay/asset/thresholds.txt" "${npu}/asset/thresholds.txt"
  printf 'oec-rknn\n' >"${npu}/asset/network/VERSION"
  printf 'rk3566 npu_server\n' >"${npu}/OEC_NPU"

  if [ -n "${RKNN_DIR}" ]; then
    [ -d "${RKNN_DIR}" ] || die "rknn dir not found: ${RKNN_DIR}"
    find "${RKNN_DIR}" -maxdepth 2 -type f \( \
      -name '*.rknn' -o -name 'librknnrt.so*' -o -name 'labels.txt' -o -name 'thresholds.txt' \
    \) -print | while IFS= read -r f; do
      case "$(basename "$f")" in
        librknnrt.so*) install -m 0755 "$f" "${npu}/lib_arm64/$(basename "$f")" ;;
        *.rknn) install -m 0644 "$f" "${npu}/asset/network/$(basename "$f")" ;;
        labels.txt|thresholds.txt) install -m 0644 "$f" "${npu}/asset/$(basename "$f")" ;;
      esac
    done
  fi
}

update_info() {
  local info="$1"
  local extract_kb="$2"
  local tmp="${info}.oec"
  # Keep package=SynologyPhotos and arch=rtd1619b so DS124 accepts the SPK.
  awk -v kb="${extract_kb}" '
    BEGIN { done_extract=0; done_desc=0 }
    /^extractsize=/ { print "extractsize=\"" kb "\""; done_extract=1; next }
    /^description=/ && done_desc==0 {
      print "description=\"Synology Photos with OEC RK3566 npu_server.\""
      done_desc=1
      next
    }
    { print }
    END {
      if (!done_extract) print "extractsize=\"" kb "\""
    }
  ' "${info}" >"${tmp}"
  mv -f "${tmp}" "${info}"
}

SPK_URL_DEFAULT="https://global.synologydownload.com/download/Package/spk/SynologyPhotos/${PHOTOS_VERSION}/SynologyPhotos-rtd1619b-${PHOTOS_VERSION}.spk"

need_cmd tar
need_cmd xz
need_cmd curl
need_cmd file
ensure_synoxtract
mkdir -p "${WORK}"
ensure_npu_server

if [ -z "${SPK_FILE}" ]; then
  SPK_FILE="${WORK}/SynologyPhotos-rtd1619b-${PHOTOS_VERSION}.spk"
  if [ ! -f "${SPK_FILE}" ]; then
    log "downloading ${SPK_URL_DEFAULT}"
    curl -L --fail --retry 3 -o "${SPK_FILE}" "${SPK_URL_DEFAULT}"
  else
    log "using cached ${SPK_FILE}"
  fi
fi
[ -f "${SPK_FILE}" ] || die "SPK not found: ${SPK_FILE}"

EXTRACT="${WORK}/extract"
PKG="${WORK}/package"
STAGE="${WORK}/spk"
rm -rf "${EXTRACT}" "${PKG}" "${STAGE}"
mkdir -p "${EXTRACT}" "${PKG}" "${STAGE}"

log "decrypting $(basename "${SPK_FILE}")"
"${SYNOXTRACT}" -i "${SPK_FILE}" -d "${EXTRACT}"
[ -f "${EXTRACT}/package.tgz" ] || die "synoxtract did not produce package.tgz"
[ -f "${EXTRACT}/INFO" ] || die "synoxtract did not produce INFO"

log "unpacking package.tgz"
FMT="$(file -b "${EXTRACT}/package.tgz")"
case "${FMT}" in
  *XZ*) tar -xJf "${EXTRACT}/package.tgz" -C "${PKG}" ;;
  *gzip*) tar -xzf "${EXTRACT}/package.tgz" -C "${PKG}" ;;
  *bzip2*) tar -xjf "${EXTRACT}/package.tgz" -C "${PKG}" ;;
  *) tar -xf "${EXTRACT}/package.tgz" -C "${PKG}" ;;
esac
[ -d "${PKG}/npu" ] || die "package.tgz has no npu/ directory (wrong arch?)"

log "overlaying RK3566 npu_server"
overlay_npu "${PKG}/npu"

# Copy extracted SPK metadata (scripts/conf/icons) then patch scripts.
cp -a "${EXTRACT}/INFO" "${STAGE}/INFO"
cp -a "${EXTRACT}/scripts" "${STAGE}/scripts"
cp -a "${EXTRACT}/conf" "${STAGE}/conf"
cp -a "${EXTRACT}/WIZARD_UIFILES" "${STAGE}/WIZARD_UIFILES"
cp -a "${EXTRACT}/PACKAGE_ICON.PNG" "${STAGE}/PACKAGE_ICON.PNG"
cp -a "${EXTRACT}/PACKAGE_ICON_256.PNG" "${STAGE}/PACKAGE_ICON_256.PNG"
patch_scripts "${STAGE}/scripts"

log "repacking package.tgz"
(
  cd "${PKG}"
  tar -cf - . | xz -T0 -9 >"${STAGE}/package.tgz"
)

extract_kb="$(du -sk "${PKG}" | awk '{print $1}')"
update_info "${STAGE}/INFO" "${extract_kb}"

if [ -z "${OUTPUT}" ]; then
  OUTPUT="${PROJECT_DIR}/output/dsm/SynologyPhotos-rtd1619b-${PHOTOS_VERSION}-oec-npu.spk"
fi
mkdir -p "$(dirname "${OUTPUT}")"
rm -f "${OUTPUT}"
(
  cd "${STAGE}"
  tar -cf "${OUTPUT}" INFO package.tgz scripts conf WIZARD_UIFILES PACKAGE_ICON.PNG PACKAGE_ICON_256.PNG
)

log "done: ${OUTPUT}"
log "npu_server $(file -b "${PKG}/npu/npu_server")"
log "package extractsize=${extract_kb} KiB"
if ! find "${PKG}/npu/asset/network" -name '*.rknn' | grep -q .; then
  log "note: no .rknn models packed; add --rknn-dir or drop them next to npu/asset/network/"
fi
if [ ! -e "${PKG}/npu/lib_arm64/librknnrt.so" ]; then
  log "note: librknnrt.so not packed; copy it into --rknn-dir for on-device inference"
fi
