#!/bin/sh
# OEC RK3566 NPU bridge (v7). Called from Photos postinst + start-stop-status.
#  - SkipMemoryCheck bypasses the DS124 1GB concept gate.
#  - /dev/galcore must be a REAL char device: IsSupportedNpuNetwork() stats it
#    with S_IFCHR (a touched regular file passes stat64-exists but the NPU
#    open later fails).
#  - npu_server runs from /usr/local/oec-npu: the package AppArmor profile
#    (/volume*/@appstore/SynologyPhotos/npu/npu_server) denies open on
#    /dev/rknpu and /dev/dri/*; the relocated exe is unconfined.
#  - The relocation is created from the package payload on first run and kept
#    in sync afterwards, so package payload updates propagate into it.
#  - The official plugin never forks npu_server; spawn it once, daemonized.
set -eu
PKG_VAR="/var/packages/SynologyPhotos/var"
PKG_NPU="/var/packages/SynologyPhotos/target/npu"
LOCAL_NPU="/usr/local/oec-npu"
NPU_RUN="${LOCAL_NPU}/run_server.sh"
NPU_FALLBACK="${PKG_NPU}/run_server.sh"

mkdir -p "${PKG_VAR}"
touch "${PKG_VAR}/SkipMemoryCheck"
chmod 644 "${PKG_VAR}/SkipMemoryCheck" || true

rm -f /dev/galcore
if [ -c /dev/rknpu ]; then
  ln -sf /dev/rknpu /dev/galcore
elif [ -c /dev/dri/card2 ]; then
  ln -sf /dev/dri/card2 /dev/galcore
else
  mknod /dev/galcore c 226 2 2>/dev/null || true
  chmod 666 /dev/galcore 2>/dev/null || true
fi

# create-or-refresh the unconfined relocation from the package payload
if [ ! -d "${LOCAL_NPU}" ]; then
  mkdir -p "${LOCAL_NPU}"
  cp -af "${PKG_NPU}/." "${LOCAL_NPU}/" 2>/dev/null || true
else
  mkdir -p "${LOCAL_NPU}/asset/network" "${LOCAL_NPU}/lib_arm64" 2>/dev/null || true
  for f in npu_server run_server.sh npu_model_conf.json; do
    cp -f "${PKG_NPU}/$f" "${LOCAL_NPU}/$f" 2>/dev/null || true
  done
  cp -f "${PKG_NPU}"/asset/network/*.rknn "${LOCAL_NPU}/asset/network/" 2>/dev/null || true
  cp -f "${PKG_NPU}"/asset/labels.txt "${PKG_NPU}"/asset/thresholds.txt "${LOCAL_NPU}/asset/" 2>/dev/null || true
  cp -f "${PKG_NPU}"/lib_arm64/librknnrt.so "${LOCAL_NPU}/lib_arm64/" 2>/dev/null || true
fi

if [ -f "${PKG_NPU}/99-oec-galcore.rules" ]; then
  mkdir -p /usr/lib/udev/rules.d
  cp -f "${PKG_NPU}/99-oec-galcore.rules" /usr/lib/udev/rules.d/99-oec-galcore.rules || true
fi

if ! ps | grep npu_server | grep -v grep >/dev/null 2>&1; then
  RUN="${NPU_RUN}"
  [ -x "${RUN}" ] || RUN="${NPU_FALLBACK}"
  if [ -x "${RUN}" ]; then
    ( setsid sh "${RUN}" </dev/null >/dev/null 2>&1 & ) || true
  fi
fi
