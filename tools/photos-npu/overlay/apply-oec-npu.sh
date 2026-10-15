#!/bin/sh
# apply-oec-npu.sh — OEC RK3566 Photos NPU overlay installer (v2).
# Run as ROOT on the box, from the extracted bundle dir:  sudo sh apply-oec-npu.sh
# Idempotent: safe to re-run after a Photos package upgrade.
set -eu
SRC="$(cd "$(dirname "$0")" && pwd)"
PKG="/var/packages/SynologyPhotos"
NPU="${PKG}/target/npu"
BIN="${PKG}/target/usr/bin"

id -u | grep -qx 0 || { echo "ERROR: run as root (sudo sh apply-oec-npu.sh)"; exit 1; }
[ -d "${NPU}" ] || { echo "ERROR: official SynologyPhotos is not installed"; exit 1; }

echo "== 1/5 backup stock npu payload (once) =="
BK="/volume1/@tmp/npu-stock-backup"
if [ ! -d "${BK}" ]; then
  mkdir -p "${BK}"
  cp -af "${NPU}/." "${BK}/"
  echo "   stock payload backed up to ${BK}"
else
  echo "   backup already exists: ${BK}"
fi

echo "== 2/5 install npu payload =="
install -m 0755 "${SRC}/npu/npu_server"           "${NPU}/npu_server"
install -m 0755 "${SRC}/npu/run_server.sh"        "${NPU}/run_server.sh"
install -m 0644 "${SRC}/npu/npu_model_conf.json"  "${NPU}/npu_model_conf.json"
install -m 0644 "${SRC}/npu/99-oec-galcore.rules" "${NPU}/99-oec-galcore.rules"
install -m 0644 "${SRC}/npu/OEC_NPU"              "${NPU}/OEC_NPU"
install -m 0644 "${SRC}/npu/asset/labels.txt"     "${NPU}/asset/labels.txt"
install -m 0644 "${SRC}/npu/asset/thresholds.txt" "${NPU}/asset/thresholds.txt"
for m in concept_network.rknn detection_network.rknn feature_network.rknn; do
  install -m 0644 "${SRC}/npu/asset/network/${m}" "${NPU}/asset/network/${m}"
done
printf 'oec-rknn\n' >"${NPU}/asset/network/VERSION"
install -m 0755 "${SRC}/npu/lib_arm64/librknnrt.so" "${NPU}/lib_arm64/librknnrt.so"

echo "== 3/5 hook + package wiring =="
install -m 0755 "${SRC}/oec-enable-npu.sh" "${PKG}/scripts/oec-enable-npu.sh"
if ! grep -q oec-enable-npu "${PKG}/scripts/postinst"; then
  printf '\n# OEC RK3566 NPU bridge\n/bin/sh %s/scripts/oec-enable-npu.sh || true\n' "${PKG}" \
    >>"${PKG}/scripts/postinst"
fi
if ! grep -q oec-enable-npu "${PKG}/scripts/start-stop-status"; then
  awk -v hook="/bin/sh ${PKG}/scripts/oec-enable-npu.sh || true" '
    $0 ~ /^  start\)$/ { print; print "    " hook; next }
    { print }
  ' "${PKG}/scripts/start-stop-status" >"${PKG}/scripts/start-stop-status.oec"
  mv -f "${PKG}/scripts/start-stop-status.oec" "${PKG}/scripts/start-stop-status"
  chmod 0755 "${PKG}/scripts/start-stop-status"
fi

echo "== 4/5 enable person/child concept themes =="
PSQL="psql -h /run/synofoto -U synofoto -d synofoto"
# official ships person/child with confidence_threshold 2.0 (impossible) in the
# versioned concept_threshold table — lower to 0.5 so person photos get themes
${PSQL} -c "UPDATE concept_threshold SET confidence_threshold=0.5 WHERE id_concept IN (SELECT id FROM concept WHERE stem IN ('person','child')) AND confidence_threshold > 1" || true
${PSQL} -c "UPDATE concept SET hidden=false, display_threshold=1 WHERE stem IN ('person','child')" || true
"${BIN}/synofoto-bin-regenerate-many-unit-has-many-concept" --all-users || true
"${BIN}/synofoto-bin-recount-concept-album-item-count" --all-users || true

echo "== 5/5 restart npu_server =="
ps | grep npu_server | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 1
/bin/sh "${PKG}/scripts/oec-enable-npu.sh"
sleep 2
if ps | grep npu_server | grep -v grep >/dev/null 2>&1; then
  echo "DONE. npu_server running; log: ${PKG}/var/npu_server.log"
else
  echo "ERROR: npu_server did not start; check ${PKG}/var/npu_server.log"
  exit 1
fi
