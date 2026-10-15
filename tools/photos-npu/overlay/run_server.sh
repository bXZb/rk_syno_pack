#!/bin/sh
# POSIX launcher (v4). DSM /bin/sh is ash — no bashisms.
# Works from the package payload dir and from the /usr/local/oec-npu
# relocation (BASEDIR derived from this script's own location).
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/var/packages/SynologyPhotos/var/npu_server.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
{
  echo "=== $(date) invoked as: $0 $* (uid=$(id -u))"
} >> "$LOG" 2>/dev/null || true
export LD_LIBRARY_PATH="$BASEDIR/lib_arm64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$BASEDIR" || exit 1

LISTEN="unix:///run/synofoto/npu-photo.sock"
case "${1:-}" in
  *://*|/*) LISTEN="$1"; shift ;;
esac

exec "$BASEDIR/npu_server" -dir "$BASEDIR" -listen "$LISTEN" "$@" >> "$LOG" 2>&1
