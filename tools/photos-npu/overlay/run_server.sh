#!/bin/bash
# Replacement launcher: same argv contract as the official rtd1619b script.
# Photos forks this from /var/packages/SynologyPhotos/target/npu/.

SCRIPT=$(realpath "$0")
BASEDIR=$(dirname "$SCRIPT")
export LD_LIBRARY_PATH="$BASEDIR/lib_arm64:${LD_LIBRARY_PATH:-}"
cd "$BASEDIR" || exit 1

LISTEN="unix:///run/synofoto/npu-photo.sock"
if [ -n "${1:-}" ] && [[ "$1" == *://* || "$1" == /* ]]; then
	LISTEN="$1"
	shift
fi

"$BASEDIR/npu_server" -dir "$BASEDIR" -listen "$LISTEN" "$@" 2>&1 | logger -p err -s -t "$(basename "$0")"
