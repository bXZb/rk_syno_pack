#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper: the actual packing lives in pack-raw-disk-img.py, which
# builds the vendor-compatible layout (env @144MB, boot FAT as GPT
# partition #7, stock vendor GPT entry table). See that file for details.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/pack-raw-disk-img.py" "$@"
