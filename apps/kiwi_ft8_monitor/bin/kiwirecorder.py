#!/usr/bin/env bash
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$THIS_DIR/.." && pwd)"
exec python3 "$APP_DIR/vendor/kiwiclient/kiwirecorder.py" "$@"
