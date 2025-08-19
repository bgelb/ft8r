#!/usr/bin/env bash
set -euo pipefail

# Install kiwirecorder locally under this app directory.
# - Clones kiwiclient into vendor/
# - Installs Python deps into current environment (use your venv)
# - Creates bin/kiwirecorder.py wrapper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR"
VENDOR_DIR="$APP_DIR/vendor"
BIN_DIR="$APP_DIR/bin"

echo "[kiwirecorder] Installing under $APP_DIR"
mkdir -p "$VENDOR_DIR" "$BIN_DIR"

REPO_URL="https://github.com/jks-prv/kiwiclient.git"
TARGET_DIR="$VENDOR_DIR/kiwiclient"

if [[ -d "$TARGET_DIR/.git" ]]; then
  echo "[kiwirecorder] Updating existing repo ..."
  git -C "$TARGET_DIR" fetch --depth=1 origin
  git -C "$TARGET_DIR" reset --hard origin/master
else
  echo "[kiwirecorder] Cloning $REPO_URL ..."
  git clone --depth=1 "$REPO_URL" "$TARGET_DIR"
fi

REQ_FILE="$TARGET_DIR/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
  echo "[kiwirecorder] Installing Python requirements (use your venv) ..."
  python3 -m pip install --upgrade pip >/dev/null
  python3 -m pip install -r "$REQ_FILE"
else
  echo "[kiwirecorder] WARNING: requirements.txt not found; skipping pip install"
fi

WRAPPER="$BIN_DIR/kiwirecorder.py"
cat > "$WRAPPER" << 'WRAP'
#!/usr/bin/env bash
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$THIS_DIR/.." && pwd)"
exec python3 "$APP_DIR/vendor/kiwiclient/kiwirecorder.py" "$@"
WRAP
chmod +x "$WRAPPER"

echo "[kiwirecorder] Installed wrapper: $WRAPPER"
echo "[kiwirecorder] Done. The app will auto-detect this local binary."

