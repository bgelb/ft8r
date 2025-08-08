#!/usr/bin/env bash
set -euo pipefail

# Portable local environment setup for WSJT-X + Python deps (macOS & Linux)

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
WSJTX_DIR="$ROOT_DIR/.wsjtx"
BIN_DIR="$WSJTX_DIR/bin"
VENV_DIR="$ROOT_DIR/.venv"
WSJTX_VERSION="2.7.0"

mkdir -p "$BIN_DIR"

log() { printf "[setup] %s\n" "$*"; }

os_name() {
  uname -s
}

arch_name() {
  uname -m
}

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' not found. Please install it and re-run." >&2
    exit 1
  fi
}

download() {
  local url="$1" out="$2"
  log "Downloading $url"
  curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
}

setup_macos() {
  ensure_cmd hdiutil
  local dmg="$WSJTX_DIR/wsjtx.dmg"
  # Try common dmg names
  local base_url="https://sourceforge.net/projects/wsjt/files/wsjtx-$WSJTX_VERSION/"
  local candidates=(
    "WSJT-X-$WSJTX_VERSION-Darwin.dmg/download"
    "wsjtx-$WSJTX_VERSION-Darwin.dmg/download"
  )

  for name in "${candidates[@]}"; do
    if download "$base_url$name" "$dmg"; then
      break
    fi
  done

  log "Mounting DMG"
  # Use default mountpoint chosen by hdiutil and discover it.
  local attach_out
  attach_out="$(hdiutil attach "$dmg" -nobrowse | tail -n 1 || true)"
  local MNT
  MNT="${attach_out##*\t}"
  if [[ -z "$MNT" || ! -d "$MNT" ]]; then
    # Fallback to common volume name
    if [[ -d "/Volumes/WSJT-X" ]]; then
      MNT="/Volumes/WSJT-X"
    else
      echo "Failed to determine DMG mountpoint" >&2
      exit 1
    fi
  fi
  trap 'hdiutil detach "$MNT" -quiet || true' EXIT

  # Copy the app bundle locally (handle "WSJT-X.app" or "wsjtx.app")
  local app_src
  app_src="$(find "$MNT" -maxdepth 1 -type d \( -iname 'WSJT-X.app' -o -iname 'wsjtx.app' \) -print -quit)"
  if [[ -z "$app_src" ]]; then
    echo "WSJT-X app bundle not found in mounted DMG" >&2
    exit 1
  fi
  log "Copying app bundle"
  rsync -a "$app_src" "$WSJTX_DIR/"

  local app_mac_bin="$WSJTX_DIR/WSJT-X.app/Contents/MacOS"
  if [[ ! -d "$app_mac_bin" ]]; then
    app_mac_bin="$WSJTX_DIR/wsjtx.app/Contents/MacOS"
  fi
  for tool in jt9 ft8code ft8sim; do
    if [[ -x "$app_mac_bin/$tool" ]]; then
      ln -sf "$app_mac_bin/$tool" "$BIN_DIR/$tool"
    fi
  done

  hdiutil detach "$MNT" -quiet || true
  trap - EXIT
}

setup_linux() {
  ensure_cmd dpkg-deb
  local deb="$WSJTX_DIR/wsjtx.deb"
  local arch
  arch="$(arch_name)"
  local deb_arch
  case "$arch" in
    x86_64) deb_arch=amd64 ;;
    aarch64) deb_arch=arm64 ;;
    armv7l|armhf) deb_arch=armhf ;;
    *) echo "Unsupported Linux arch: $arch" >&2; exit 1 ;;
  esac
  local url="https://sourceforge.net/projects/wsjt/files/wsjtx-$WSJTX_VERSION/wsjtx_${WSJTX_VERSION}_${deb_arch}.deb/download"
  download "$url" "$deb"

  local extract_dir="$WSJTX_DIR/linux-pkg"
  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  log "Extracting DEB"
  dpkg-deb -x "$deb" "$extract_dir"

  # Link binaries and wrap them to set LD_LIBRARY_PATH to local libs
  local usrbin="$extract_dir/usr/bin"
  local usrlibwsjtx="$extract_dir/usr/lib/wsjtx"
  mkdir -p "$BIN_DIR"
  for tool in jt9 ft8code ft8sim; do
    if [[ -x "$usrbin/$tool" ]]; then
      cat >"$BIN_DIR/$tool" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$usrlibwsjtx"):
${LD_LIBRARY_PATH:-}"
exec "$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$usrbin/$tool")" "$@"
EOF
      chmod +x "$BIN_DIR/$tool"
    fi
  done
}

setup_python() {
  ensure_cmd python3
  log "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r "$ROOT_DIR/requirements.txt"
  deactivate
}

main() {
  log "Setting up WSJT-X $WSJTX_VERSION locally under $WSJTX_DIR"
  case "$(os_name)" in
    Darwin) setup_macos ;;
    Linux) setup_linux ;;
    *) echo "Unsupported OS: $(os_name)" >&2; exit 1 ;;
  esac

  setup_python
  log "Done. Binaries (if available) are in $BIN_DIR"
  log "Activate venv with: source $VENV_DIR/bin/activate"
  log "Optionally set WSJTX_BIN_DIR=$BIN_DIR to force binary discovery"
}

main "$@"


