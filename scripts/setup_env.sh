#!/usr/bin/env bash
set -euo pipefail

# Portable local environment setup for WSJT-X + Python deps (macOS & Linux)

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
WSJTX_DIR="$ROOT_DIR/.wsjtx"
BIN_DIR="$WSJTX_DIR/bin"
VENV_DIR="$ROOT_DIR/.venv"
TMP_DIR="$ROOT_DIR/.tmp"
SAMPLES_DST_DIR="$ROOT_DIR/ft8_lib-2.0/test/wav"
WSJTX_VERSION="2.7.0"

mkdir -p "$BIN_DIR"

log() { printf "[setup] %s\n" "$*"; }

os_name() { uname -s; }
arch_name() { uname -m; }

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' not found. Please install it and re-run." >&2
    exit 1
  fi
}

# Find a Python interpreter >= 3.11 for creating the venv.
# Honors explicit override via FT8R_PYTHON if set.
find_python311() {
  local override
  override="${FT8R_PYTHON:-}"
  if [[ -n "$override" ]]; then
    if ! command -v "$override" >/dev/null 2>&1; then
      echo "FT8R_PYTHON is set to '$override' but it is not executable or not on PATH" >&2
      return 1
    fi
    if "$override" -c 'import sys; import sys; sys.exit(0 if sys.version_info[:2] >= (3,11) else 1)' 2>/dev/null; then
      echo "$override"
      return 0
    else
      echo "FT8R_PYTHON ('$override') is too old; need Python >= 3.11" >&2
      return 1
    fi
  fi

  # Prefer the newest versions first if available on PATH.
  local candidates=(
    python3.13
    python3.12
    python3.11
    python3
    python
  )
  local c
  for c in "${candidates[@]}"; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; import sys; sys.exit(0 if sys.version_info[:2] >= (3,11) else 1)' 2>/dev/null; then
        echo "$c"
        return 0
      fi
    fi
  done

  return 1
}

download() {
  local url="$1" out="$2"
  log "Downloading $url"
  curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
}

extract_zip() {
  local zip="$1" dest="$2"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$zip" -d "$dest"
  else
    ensure_cmd python3
    python3 - "$zip" "$dest" <<'PY'
import sys, zipfile, os
zip_path, dest = sys.argv[1], sys.argv[2]
os.makedirs(dest, exist_ok=True)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(dest)
PY
  fi
}

setup_macos() {
  mkdir -p "$WSJTX_DIR" "$BIN_DIR"
  ensure_cmd hdiutil
  local dmg="$WSJTX_DIR/wsjtx.dmg"
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
  local attach_out
  attach_out="$(hdiutil attach "$dmg" -nobrowse | tail -n 1 || true)"
  local MNT
  MNT="${attach_out##*\t}"
  if [[ -z "$MNT" || ! -d "$MNT" ]]; then
    if [[ -d "/Volumes/WSJT-X" ]]; then
      MNT="/Volumes/WSJT-X"
    else
      echo "Failed to determine DMG mountpoint" >&2
      exit 1
    fi
  fi
  trap 'hdiutil detach "$MNT" -quiet || true' EXIT

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
  mkdir -p "$WSJTX_DIR" "$BIN_DIR"
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

  local usrbin="$extract_dir/usr/bin"
  local usrlibwsjtx="$extract_dir/usr/lib/wsjtx"
  mkdir -p "$BIN_DIR"
  for tool in jt9 ft8code ft8sim; do
    if [[ -x "$usrbin/$tool" ]]; then
      cat >"$BIN_DIR/$tool" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$usrlibwsjtx"):\${LD_LIBRARY_PATH:-}"
exec "$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$usrbin/$tool")" "\$@"
EOF
      chmod +x "$BIN_DIR/$tool"
    fi
  done
}

setup_python() {
  local PY311
  if ! PY311="$(find_python311)"; then
    echo "Could not find a Python >= 3.11 interpreter on PATH. Install python3.11+ or set FT8R_PYTHON to a suitable binary." >&2
    exit 1
  fi
  local ver
  ver="$($PY311 -c 'import platform; print(platform.python_version())' 2>/dev/null || echo unknown)"
  log "Using Python $ver at '$PY311'"
  log "Creating venv at $VENV_DIR"
  "$PY311" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r "$ROOT_DIR/requirements.txt"
  deactivate
}

fetch_samples() {
  local zip="$TMP_DIR/ft8_lib.zip"
  local outdir="$TMP_DIR/ft8_lib-master"
  mkdir -p "$TMP_DIR"
  log "Fetching ft8_lib sample dataset"
  download "https://codeload.github.com/kgoba/ft8_lib/zip/refs/heads/master" "$zip"
  rm -rf "$outdir"
  extract_zip "$zip" "$TMP_DIR"
  mkdir -p "$SAMPLES_DST_DIR"
  rsync -a "$outdir/test/wav/" "$SAMPLES_DST_DIR/"
  local count
  count=$(find "$SAMPLES_DST_DIR" -type f \( -name '*.wav' -o -name '*.txt' \) | wc -l | tr -d ' ')
  log "Installed sample files to $SAMPLES_DST_DIR ($count files)"
}

run_tests() {
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
  # Always use repo-local WSJT-X binaries for tests, regardless of the
  # caller's environment. This avoids accidental use of a global install.
  unset WSJTX_BIN_DIR || true
  export WSJTX_BIN_DIR="$BIN_DIR"
  # Enforce baseline FT8R defaults for CI/regression runs even if the
  # outer environment exported blank or conflicting values.
  export FT8R_WHITEN_ENABLE="${FT8R_WHITEN_ENABLE:-1}"
  export FT8R_COARSE_MODE="${FT8R_COARSE_MODE:-budget}"
  export FT8R_MAX_CANDIDATES="${FT8R_MAX_CANDIDATES:-1500}"
  export FT8R_MIN_LLR_AVG="${FT8R_MIN_LLR_AVG:-0.2}"
  log "Using WSJTX_BIN_DIR=$WSJTX_BIN_DIR"
  log "Running pytest (verbose, durations, show skip reasons)"
  pytest -vv -ra --durations=25
  deactivate || true
}

clean_all() {
  log "Cleaning generated artifacts"
  rm -rf "$VENV_DIR" "$WSJTX_DIR" "$TMP_DIR"
  # Only remove samples we fetched; keep parent structure
  rm -rf "$SAMPLES_DST_DIR"
}

main() {
  local do_clean=0 do_wsjt=1 do_python=1 do_samples=1 do_test=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --clean) do_clean=1 ;;
      --no-wsjt) do_wsjt=0 ;;
      --no-python) do_python=0 ;;
      --no-samples) do_samples=0 ;;
      --test) do_test=1 ;;
      --help|-h)
        cat <<USAGE
Usage: $0 [options]
  --clean        Remove venv, local WSJT-X, temp files, and fetched samples
  --no-wsjt      Skip WSJT-X binary download/setup
  --no-python    Skip Python venv creation/install
  --no-samples   Skip fetching ft8_lib sample dataset
  --test         Run pytest after setup (uses local venv)
  --help         Show this help

Default (no options): install WSJT-X (if supported), create venv, fetch samples.
USAGE
        exit 0 ;;
      *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
  done

  [[ $do_clean -eq 1 ]] && clean_all

  if [[ $do_wsjt -eq 1 ]]; then
    log "Setting up WSJT-X $WSJTX_VERSION locally under $WSJTX_DIR"
    case "$(os_name)" in
      Darwin) setup_macos ;;
      Linux) setup_linux ;;
      *) echo "Unsupported OS: $(os_name)" >&2; exit 1 ;;
    esac
    log "Binaries (if available) are in $BIN_DIR"
  else
    log "Skipping WSJT-X setup per --no-wsjt"
  fi

  [[ $do_python -eq 1 ]] && setup_python || log "Skipping Python setup per --no-python"
  [[ $do_samples -eq 1 ]] && fetch_samples || log "Skipping sample fetch per --no-samples"

  log "Activate venv with: source $VENV_DIR/bin/activate"
  log "WSJT-X binaries directory: $BIN_DIR (export WSJTX_BIN_DIR to override)"

  # If tests are requested, run them and honor their exit code so CI fails
  # when pytest reports failures or errors.
  if [[ $do_test -eq 1 ]]; then
    run_tests
  fi
}

main "$@"
