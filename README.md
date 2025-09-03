# ft8r

This repository explores building an FT8 demodulator. The test suite can run fully self-contained using a local WSJT-X binary distribution and a Python virtual environment — no system-wide installation required.

Optional runtime features (env toggles):
- Coarse whitening (improves contrast in busy bands)
  - Enabled by default; set `FT8R_WHITEN_ENABLE=0` to disable
  - `FT8R_WHITEN_MODE=tile|global` (default `tile`); `tile` uses robust per‑tile scaling (median + α·MAD)
- Coarse candidate selection
  - `FT8R_COARSE_MODE=budget|peak` (default `budget`); `budget` distributes picks per tile up to `FT8R_MAX_CANDIDATES`
- Light microsearch (single‑pass; small frequency nudges on CRC failure)
  - Configured via `FT8R_MICRO_LIGHT_DF_SPAN` (default `1.0` Hz) and `FT8R_MICRO_LIGHT_DF_STEP` (default `0.5` Hz)

### Quick start

One command to fully prepare the environment (WSJT-X, Python venv, sample dataset) and optionally run tests:

```bash
./scripts/setup_env.sh --test
```

What this does:
- Downloads WSJT-X locally into `.wsjtx/` and exposes CLI tools (e.g., `jt9`, `ft8code`, optionally `ft8sim`) via `.wsjtx/bin/`.
- Creates a Python virtual environment at `.venv/` and installs `requirements.txt`.
- Fetches the FT8 sample dataset from `kgoba/ft8_lib` into `ft8_lib-2.0/test/wav` used by the test suite.
- Runs the test suite (`pytest -q`).

To work interactively after setup:

```bash
source .venv/bin/activate
pytest -q
```

Notes
- A small subset of tests require `ft8sim` (used only to synthesize FT8 test audio). If `ft8sim` is not present in the WSJT-X binary package for your platform, those tests are automatically skipped. All other tests still run, including those using the bundled `ft8_lib-2.0/test/wav` sample files and the `jt9` decoder.
- If you already have WSJT-X installed elsewhere, you can point the test suite at its CLI tools by setting `WSJTX_BIN_DIR` to the directory containing `jt9`, `ft8code`, and `ft8sim`.
- To only set up without running tests, omit `--test`.

### Maintenance

The setup script supports convenient flags:

```bash
# Remove venv, local WSJT-X, temp files, and fetched samples
./scripts/setup_env.sh --clean

# Install everything but skip WSJT-X or dataset fetch
./scripts/setup_env.sh --no-wsjt --no-samples

# Run tests after setup
./scripts/setup_env.sh --test
```

### Platform support
- macOS: Downloads and mounts the WSJT-X `.dmg`, keeps the app bundle under `.wsjtx/`, and links CLI tools from inside the bundle.
- Linux: Downloads the appropriate `.deb` and extracts it locally (no root needed). Wrapper scripts set `LD_LIBRARY_PATH` so the CLI tools can run from the extracted package tree.

### Python dependencies
The venv installs the required Python packages listed in `requirements.txt`.

Python version
- The setup script selects the highest Python available that is ≥ 3.11 (tries `python3.13`, `python3.12`, `python3.11`, then `python3`/`python`).
- Override with `FT8R_PYTHON` to force a specific interpreter:
  - `FT8R_PYTHON=python3.12 ./scripts/setup_env.sh`
  - `FT8R_PYTHON=/usr/bin/python3.11 ./scripts/setup_env.sh`
- After activation (`source .venv/bin/activate`), `python --version` should report 3.11+.
