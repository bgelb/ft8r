# ft8r

This repository explores building an FT8 demodulator. The test suite can run fully self-contained using a local WSJT-X binary distribution and a Python virtual environment — no system-wide installation required.

### Quick start

1) Set up the local environment (downloads WSJT-X and creates a venv):

```bash
./scripts/setup_env.sh
```

2) Activate the virtual environment:

```bash
source .venv/bin/activate
```

3) Run the tests:

```bash
pytest -q
```

Notes
- The setup script downloads a WSJT-X binary package locally into `.wsjtx/` and exposes CLI tools (e.g., `jt9`, `ft8code`, optionally `ft8sim`) via `.wsjtx/bin/`.
- A small subset of tests require `ft8sim` (used only to synthesize FT8 test audio). If `ft8sim` is not present in the WSJT-X binary package for your platform, those tests are automatically skipped. All other tests still run, including those using the bundled `ft8_lib-2.0/test/wav` sample files and the `jt9` decoder.
- If you already have WSJT-X installed elsewhere, you can point the test suite at its CLI tools by setting `WSJTX_BIN_DIR` to the directory containing `jt9`, `ft8code`, and `ft8sim`.

### Platform support
- macOS: Downloads and mounts the WSJT-X `.dmg`, keeps the app bundle under `.wsjtx/`, and links CLI tools from inside the bundle.
- Linux: Downloads the appropriate `.deb` and extracts it locally (no root needed). Wrapper scripts set `LD_LIBRARY_PATH` so the CLI tools can run from the extracted package tree.

### Python dependencies
The venv installs the required Python packages listed in `requirements.txt`.
