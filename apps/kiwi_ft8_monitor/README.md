FT8 Monitor (console)

This console app streams 12 kHz audio from either a KiwiSDR or an SDRplay RSP,
aligns to 15‑second UTC boundaries (:00/:15/:30/:45), decodes FT8 using the
local library, and shows results with a small metrics panel.

Requirements
- Python 3.10+
- Repo Python deps: from repo root run `python -m pip install -r requirements.txt`
  (includes NumPy/SciPy/ldpc)
- One of:
  - KiwiSDR reachable at 192.168.2.10:8073 (default). Optional: `pip install kiwisdr`
    for direct streaming, or use the recorder fallback below.
  - SDRplay RSP (tested with RSPduo) with:
    - SoapySDR Python bindings available to Python
    - SDRplay API/driver installed system‑wide
    - SciPy (already in `requirements.txt`) for decimation

Optional: install Kiwi recorder locally (enables recorder fallback)
  bash apps/kiwi_ft8_monitor/install_kiwirecorder.sh

Usage
- List SDRplay devices
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py --source sdrplay --list-sdrplay

- Kiwi live monitor
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py \
    --source kiwi --host 192.168.2.10 --freq-khz 14074 --mode usb --rate 12000

- SDRplay live (by index)
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py \
    --source sdrplay --device-index 0 --freq-khz 14074 --rate 12000 --sdr-rate 192000

- SDRplay live (by device args)
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py \
    --source sdrplay --device-args "driver=sdrplay,serial=YOUR_SERIAL" \
    --freq-khz 14074 --rate 12000 --sdr-rate 192000

- Oneshot (capture one 15 s window, decode, exit)
  Kiwi:    PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py --source kiwi --host 192.168.2.10 --freq-khz 14074 --mode usb --rate 12000 --oneshot --no-ui
  SDRplay: PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py --source sdrplay --device-index 0 --freq-khz 14074 --rate 12000 --sdr-rate 192000 --oneshot --no-ui

- Offline (use a 12 kHz mono WAV)
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py --wav ft8_lib-2.0/test/wav/websdr_test6.wav --no-ui

Keys
- q: quit

Notes
- Windows align to UTC boundaries. Audio is continuously buffered; decoding runs
  immediately after each boundary over the last 15 seconds of audio.
- Metrics panel shows: candidate count, decode count, and decode wall time for
  the most recent window.
- Recorder fallback: the app auto‑detects a local binary at
  `apps/kiwi_ft8_monitor/bin/kiwirecorder.py` (installed by the script) or a
  `kiwirecorder.py` found on PATH.
- SDRplay path: complex IQ → low‑pass + decimate → real audio (USB demod) to 12 kHz.
