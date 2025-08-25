FT8 Monitor (console)

This is a simple console application that connects to a KiwiSDR, streams audio
at 12 kHz from a single receive channel (USB), chops the stream into 15‑second
windows aligned to :00/:15/:30/:45 each minute, decodes FT8 with the local
library, and displays results along with a small metrics panel.

Requirements
- Python 3.10+
- This repository’s Python dependencies (install from repo root)
- One of:
  - KiwiSDR reachable at 192.168.2.10:8073 (default), or
  - SDRplay RSP (tested with RSPduo) with SoapySDR + SDRplay API installed system-wide

Install Kiwi client locally (for one-shot/live via recorder)
  bash apps/kiwi_ft8_monitor/install_kiwirecorder.sh

Run (Kiwi live monitor)
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py \
    --source kiwi --host 192.168.2.10 --freq-khz 14074 --mode usb --rate 12000

List SDRplay devices
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py --source sdrplay --list-sdrplay

Run (SDRplay live monitor, choose by index)
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py \
    --source sdrplay --device-index 0 --freq-khz 14074 --rate 12000 --sdr-rate 192000

Run (SDRplay live monitor, explicit device args)
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py \
    --source sdrplay --device-args "driver=sdrplay,serial=YOUR_SERIAL" \
    --freq-khz 14074 --rate 12000 --sdr-rate 192000

One‑shot (capture one 15 s, decode, exit)
  Kiwi:    PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py --source kiwi --host 192.168.2.10 --freq-khz 14074 --mode usb --rate 12000 --oneshot --no-ui
  SDRplay: PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py --source sdrplay --device-index 0 --freq-khz 14074 --rate 12000 --sdr-rate 192000 --oneshot --no-ui

Offline test (use a WAV)
  PYTHONPATH=. python apps/kiwi_ft8_monitor/main.py --wav ft8_lib-2.0/test/wav/websdr_test6.wav --no-ui

Keys
- q: quit

Notes
- The app aligns windows to UTC wall‑clock boundaries (:00/:15/:30/:45). Audio
  is continuously buffered in a background reader, and decoding happens
  immediately after each boundary using the last 15 seconds of buffered audio.
- The metrics panel shows: candidate count (coarse search), decode count,
  and decode wall time for the most recent window.
 - For recorder-based capture, the app auto-detects a local binary at
   apps/kiwi_ft8_monitor/bin/kiwirecorder.py (installed by the script),
   or falls back to a kiwirecorder.py found on PATH.
 - SDRplay mode uses a simple complex low‑pass + decimate and real extraction
   to produce 12 kHz mono audio suitable for FT8 decoding. SoapySDR and the
   SDRplay driver must be installed on the system.
