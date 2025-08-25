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

SDRplay status panel
- When running with `--source sdrplay`, the UI shows a compact receiver status line beneath the metrics:
  - `agc=on/off`: whether SoapySDR gain mode (AGC) is enabled.
  - `gain=XdB`: overall gain reported by the driver (see “Gain model” below).
  - `over=YES/no`: overload indication if exposed by the driver.
  - `rssi=XdBm`: received signal strength if the driver exposes an RSSI sensor.
  - `audio=XdBFS`: recent audio RMS level after decimation, referenced to full scale.
  - If the driver exposes per‑stage gains, a `gains:` line lists a few elements (e.g., `LNA`, `IF`, etc.).

SDRplay gain model, AGC, and ranges
- Gain vs “gain reduction”: SDRplay devices often model the internal IF attenuator as “gain reduction (GR)” in dB, where a larger number means lower gain. SoapySDR presents a normalized gain interface and, depending on driver version, may report either:
  - a conventional “gain in dB” that increases with amplification, or
  - the raw IF “gain reduction” value. The app surfaces whatever the driver reports for overall and per‑element gains.
- AGC behavior: Enabling AGC via SoapySDR’s `getGainMode(true)` lets the driver manage internal stages (typically the IF gain reduction, and sometimes LNA state steps per band) to maintain a target level. When AGC is on, manual overall gain is usually ignored or partially overridden by the driver.
- RSPduo ranges: Ranges are device/driver dependent and can vary with band/mode. Typical RSPduo controls include a discrete LNA state (several steps) and an IF gain reduction range in dB. To see the exact ranges on your system, probe the driver:
  - `SoapySDRUtil --probe="driver=sdrplay"` (or `driver=sdrplay3`)
  - Look for `Gain` elements and their ranges, e.g., `LNA`, `IF`, or `GR`. The app will display whichever elements the driver exposes.
- Recommended starting points:
  - Start with `agc=on` (default if your driver enables it). Watch the UI for `over=YES` and ensure `audio` sits roughly between −20 dBFS and −6 dBFS during normal activity.
  - If you prefer manual control: disable AGC, set a modest overall gain (or lower IF “gain reduction”), and increase until `over` does not trigger while `audio` remains below clipping (≈ −6 dBFS). Bands with strong broadcast signals may require more attenuation.
  - Leave `--sdr-rate` at 192000 unless you have a reason to change it; the app low‑passes and decimates to 12 kHz for FT8.

Tip: Since driver semantics differ, treat the UI’s `gain` and `gains` as authoritative for your platform. If values appear inverted relative to expectation, they may be reported as “gain reduction”. Use `SoapySDRUtil --probe` to confirm names and ranges.
