#!/usr/bin/env python3
import argparse
import collections
import curses
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Deque, List

import numpy as np

try:
    # External dependency providing a simple KiwiSDR audio client
    # pip install kiwisdr
    from kiwisdr.client import KiwiSDRStream  # type: ignore
except Exception:  # pragma: no cover - best effort import
    KiwiSDRStream = None  # type: ignore

from utils import RealSamples
from search import find_candidates
"""Runtime configuration values with safe defaults."""
# Threshold used during candidate search/decoding. Try to import from tests when
# running inside the repo (developer environment), but default to a safe value
# when running the app standalone so it does not depend on test modules.
try:  # pragma: no cover - optional import for dev environment
    from tests.utils import DEFAULT_SEARCH_THRESHOLD as _DEFAULT_SEARCH_THRESHOLD  # type: ignore
    DEFAULT_SEARCH_THRESHOLD = _DEFAULT_SEARCH_THRESHOLD
except Exception:  # pragma: no cover - production/runtime fallback
    DEFAULT_SEARCH_THRESHOLD = 1.0
from demod import decode_full_period, TONE_SPACING_IN_HZ


@dataclass
class Metrics:
    dt_utc: float = 0.0
    window_start: float = 0.0
    window_end: float = 0.0
    num_candidates: int = 0
    num_decodes: int = 0
    decode_time_s: float = 0.0


class KiwiAudioSource:
    def __init__(self, host: str, port: int, freq_khz: float, mode: str, rate: int):
        if KiwiSDRStream is None:
            print(
                "ERROR: The 'kiwisdr' package is required. Install with 'pip install kiwisdr'",
                file=sys.stderr,
            )
            sys.exit(2)
        self.stream = KiwiSDRStream(
            host=host,
            port=port,
            freq=freq_khz,
            mode=mode,
            samp_rate=rate,
            chan=0,
        )
        self.rate = rate
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None
        self.buffer: Deque[float] = collections.deque(maxlen=rate * 15)

    def start(self):
        def _reader():
            for chunk in self.stream.get_audio():  # yields numpy arrays float32
                if self._stop.is_set():
                    break
                # Normalize chunk to python floats
                try:
                    self.buffer.extend(chunk.astype(float, copy=False).tolist())
                except Exception:
                    self.buffer.extend(float(x) for x in chunk)

        self._thr = threading.Thread(target=_reader, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        try:
            self.stream.close()
        except Exception:
            pass
        if self._thr is not None:
            self._thr.join(timeout=1.0)
            self._thr = None

    def snapshot_last_15s(self) -> RealSamples:
        # Copy buffer atomically
        arr = np.fromiter(self.buffer, dtype=float)
        return RealSamples(arr, sample_rate_in_hz=self.rate)


class SdrplayAudioSource:
    """Stream complex IQ from an SDRplay (via SoapySDR), produce 12 kHz real audio.

    Implementation notes:
    - Tunes to the provided dial frequency (USB). We low-pass around DC to ~4 kHz,
      decimate to 12 kHz, and take the real part to obtain a mono audio stream.
    - This is a pragmatic USB demodulation approach adequate for FT8 monitoring
      on the standard sub-band with minimal adjacent-LSB content.
    - Requires SoapySDR Python bindings and SDRplay driver installed on the system.
    """

    def __init__(self, freq_khz: float, audio_rate: int = 12000, sdr_rate: int = 192000,
                 device_args: dict | None = None, gain_db: float | None = None):
        self.freq_hz = int(freq_khz * 1000)
        self.audio_rate = audio_rate
        self.sdr_rate = sdr_rate
        self.decim = int(round(self.sdr_rate / self.audio_rate))
        if self.sdr_rate % self.audio_rate != 0:
            raise ValueError("sdr_rate must be an integer multiple of audio_rate")
        self.device_args = device_args or {"driver": "sdrplay"}
        self.gain_db = gain_db

        # Lazy imports to avoid hard dependency when not used
        try:
            import SoapySDR  # type: ignore
            from SoapySDR import SOAPY_SDR_RX  # type: ignore
        except Exception:
            # Try common system site directories when running inside a venv
            import sys as _sys
            import os as _os
            for p in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
                if _os.path.isdir(p) and p not in _sys.path:
                    _sys.path.append(p)
            try:
                import SoapySDR  # type: ignore
                from SoapySDR import SOAPY_SDR_RX  # type: ignore
            except Exception as e:  # pragma: no cover
                print("ERROR: SoapySDR Python bindings not available:", e, file=sys.stderr)
                sys.exit(2)
        self._SoapySDR = SoapySDR
        self._SOAPY_SDR_RX = SOAPY_SDR_RX

        # DSP helpers
        from scipy.signal import resample_poly  # type: ignore
        self._resample_poly = resample_poly

        # Stream state
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None
        self.buffer: Deque[float] = collections.deque(maxlen=audio_rate * 15)
        self._init_device()

    def get_status(self) -> dict:
        """Return a snapshot of RX status for UI: AGC, gains, sensors, levels.

        Tries to be generic w.r.t. SoapySDR driver capabilities. Values may be
        missing depending on the underlying driver and platform.
        """
        st: dict = {}
        try:
            # AGC state (if supported)
            try:
                st['agc'] = bool(self.dev.getGainMode(self._SOAPY_SDR_RX, 0))  # type: ignore[attr-defined]
            except Exception:
                pass
            # Overall gain
            try:
                st['gain_db'] = float(self.dev.getGain(self._SOAPY_SDR_RX, 0))
            except Exception:
                pass
            # Element gains
            gains: dict = {}
            try:
                names = list(self.dev.listGains(self._SOAPY_SDR_RX, 0))
                for name in names:
                    try:
                        val = self.dev.getGain(self._SOAPY_SDR_RX, 0, name)
                        gains[str(name)] = float(val)
                    except Exception:
                        continue
            except Exception:
                pass
            if gains:
                st['gains'] = gains
            # Sensors (device + channel). Try to extract RSSI/overload if present.
            sensors: dict = {}
            try:
                for s in list(self.dev.listSensors()):
                    try:
                        sensors[str(s)] = str(self.dev.readSensor(s))
                    except Exception:
                        continue
            except Exception:
                pass
            try:
                for s in list(self.dev.listChannelSensors(self._SOAPY_SDR_RX, 0)):
                    try:
                        sensors[str(s)] = str(self.dev.readChannelSensor(self._SOAPY_SDR_RX, 0, s))
                    except Exception:
                        continue
            except Exception:
                pass
            if sensors:
                st['sensors'] = sensors
                # Heuristics for common fields
                rs = None
                for k, v in sensors.items():
                    lk = k.lower()
                    if rs is None and ('rssi' in lk or 'power' in lk):
                        try:
                            rs = float(str(v).replace('dBm','').strip())
                        except Exception:
                            pass
                if rs is not None:
                    st['rssi_dbm'] = rs
                ov = None
                for k, v in sensors.items():
                    lk = k.lower(); sv = str(v).lower()
                    if 'over' in lk or 'ovld' in lk:
                        ov = (sv in ('1', 'true', 'yes', 'on'))
                        break
                if ov is not None:
                    st['overload'] = bool(ov)
            # Audio level from recent buffer (RMS dBFS, assume full scale ~1.0)
            try:
                n = min(len(self.buffer), int(self.audio_rate * 0.5))
                if n > 0:
                    import numpy as _np
                    a = _np.fromiter(list(self.buffer)[-n:], dtype=float)
                    rms = float(_np.sqrt(_np.mean(a * a)))
                    eps = 1e-12
                    st['audio_dbfs'] = 20.0 * math.log10(max(rms, eps))
            except Exception:
                pass
        except Exception:
            pass
        return st

    @staticmethod
    def enumerate_devices() -> list[dict]:
        try:
            import SoapySDR  # type: ignore
        except Exception:
            # Try common system site directories when running inside a venv
            import sys as _sys, os as _os
            for p in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
                if _os.path.isdir(p) and p not in _sys.path:
                    _sys.path.append(p)
            try:
                import SoapySDR  # type: ignore
            except Exception:
                return []
        devs: list[dict] = []
        for drv in ("sdrplay", "sdrplay3"):
            try:
                devs.extend(list(SoapySDR.Device.enumerate({"driver": drv})))  # type: ignore
            except Exception:
                pass
        if not devs:
            try:
                devs = list(SoapySDR.Device.enumerate())  # type: ignore
            except Exception:
                pass
        return devs

    def _init_device(self):
        SoapySDR = self._SoapySDR
        SOAPY_SDR_RX = self._SOAPY_SDR_RX
        # Open device
        try:
            self.dev = SoapySDR.Device(self.device_args)  # type: ignore
        except Exception as e:
            print(f"ERROR: Failed to open SDRplay via SoapySDR with args {self.device_args}: {e}", file=sys.stderr)
            sys.exit(2)
        # Configure stream
        self.dev.setSampleRate(SOAPY_SDR_RX, 0, self.sdr_rate)
        try:
            self.dev.setBandwidth(SOAPY_SDR_RX, 0, min(6000000, max(200000, self.sdr_rate)))
        except Exception:
            pass
        self.dev.setFrequency(SOAPY_SDR_RX, 0, self.freq_hz)
        if self.gain_db is not None:
            try:
                self.dev.setGain(SOAPY_SDR_RX, 0, float(self.gain_db))
            except Exception:
                pass
        # Setup RX stream (complex float32)
        self.stream = self.dev.setupStream(SOAPY_SDR_RX, self._SoapySDR.SOAPY_SDR_CF32, [0], {})

    def start(self):
        self.dev.activateStream(self.stream)

        def _reader():
            import numpy as np
            dec = self.decim
            # Use reasonably large read chunk for better FIR efficiency
            chunk = 49152  # 256 ms at 192 kS/s
            buf = np.empty(chunk, dtype=np.complex64)
            while not self._stop.is_set():
                r = self.dev.readStream(self.stream, [buf], chunk)
                # r may be a struct-like with attrs (ret, flags, timeNs) or a tuple
                if hasattr(r, 'ret'):
                    nread = int(getattr(r, 'ret'))
                elif isinstance(r, tuple) and len(r) >= 1:
                    nread = int(r[0]) if r[0] is not None else 0
                else:
                    nread = 0
                if nread < 0:
                    # transient read error; backoff a bit
                    time.sleep(0.01)
                    continue
                if nread == 0:
                    continue
                iq = buf[:nread]
                # Decimate to audio rate with FIR low-pass (Kaiser window)
                y = self._resample_poly(iq, up=1, down=dec, window=("kaiser", 8.6))
                # Real mono audio
                audio = np.real(y).astype(float)
                try:
                    self.buffer.extend(audio.tolist())
                except Exception:
                    self.buffer.extend(float(v) for v in audio)

        self._thr = threading.Thread(target=_reader, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        try:
            self.dev.deactivateStream(self.stream)
        except Exception:
            pass
        try:
            self.dev.closeStream(self.stream)
        except Exception:
            pass
        if self._thr is not None:
            self._thr.join(timeout=1.0)
            self._thr = None

    def snapshot_last_15s(self) -> RealSamples:
        import numpy as np
        arr = np.fromiter(self.buffer, dtype=float)
        return RealSamples(arr, sample_rate_in_hz=self.audio_rate)


def next_boundary_utc(now: float) -> float:
    # Align to :00/:15/:30/:45 boundaries
    secs = int(now)
    mod = secs % 15
    return secs + (15 - mod) if mod != 0 else secs

def next_strict_boundary_utc(now: float, *, epsilon: float = 0.2) -> float:
    """Return the next 15s boundary strictly after 'now'.
    If 'now' is at/near a boundary (within epsilon), skip to the following one.
    """
    nb = next_boundary_utc(now)
    if nb - now <= epsilon:
        nb += 15
    return nb
def _snr_db_from_rec(rec: dict) -> float | None:
    # Prefer explicit SNR if present
    snr = rec.get('snr')
    if isinstance(snr, (int, float)):
        try:
            return float(snr)
        except Exception:
            return None
    # Fallback: approximate from search score if available (ratio -> dB)
    score = rec.get('score')
    if isinstance(score, (int, float)) and score > 0:
        try:
            return 10.0 * math.log10(float(score))
        except Exception:
            return None
    return None


def render(stdscr, decodes: List[dict], metrics: Metrics, now_ts: float | None = None, next_boundary: float | None = None, rx_status: dict | None = None):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    now_ts = now_ts if now_ts is not None else time.time()
    # Header
    stdscr.addstr(0, 0, f"FT8 monitor — window {time.strftime('%H:%M:%S', time.gmtime(metrics.window_start))}–{time.strftime('%H:%M:%S', time.gmtime(metrics.window_end))}Z")
    # Clock and progress bar
    utc_str = time.strftime('%H:%M:%S', time.gmtime(now_ts))
    if next_boundary is None:
        # Use strict boundary for progress calculation to avoid freezing at :00
        next_boundary = next_strict_boundary_utc(now_ts)
    start = next_boundary - 15
    elapsed = max(0.0, min(15.0, now_ts - start))
    ratio = elapsed / 15.0 if 15.0 > 0 else 0.0
    bar_w = max(10, min(w - 40, 40))
    filled = int(bar_w * ratio)
    bar = '[' + ('#' * filled) + ('-' * (bar_w - filled)) + ']'
    stdscr.addstr(1, 0, f"UTC {utc_str}  next {time.strftime('%H:%M:%S', time.gmtime(next_boundary))}Z  {bar} {elapsed:4.1f}/15s")
    stdscr.addstr(2, 0, f"candidates: {metrics.num_candidates}  decodes: {metrics.num_decodes}  decode_time: {metrics.decode_time_s:.2f}s")
    row_sep = 3
    if rx_status:
        # Compose a compact status line
        agc = rx_status.get('agc')
        agc_str = f"agc={'on' if agc else 'off'}" if isinstance(agc, bool) else "agc=--"
        gain = rx_status.get('gain_db')
        gain_str = f"gain={gain:.1f}dB" if isinstance(gain, (int, float)) else "gain=--"
        over = rx_status.get('overload')
        over_str = f"over={'YES' if over else 'no'}" if isinstance(over, bool) else "over=--"
        rssi = rx_status.get('rssi_dbm')
        rssi_str = f"rssi={rssi:.1f}dBm" if isinstance(rssi, (int, float)) else "rssi=--"
        adbfs = rx_status.get('audio_dbfs')
        adbfs_str = f"audio={adbfs:.1f}dBFS" if isinstance(adbfs, (int, float)) else "audio=--"
        stdscr.addstr(3, 0, f"RX {agc_str}  {gain_str}  {over_str}  {rssi_str}  {adbfs_str}"[:w-1])
        row_sep = 4
        gains = rx_status.get('gains') or {}
        if gains:
            items = ", ".join(f"{k}={v:.1f}dB" for k, v in list(gains.items())[:6])
            stdscr.addstr(4, 0, f"gains: {items}"[:w-1])
            row_sep = 5
    stdscr.hline(row_sep, 0, ord('-'), w)
    # Decodes list (sorted externally); include SNR column when available/approx)
    row = row_sep + 1
    for d in decodes:
        snr = _snr_db_from_rec(d)
        snr_str = f"{snr:+4.0f}dB" if (snr is not None and math.isfinite(snr)) else "   --"
        line = f"{d['dt']:+5.2f}s  {d['freq']:7.1f} Hz  {snr_str:>6}  {d['message']}"
        if row < h-1:
            stdscr.addstr(row, 0, line[:w-1])
            row += 1
        else:
            break
    stdscr.addstr(h-1, 0, "Press q to quit")
    stdscr.refresh()


def run_monitor_source(src_factory):
    src = src_factory()
    src.start()
    try:
        time.sleep(1.0)  # fill buffer a bit

        # Curses UI with ticking clock/progress
        def _main(stdscr):
            curses.curs_set(0)
            stdscr.nodelay(True)
            last_decodes: List[dict] = []
            last_metrics = Metrics()
            nb = next_strict_boundary_utc(time.time())
            while True:
                now = time.time()
                if now >= nb:
                    audio = src.snapshot_last_15s()
                    sym_len = int(audio.sample_rate_in_hz / TONE_SPACING_IN_HZ)
                    max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
                    max_dt_samples = len(audio.samples) - int(audio.sample_rate_in_hz * 0.5)
                    max_dt_symbols = -(-max_dt_samples // sym_len)
                    cand_count = len(find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD))
                    t0 = time.perf_counter()
                    decs = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD)
                    t1 = time.perf_counter()
                    decs = sorted(decs, key=lambda d: d.get('freq', 0.0))
                    last_decodes = decs
                    last_metrics = Metrics(
                        dt_utc=nb,
                        window_start=nb-15,
                        window_end=nb,
                        num_candidates=cand_count,
                        num_decodes=len(decs),
                        decode_time_s=(t1 - t0),
                    )
                    # Schedule strictly after 'now' to avoid re-triggering same boundary
                    nb = next_strict_boundary_utc(now)
                # Always render clock/progress with last results
                rx_status = None
                try:
                    if hasattr(src, 'get_status') and callable(getattr(src, 'get_status')):
                        rx_status = src.get_status()  # type: ignore[attr-defined]
                except Exception:
                    rx_status = None
                render(stdscr, last_decodes, last_metrics, now_ts=now, next_boundary=nb, rx_status=rx_status)
                # Handle key
                try:
                    ch = stdscr.getch()
                    if ch in (ord('q'), ord('Q')):
                        break
                except Exception:
                    pass
                time.sleep(0.2)

        curses.wrapper(_main)
    finally:
        src.stop()


def run_monitor(host: str, port: int, freq_khz: float, mode: str, rate: int):
    return run_monitor_source(lambda: KiwiAudioSource(host, port, freq_khz, mode, rate))

def run_monitor_recorder(host: str, port: int, freq_khz: float, mode: str, rate: int):
    """Live monitor using external kiwirecorder.py rotating 15s WAV files.
    Requires local bin at apps/kiwi_ft8_monitor/bin/kiwirecorder.py or PATH.
    """
    import shutil, subprocess, tempfile, os
    from pathlib import Path
    from utils import read_wav

    app_dir = Path(__file__).resolve().parent
    local_rec = app_dir / 'bin' / 'kiwirecorder.py'
    rec = str(local_rec) if local_rec.exists() else shutil.which('kiwirecorder.py')
    if not rec:
        print("ERROR: kiwirecorder.py not found. Run apps/kiwi_ft8_monitor/install_kiwirecorder.sh", file=sys.stderr)
        return

    tdir_ctx = tempfile.TemporaryDirectory()
    tdir = tdir_ctx.name
    cmd = [
        rec,
        '-s', str(host),
        '-p', str(port),
        '-f', str(freq_khz),
        '-m', mode,
        '-r', str(rate),
        '--dt-sec', '15',
        '--connect-timeout', '5',
        '--busy-timeout', '5',
        '-d', tdir,
        '--not-quiet',
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    processed: set[str] = set()

    def _main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        last_decodes: List[dict] = []
        last_metrics = Metrics()
        try:
            nb = next_strict_boundary_utc(time.time())
            while True:
                wavs = sorted([str(Path(tdir)/p) for p in os.listdir(tdir) if p.lower().endswith('.wav')])
                # Decode any new completed file(s) except the newest (being written)
                to_decode = [w for w in wavs[:-1] if w not in processed]
                if to_decode:
                    wav_path = to_decode[-1]
                    audio = read_wav(wav_path)
                    sym_len = int(audio.sample_rate_in_hz / TONE_SPACING_IN_HZ)
                    max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
                    max_dt_samples = len(audio.samples) - int(audio.sample_rate_in_hz * 0.5)
                    max_dt_symbols = -(-max_dt_samples // sym_len)
                    t0 = time.perf_counter()
                    cands = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD)
                    decs = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD)
                    decs = sorted(decs, key=lambda d: d.get('freq', 0.0))
                    t1 = time.perf_counter()
                    # Approximate window bounds from current UTC boundary
                    now = time.time(); nb = next_strict_boundary_utc(now)
                    last_decodes[:] = decs
                    last_metrics.dt_utc = nb
                    last_metrics.window_start = nb-15
                    last_metrics.window_end = nb
                    last_metrics.num_candidates = len(cands)
                    last_metrics.num_decodes = len(decs)
                    last_metrics.decode_time_s = (t1 - t0)
                    processed.add(wav_path)
                # periodic render with ticking clock/progress
                now2 = time.time(); nb2 = next_strict_boundary_utc(now2)
                render(stdscr, last_decodes, last_metrics, now_ts=now2, next_boundary=nb2)
                # handle key
                try:
                    ch = stdscr.getch()
                    if ch in (ord('q'), ord('Q')):
                        break
                except Exception:
                    pass
                time.sleep(0.25)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
            tdir_ctx.cleanup()

    curses.wrapper(_main)


def run_oneshot(host: str, port: int, freq_khz: float, mode: str, rate: int, no_ui: bool = True):
    """Connect to Kiwi, capture a single 15 s FT8 window aligned to UTC boundaries, decode, exit.

    If the kiwisdr Python package is unavailable, falls back to using the external
    'kiwirecorder.py' if found on PATH to capture a 15 s WAV.
    """
    # DEFAULT_SEARCH_THRESHOLD is defined at module scope with a runtime-safe default
    if KiwiSDRStream is None:
        # Fallback: use kiwirecorder.py to capture a single 15 s WAV aligned to boundary
        import shutil, tempfile, subprocess, os
        from pathlib import Path
        # Prefer local bin under app dir, else PATH
        app_dir = Path(__file__).resolve().parent
        local_rec = app_dir / 'bin' / 'kiwirecorder.py'
        rec = str(local_rec) if local_rec.exists() else shutil.which('kiwirecorder.py')
        if not rec:
            print("ERROR: neither 'kiwisdr' package nor 'kiwirecorder.py' found. Run apps/kiwi_ft8_monitor/install_kiwirecorder.sh to install locally.", file=sys.stderr)
            return
        # Align to boundary
        now = time.time(); nb = next_strict_boundary_utc(now); time.sleep(max(0.0, nb - now))
        with tempfile.TemporaryDirectory() as td:
            # Build kiwirecorder command: 15 s audio, USB, sample rate
            cmd = [
                rec,
                '-s', str(host),
                '-p', str(port),
                '-f', str(freq_khz),
                '-m', mode,
                '-r', str(rate),
                '--tlimit', '15',
                '--connect-timeout', '5',
                '--connect-retries', '2',
                '--busy-timeout', '5',
                '--busy-retries', '2',
                '-d', td,
            ]
            try:
                subprocess.run(cmd, check=True, timeout=75)
            except subprocess.TimeoutExpired:
                print('ERROR: kiwirecorder timed out', file=sys.stderr)
                return
            # Find the WAV written
            wavs = [p for p in os.listdir(td) if p.lower().endswith('.wav')]
            if not wavs:
                print('ERROR: kiwirecorder did not produce a WAV', file=sys.stderr)
                return
            from utils import read_wav
            audio = read_wav(os.path.join(td, wavs[0]))
        # Decode
        sym_len = int(audio.sample_rate_in_hz / TONE_SPACING_IN_HZ)
        max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
        max_dt_samples = len(audio.samples) - int(audio.sample_rate_in_hz * 0.5)
        max_dt_symbols = -(-max_dt_samples // sym_len)
        cands = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD)
        t1 = time.perf_counter(); decs = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD); t2 = time.perf_counter()
        decs = sorted(decs, key=lambda d: d.get('freq', 0.0))
        metrics = Metrics(dt_utc=nb, window_start=nb-15, window_end=nb, num_candidates=len(cands), num_decodes=len(decs), decode_time_s=(t2-t1))
        print(f"candidates={metrics.num_candidates} decodes={metrics.num_decodes} decode_time={metrics.decode_time_s:.2f}s")
        for d in decs:
            snr = _snr_db_from_rec(d)
            snr_str = f"{snr:+4.0f}dB" if (snr is not None and math.isfinite(snr)) else "   --"
            print(f"{d['dt']:+5.2f}s {d['freq']:7.1f}Hz {snr_str:>6} {d['message']}")
        return
    # Normal streaming oneshot
    src = KiwiAudioSource(host, port, freq_khz, mode, rate)
    src.start()
    try:
        need = rate * 15; t0 = time.time()
        while len(src.buffer) < need:
            time.sleep(0.05)
            if time.time() - t0 > 60:
                print("ERROR: timed out waiting for audio buffer to fill", file=sys.stderr)
                return
        now = time.time(); nb = next_strict_boundary_utc(now); time.sleep(max(0.0, nb - now))
        audio = src.snapshot_last_15s()
        sym_len = int(audio.sample_rate_in_hz / TONE_SPACING_IN_HZ)
        max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
        max_dt_samples = len(audio.samples) - int(audio.sample_rate_in_hz * 0.5)
        max_dt_symbols = -(-max_dt_samples // sym_len)
        cands = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD)
        t1 = time.perf_counter(); decs = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD); t2 = time.perf_counter()
        decs = sorted(decs, key=lambda d: d.get('freq', 0.0))
        metrics = Metrics(dt_utc=nb, window_start=nb-15, window_end=nb, num_candidates=len(cands), num_decodes=len(decs), decode_time_s=(t2-t1))
        if no_ui:
            print(f"candidates={metrics.num_candidates} decodes={metrics.num_decodes} decode_time={metrics.decode_time_s:.2f}s")
            for d in decs:
                snr = _snr_db_from_rec(d)
                snr_str = f"{snr:+4.0f}dB" if (snr is not None and math.isfinite(snr)) else "   --"
                print(f"{d['dt']:+5.2f}s {d['freq']:7.1f}Hz {snr_str:>6} {d['message']}")
        else:
            def _once(stdscr):
                curses.curs_set(0); render(stdscr, decs, metrics); stdscr.getch()
            curses.wrapper(_once)
    finally:
        src.stop()


def run_oneshot_sdrplay(freq_khz: float, rate: int, sdr_rate: int, device_args: dict | None = None, gain_db: float | None = None, no_ui: bool = True):
    # DEFAULT_SEARCH_THRESHOLD is defined at module scope with a runtime-safe default
    src = SdrplayAudioSource(freq_khz=freq_khz, audio_rate=rate, sdr_rate=sdr_rate, device_args=device_args, gain_db=gain_db)
    src.start()
    try:
        need = rate * 15; t0 = time.time()
        while len(src.buffer) < need:
            time.sleep(0.05)
            if time.time() - t0 > 30: break
        now = time.time(); nb = next_strict_boundary_utc(now); time.sleep(max(0.0, nb - now))
        audio = src.snapshot_last_15s()
        sym_len = int(audio.sample_rate_in_hz / TONE_SPACING_IN_HZ)
        max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
        max_dt_samples = len(audio.samples) - int(audio.sample_rate_in_hz * 0.5)
        max_dt_symbols = -(-max_dt_samples // sym_len)
        cands = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD)
        t1 = time.perf_counter(); decs = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD); t2 = time.perf_counter()
        decs = sorted(decs, key=lambda d: d.get('freq', 0.0))
        metrics = Metrics(dt_utc=nb, window_start=nb-15, window_end=nb, num_candidates=len(cands), num_decodes=len(decs), decode_time_s=(t2-t1))
        if no_ui:
            print(f"candidates={metrics.num_candidates} decodes={metrics.num_decodes} decode_time={metrics.decode_time_s:.2f}s")
            for d in decs:
                snr = _snr_db_from_rec(d)
                snr_str = f"{snr:+4.0f}dB" if (snr is not None and math.isfinite(snr)) else "   --"
                print(f"{d['dt']:+5.2f}s {d['freq']:7.1f}Hz {snr_str:>6} {d['message']}")
        else:
            def _once(stdscr):
                curses.curs_set(0); render(stdscr, decs, metrics); stdscr.getch()
            curses.wrapper(_once)
    finally:
        src.stop()


def main():
    ap = argparse.ArgumentParser(description="FT8 live monitor (KiwiSDR or SDRplay)")
    src_grp = ap.add_argument_group("source selection")
    src_grp.add_argument("--source", choices=["kiwi", "sdrplay"], default="kiwi", help="audio source")
    kiwi = ap.add_argument_group("kiwi options")
    kiwi.add_argument("--host", default="192.168.2.10")
    kiwi.add_argument("--port", type=int, default=8073)
    kiwi.add_argument("--mode", default="usb")
    sdr = ap.add_argument_group("sdrplay options")
    sdr.add_argument("--list-sdrplay", action="store_true", help="list detected SDRplay devices and exit")
    sdr.add_argument("--device-index", type=int, default=None, help="index into detected SDRplay devices")
    sdr.add_argument("--device-args", type=str, default="", help="SoapySDR device args string, e.g. 'driver=sdrplay,serial=XXXX'")
    sdr.add_argument("--sdr-rate", type=int, default=192000, help="SDR sample rate before decimation")
    sdr.add_argument("--gain-db", type=float, default=None, help="Optional RF gain in dB")
    ap.add_argument("--freq-khz", type=float, default=14074.0, help="dial frequency in kHz (USB)")
    ap.add_argument("--rate", type=int, default=12000)
    ap.add_argument("--wav", type=str, default="", help="offline: path to 12 kHz mono WAV to process")
    ap.add_argument("--no-ui", action="store_true", help="disable curses UI; print plain lines")
    ap.add_argument("--oneshot", action="store_true", help="capture one 15 s window, decode, exit")
    args = ap.parse_args()

    if args.wav:
        # Offline test mode: process a single 15 s window from a WAV
        from utils import read_wav
        # DEFAULT_SEARCH_THRESHOLD is defined at module scope with a runtime-safe default

        audio = read_wav(args.wav)
        sym_len = int(audio.sample_rate_in_hz / TONE_SPACING_IN_HZ)
        max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
        max_dt_samples = len(audio.samples) - int(audio.sample_rate_in_hz * 0.5)
        max_dt_symbols = -(-max_dt_samples // sym_len)
        t0 = time.perf_counter()
        cands = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD)
        decs = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD)
        decs = sorted(decs, key=lambda d: d.get('freq', 0.0))
        t1 = time.perf_counter()
        if args.no_ui:
            print(f"candidates={len(cands)} decodes={len(decs)} decode_time={t1-t0:.2f}s")
            for d in decs:
                snr = _snr_db_from_rec(d)
                snr_str = f"{snr:+4.0f}dB" if (snr is not None and math.isfinite(snr)) else "   --"
                print(f"{d['dt']:+5.2f}s {d['freq']:7.1f}Hz {snr_str:>6} {d['message']}")
        else:
            def _one(stdscr):
                curses.curs_set(0)
                metrics = Metrics(window_start=0, window_end=15, num_candidates=len(cands), num_decodes=len(decs), decode_time_s=(t1-t0))
                render(stdscr, decs, metrics)
                stdscr.getch()
            curses.wrapper(_one)
        return

    if args.source == "sdrplay" and args.list_sdrplay:
        devs = SdrplayAudioSource.enumerate_devices()
        if not devs:
            print("No SDRplay devices detected via SoapySDR.")
        else:
            for i, d in enumerate(devs):
                print(f"[{i}] {d}")
        return

    if args.oneshot:
        if args.source == "sdrplay":
            # Resolve device args
            dargs: dict | None = None
            if args.device_args:
                dargs = {}
                for kv in args.device_args.split(','):
                    if not kv:
                        continue
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        dargs[k.strip()] = v.strip()
                    else:
                        dargs[kv.strip()] = ""
            elif args.device_index is not None:
                devs = SdrplayAudioSource.enumerate_devices()
                if args.device_index < 0 or args.device_index >= len(devs):
                    print("Invalid --device-index; use --list-sdrplay to see devices", file=sys.stderr)
                    sys.exit(2)
                dargs = dict(devs[args.device_index])
            run_oneshot_sdrplay(args.freq_khz, args.rate, args.sdr_rate, device_args=dargs, gain_db=args.gain_db, no_ui=args.no_ui)
        else:
            run_oneshot(args.host, args.port, args.freq_khz, args.mode, args.rate, no_ui=args.no_ui)
        return

    if args.no_ui:
        print("--no-ui is supported with --wav or --oneshot. For live monitoring use the curses UI.")
        return

    # Dispatch by source
    if args.source == "sdrplay":
        dargs: dict | None = None
        if args.device_args:
            dargs = {}
            for kv in args.device_args.split(','):
                if not kv:
                    continue
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    dargs[k.strip()] = v.strip()
                else:
                    dargs[kv.strip()] = ""
        elif args.device_index is not None:
            devs = SdrplayAudioSource.enumerate_devices()
            if args.device_index < 0 or args.device_index >= len(devs):
                print("Invalid --device-index; use --list-sdrplay to see devices", file=sys.stderr)
                sys.exit(2)
            dargs = dict(devs[args.device_index])
        return run_monitor_source(lambda: SdrplayAudioSource(args.freq_khz, audio_rate=args.rate, sdr_rate=args.sdr_rate, device_args=dargs, gain_db=args.gain_db))
    # Else kiwi source
    if KiwiSDRStream is not None:
        run_monitor(args.host, args.port, args.freq_khz, args.mode, args.rate)
    else:
        run_monitor_recorder(args.host, args.port, args.freq_khz, args.mode, args.rate)


if __name__ == "__main__":
    main()
