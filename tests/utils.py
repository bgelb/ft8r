from pathlib import Path
import subprocess

DEFAULT_SEARCH_THRESHOLD = 1.0

# Epsilon for validating decoded frequency in tests (Hz)
DEFAULT_FREQ_EPS = 1.0
# Epsilon for validating decoded time offset in tests (s)
DEFAULT_DT_EPS = 0.2


def generate_ft8_wav(
    message: str,
    workdir: Path,
    freq: int = 1500,
    snr: int = -10,
    dt: float = 0.0,
    fdop: float = 0.0,
) -> Path:
    """Run ft8sim to generate a WAV file containing ``message``."""

    snr_for_ft8sim = snr
    cmd = [
        "ft8sim",
        message,
        str(freq),
        str(dt),
        str(fdop),
        "0",
        "1",
        str(snr_for_ft8sim),
    ]
    subprocess.run(cmd, cwd=workdir, check=True, stdout=subprocess.PIPE, text=True)
    return workdir / "000000_000001.wav"


def ft8code_bits(message: str) -> str:
    """Return the 174-bit FT8 payload for ``message`` using ``ft8code``."""
    result = subprocess.run(
        ["ft8code", message], check=True, stdout=subprocess.PIPE, text=True
    )
    bits = []
    grab = False
    for line in result.stdout.splitlines():
        if "Source-encoded message" in line:
            grab = True
            continue
        if grab and line.strip():
            bits.append(line.strip())
            if len("".join(bits)) >= 77:
                grab = False
        if "14-bit CRC:" in line:
            grab = True
            continue
        if "83 Parity bits:" in line:
            grab = True
            continue
        if grab and line.strip():
            bits.append(line.strip())
            if len("".join(bits)) >= 174:
                break
    return "".join(bits)[:174]


def default_search_params(sample_rate_in_hz: int):
    """Return ``(max_freq_bin, max_dt_symbols)`` used for candidate searches."""
    from utils import TONE_SPACING_IN_HZ

    sym_len = int(sample_rate_in_hz / TONE_SPACING_IN_HZ)
    max_freq_bin = int(2500 / TONE_SPACING_IN_HZ)
    # Limit the Costas search to a window starting at the beginning of the
    # 15 second cycle and ending 3.12 seconds later.  This window spans a
    # 0.5 s pre interval, the seven Costas symbols (7 * 0.16 s) and an
    # additional 1.5 s of trailing audio.  ``max_dt_samples`` controls the
    # furthest point in the input audio examined by :func:`search.find_candidates`.
    max_dt_samples = int(sample_rate_in_hz * 3.12)
    # ``max_dt_symbols`` is rounded up so the 3.12 s window is fully included
    max_dt_symbols = -(-max_dt_samples // sym_len)
    return max_freq_bin, max_dt_symbols
