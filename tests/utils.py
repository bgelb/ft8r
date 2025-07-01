from pathlib import Path
import subprocess


def generate_ft8_wav(message: str, workdir: Path, freq: int = 1500) -> Path:
    """Run ft8sim to generate a WAV file containing ``message``."""
    cmd = [
        "ft8sim",
        message,
        str(freq),
        "0",
        "0",
        "0",
        "1",
        "-10",
    ]
    subprocess.run(cmd, cwd=workdir, check=True, stdout=subprocess.PIPE, text=True)
    return workdir / "000000_000001.wav"


def default_search_params(sample_rate_in_hz: int):
    """Return ``(max_freq_bin, max_dt_symbols)`` used for candidate searches."""
    from utils import TONE_SPACING_IN_HZ

    sym_len = int(sample_rate_in_hz / TONE_SPACING_IN_HZ)
    max_freq_bin = int(2500 / TONE_SPACING_IN_HZ)
    max_dt_samples = int(sample_rate_in_hz * 2)
    max_dt_symbols = max_dt_samples // sym_len
    return max_freq_bin, max_dt_symbols
