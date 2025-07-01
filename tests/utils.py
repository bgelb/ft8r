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
