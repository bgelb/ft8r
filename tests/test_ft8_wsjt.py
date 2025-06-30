import subprocess
from pathlib import Path


def generate_ft8_wav(message: str, workdir: Path) -> Path:
    """Run ft8sim to generate a wav file for a message."""
    cmd = [
        "ft8sim",
        message,
        "1500",
        "0",
        "0",
        "0",
        "1",
        "-10",
    ]
    subprocess.run(cmd, cwd=workdir, check=True, stdout=subprocess.PIPE, text=True)
    return workdir / "000000_000001.wav"


def decode_ft8_wav(path: Path) -> str:
    """Decode the wav file using jt9 in FT8 mode and return stdout."""
    cmd = ["jt9", "--ft8", str(path)]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        cwd=path.parent,
    )
    return result.stdout


def parse_jt9_output(line: str):
    """Return decoded parameters from a jt9 output line."""
    parts = line.split(maxsplit=5)
    snr = float(parts[1])
    dt = float(parts[2])
    freq = float(parts[3])
    message = parts[5].strip()
    return snr, dt, freq, message


def test_ft8sim_to_jt9(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav_path = generate_ft8_wav(msg, tmp_path)
    output = decode_ft8_wav(wav_path)
    first_line = output.splitlines()[0]
    snr, dt, freq, decoded_msg = parse_jt9_output(first_line)

    assert decoded_msg == msg
    assert abs(snr - (-10)) < 1.0
    assert abs(dt - 0.0) < 0.2
    assert abs(freq - 1500) < 2.0

