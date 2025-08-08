import subprocess
from pathlib import Path

from tests.utils import (
    generate_ft8_wav,
    DEFAULT_FREQ_EPS,
    DEFAULT_DT_EPS,
    resolve_wsjt_binary,
)


def decode_ft8_wav(path: Path) -> str:
    """Decode the wav file using jt9 in FT8 mode and return stdout."""
    jt9_path = resolve_wsjt_binary("jt9")
    if not jt9_path:
        raise AssertionError(
            "jt9 not found. Please run scripts/setup_env.sh or set WSJTX_BIN_DIR"
        )
    cmd = [jt9_path, "--ft8", str(path)]
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
    dt = float(parts[2])
    freq = float(parts[3])
    message = parts[5].strip()
    return dt, freq, message


def test_ft8sim_to_jt9(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav_path = generate_ft8_wav(msg, tmp_path)
    output = decode_ft8_wav(wav_path)
    first_line = output.splitlines()[0]
    dt, freq, decoded_msg = parse_jt9_output(first_line)

    assert decoded_msg == msg
    assert abs(dt - 0.0) < DEFAULT_DT_EPS
    assert abs(freq - 1500) < 2.0

