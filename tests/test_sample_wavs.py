import re
from pathlib import Path

from demod import decode_full_period
from utils import read_wav
from tests.utils import DEFAULT_DT_EPS, DEFAULT_FREQ_EPS

# Directory containing sample WAVs and accompanying TXT files from ft8_lib
DATA_DIR = Path(__file__).resolve().parent.parent / "ft8_lib-2.0" / "test" / "wav"

# Mapping of sample file stem to a tuple of ``(decoded, not_decoded)`` counts.
# ``decoded`` is the number of lines from the accompanying ``.txt`` file that
# should be found by the decoder while ``not_decoded`` indicates how many
# listed transmissions we currently expect to miss.  Additional decodes not
# present in the ``.txt`` file are ignored by the assertions below.
SAMPLES = {
    "191111_110115": (0, 1),
    "191111_110130": (2, 3),
    "191111_110145": (0, 1),
    "191111_110200": (2, 3),
    "191111_110215": (2, 1),
    "191111_110615": (12, 9),
    "191111_110630": (5, 9),
    "191111_110645": (7, 12),
    "191111_110700": (8, 7),
}


def parse_expected(path: Path):
    """Parse WSJT-X style decode lines from ``path``."""
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "<" in line:
            # Skip truncated or empty lines
            continue
        m = re.match(r"\d+\s+(-?\d+)\s+([\d.-]+)\s+(\d+)\s+~\s+(.*)", line)
        if not m:
            continue
        _snr, dt, freq, msg = m.groups()
        msg = re.sub(r"\s{2,}.*", "", msg).strip()
        records.append((msg, float(dt), float(freq)))
    return records


def check_decodes(results, expected):
    """Return number of ``expected`` records found in ``results``."""
    matched = 0
    for msg, dt, freq in expected:
        for rec in results:
            if (
                rec["message"] == msg
                and abs(rec["dt"] - dt) < DEFAULT_DT_EPS
                and abs(rec["freq"] - freq) < DEFAULT_FREQ_EPS
            ):
                matched += 1
                break
    return matched


def idfn(param):
    return str(param)


import pytest


@pytest.mark.parametrize("stem,expected", SAMPLES.items(), ids=idfn)
def test_decode_sample_wavs(stem, expected):
    expected_success, expected_fail = expected
    wav_path = DATA_DIR / f"{stem}.wav"
    txt_path = DATA_DIR / f"{stem}.txt"

    audio = read_wav(str(wav_path))
    results = decode_full_period(audio)

    expected_records = parse_expected(txt_path)
    matched = check_decodes(results, expected_records)

    assert matched == expected_success
    assert len(expected_records) - matched == expected_fail
