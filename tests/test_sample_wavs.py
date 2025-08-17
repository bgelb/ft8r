import re
from pathlib import Path

import pytest

from demod import decode_full_period
from utils import read_wav
from tests.utils import DEFAULT_DT_EPS, DEFAULT_FREQ_EPS

# Directory containing sample WAVs and accompanying TXT files from ft8_lib
DATA_DIR = Path(__file__).resolve().parent.parent / "ft8_lib-2.0" / "test" / "wav"

# Minimum aggregate decode ratio required to pass across the full library
FULL_MIN_RATIO = 0.645


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


def list_all_stems() -> list[str]:
    """Return sorted list of sample stems relative to DATA_DIR without suffix.

    Examples: "websdr_test8", "20m_busy/test_11"
    """
    stems: list[str] = []
    for wav in sorted(DATA_DIR.rglob("*.wav")):
        # Only include WAVs that also have a corresponding .txt
        rel = wav.relative_to(DATA_DIR)
        txt = DATA_DIR / rel.with_suffix(".txt")
        if not txt.exists():
            continue
        stem = rel.with_suffix("")
        stems.append(stem.as_posix())
    return stems


def test_decode_sample_wavs_aggregate():
    matched_total = 0
    expected_total = 0
    for stem in list_all_stems():
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"

        audio = read_wav(str(wav_path))
        results = decode_full_period(audio)

        expected_records = parse_expected(txt_path)
        matched = check_decodes(results, expected_records)

        matched_total += matched
        expected_total += len(expected_records)

    assert expected_total > 0, "No sample records found"
    ratio = matched_total / expected_total
    assert ratio >= FULL_MIN_RATIO, f"Aggregate decode ratio {ratio:.3f} < {FULL_MIN_RATIO:.3f}"
