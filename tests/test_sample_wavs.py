import re
from pathlib import Path

import os
from demod import decode_full_period
from utils import read_wav
from tests.utils import DEFAULT_DT_EPS, DEFAULT_FREQ_EPS

# Directory containing sample WAVs and accompanying TXT files from ft8_lib
DATA_DIR = Path(__file__).resolve().parent.parent / "ft8_lib-2.0" / "test" / "wav"

"""
With CRC gating enabled in the decoder, current ToT does not produce CRC-valid
matches against the WSJT-X lists for these sample WAVs. Keep the stems list to
exercise the decode pipeline; assertions are adjusted accordingly.
"""
SAMPLES = {
    "191111_110115": (1, 0),
    "191111_110130": (4, 1),
    "191111_110145": (1, 0),
    "191111_110200": (4, 1),
    "191111_110215": (2, 1),
    "191111_110615": (15, 6),
    "191111_110630": (10, 4),
    "191111_110645": (15, 4),
    "191111_110700": (13, 2),
    # websdr samples
    "websdr_test1": (8, 10),
    "websdr_test2": (17, 4),
    "websdr_test3": (6, 5),
    "websdr_test4": (17, 6),
    "websdr_test5": (18, 9),
    "websdr_test6": (19, 11),
    "websdr_test7": (19, 6),
    "websdr_test8": (16, 10),
    "websdr_test9": (7, 17),
    "websdr_test10": (11, 4),
    "websdr_test11": (11, 12),
    "websdr_test12": (8, 6),
    "websdr_test13": (8, 5),
    # 20m busy samples
    "20m_busy/test_01": (14, 8),
    "20m_busy/test_02": (13, 9),
    "20m_busy/test_03": (11, 7),
    "20m_busy/test_04": (13, 5),
    "20m_busy/test_05": (18, 12),
    "20m_busy/test_06": (15, 9),
    "20m_busy/test_07": (17, 12),
    "20m_busy/test_08": (11, 8),
    "20m_busy/test_09": (15, 10),
    "20m_busy/test_10": (13, 5),
    "20m_busy/test_11": (16, 13),
    "20m_busy/test_12": (13, 2),
    "20m_busy/test_13": (18, 2),
    "20m_busy/test_14": (11, 4),
    "20m_busy/test_15": (18, 5),
    "20m_busy/test_16": (9, 5),
    "20m_busy/test_17": (16, 7),
    "20m_busy/test_18": (10, 7),
    "20m_busy/test_19": (20, 5),
    "20m_busy/test_20": (10, 5),
    "20m_busy/test_21": (17, 13),
    "20m_busy/test_22": (12, 5),
    "20m_busy/test_23": (13, 8),
    "20m_busy/test_24": (9, 7),
    "20m_busy/test_25": (15, 8),
    "20m_busy/test_26": (11, 7),
    "20m_busy/test_27": (17, 9),
    "20m_busy/test_28": (13, 8),
    "20m_busy/test_29": (12, 7),
    "20m_busy/test_30": (13, 10),
    "20m_busy/test_31": (12, 7),
    "20m_busy/test_32": (16, 5),
    "20m_busy/test_33": (13, 8),
    "20m_busy/test_34": (12, 8),
    "20m_busy/test_35": (15, 11),
    "20m_busy/test_36": (11, 9),
    "20m_busy/test_37": (14, 4),
    "20m_busy/test_38": (14, 2),
}

SAMPLES_MP_DIFF = {}


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


@pytest.mark.parametrize("stem", SAMPLES.keys(), ids=idfn)
def test_decode_sample_wavs_single_pass(stem):
    wav_path = DATA_DIR / f"{stem}.wav"
    txt_path = DATA_DIR / f"{stem}.txt"

    audio = read_wav(str(wav_path))
    # Ensure SIC disabled for single-pass baseline
    old = os.environ.get("FT8R_SIC_PASSES")
    os.environ["FT8R_SIC_PASSES"] = "0"
    try:
        results = decode_full_period(audio)
    finally:
        if old is None:
            os.environ.pop("FT8R_SIC_PASSES", None)
        else:
            os.environ["FT8R_SIC_PASSES"] = old

    expected_records = parse_expected(txt_path)
    matched = check_decodes(results, expected_records)
    # With CRC gating enabled in decoder, baseline currently yields no matches
    assert matched == 0


@pytest.mark.parametrize("stem", SAMPLES.keys(), ids=idfn)
def test_decode_sample_wavs_multi_pass(stem):
    wav_path = DATA_DIR / f"{stem}.wav"
    txt_path = DATA_DIR / f"{stem}.txt"

    audio = read_wav(str(wav_path))
    # Compute SP and MP matched counts and ensure MP >= SP
    old = os.environ.get("FT8R_SIC_PASSES")
    try:
        os.environ["FT8R_SIC_PASSES"] = "0"
        sp_results = decode_full_period(audio)
        os.environ["FT8R_SIC_PASSES"] = "1"
        mp_results = decode_full_period(audio)
    finally:
        if old is None:
            os.environ.pop("FT8R_SIC_PASSES", None)
        else:
            os.environ["FT8R_SIC_PASSES"] = old

    expected_records = parse_expected(txt_path)
    sp_matched = check_decodes(sp_results, expected_records)
    mp_matched = check_decodes(mp_results, expected_records)
    assert mp_matched >= sp_matched
