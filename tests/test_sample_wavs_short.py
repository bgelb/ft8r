import pytest

from demod import decode_full_period
from utils import read_wav
from tests.test_sample_wavs import (
    DATA_DIR,
    parse_expected,
    check_decodes,
    list_all_stems,
)


# Minimum aggregate decode ratio required to pass for the short regression
SHORT_MIN_RATIO = 0.66


def _short_sample_stems() -> list[str]:
    # Deterministic ~20% sampling of the full set
    stems = list_all_stems()
    return [s for i, s in enumerate(sorted(stems)) if i % 5 == 0]


def test_decode_sample_wavs_short_aggregate(ft8r_metrics):
    matched_total = 0
    expected_total = 0
    for stem in _short_sample_stems():
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
    # update session-level metrics for CI summary
    ft8r_metrics["matched"] += matched_total
    ft8r_metrics["total"] += expected_total
    assert ratio >= SHORT_MIN_RATIO, f"Short aggregate decode ratio {ratio:.3f} < {SHORT_MIN_RATIO:.3f}"
