import pytest
import time

from demod import decode_full_period
from utils import read_wav
from tests.test_sample_wavs import (
    DATA_DIR,
    parse_expected,
    check_decodes,
    find_wrong_text_decodes,
    list_all_stems,
)


# Minimum aggregate decode ratio required to pass for the short regression
SHORT_MIN_RATIO = 0.68
# Maximum proportion of wrong-text decodes among overlapping decodes (success+wrong)
SHORT_MAX_FALSE_OVERLAP_RATIO = 0.18


def _short_sample_stems() -> list[str]:
    # Deterministic ~20% sampling of the full set
    stems = list_all_stems()
    return [s for i, s in enumerate(sorted(stems)) if i % 5 == 0]


def test_decode_sample_wavs_short_aggregate(ft8r_metrics):
    t0 = time.monotonic()
    matched_total = 0
    expected_total = 0
    wrong_total = 0
    produced_total = 0
    hard_crc_total = 0
    for stem in _short_sample_stems():
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"

        audio = read_wav(str(wav_path))
        results = decode_full_period(audio)

        expected_records = parse_expected(txt_path)
        matched = check_decodes(results, expected_records)
        wrong = find_wrong_text_decodes(results, expected_records)
        if wrong:
            # Aid debugging by logging a few mismatches per file
            print(f"{stem}: {len(wrong)} wrong-text decodes")
            for rec, (msg, dt_e, freq_e) in wrong[:3]:
                print(
                    f"  dt={rec['dt']:.3f}s freq={rec['freq']:.1f}Hz got='{rec['message']}' expected='{msg}'"
                )

        matched_total += matched
        expected_total += len(expected_records)
        wrong_total += len(wrong)
        produced_total += len(results)
        hard_crc_total += sum(1 for r in results if r.get("method") == "hard")

    assert expected_total > 0, "No sample records found"
    ratio = matched_total / expected_total
    overlap = matched_total + wrong_total
    false_ratio = (wrong_total / overlap) if overlap else 0.0
    success_overlap_ratio = (matched_total / overlap) if overlap else 0.0
    # update session-level metrics for CI summary
    ft8r_metrics["matched"] += matched_total
    ft8r_metrics["total"] += expected_total
    print(f"Short summary: produced={produced_total} expected={expected_total} successful={matched_total} false={wrong_total} hard_crc={hard_crc_total}")
    print(f"Short metrics: coverage_success_ratio={ratio:.3f} success_overlap_ratio={success_overlap_ratio:.3f} false_overlap_ratio={false_ratio:.3f}")
    # Persist detailed metrics for CI PR comment
    try:
        import json, os
        os.makedirs(".tmp", exist_ok=True)
        total_decodes = int(produced_total)
        correct_decodes = int(matched_total)
        false_decodes = int(total_decodes - correct_decodes)
        total_signals = int(expected_total)
        decode_rate = (correct_decodes / total_signals) if total_signals else 0.0
        false_decode_rate = (false_decodes / total_decodes) if total_decodes else 0.0
        duration_sec = time.monotonic() - t0
        with open(".tmp/ft8r_short_metrics.json", "w") as f:
            json.dump({
                "total_decodes": total_decodes,
                "correct_decodes": correct_decodes,
                "false_decodes": false_decodes,
                "total_signals": total_signals,
                "decode_rate": decode_rate,
                "false_decode_rate": false_decode_rate,
                "duration_sec": duration_sec,
            }, f, indent=2)
    except Exception:
        pass
    assert ratio >= SHORT_MIN_RATIO, f"Short aggregate decode ratio {ratio:.3f} < {SHORT_MIN_RATIO:.3f}"
    assert false_ratio <= SHORT_MAX_FALSE_OVERLAP_RATIO, (
        f"Short false overlap ratio {false_ratio:.3f} > {SHORT_MAX_FALSE_OVERLAP_RATIO:.3f}"
    )
