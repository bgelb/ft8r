import pytest
import time

from demod import decode_full_period
from utils import read_wav
from tests.test_sample_wavs import (
    DATA_DIR,
    parse_expected,
    list_all_stems,
)


# Minimum aggregate decode ratio required to pass for the short regression
SHORT_MIN_RATIO = 0.68
# Maximum proportion of wrong-text decodes among overlapping decodes (success+wrong)
SHORT_MAX_FALSE_OVERLAP_RATIO = 0.25


def _short_sample_stems() -> list[str]:
    # Deterministic ~20% sampling of the full set
    stems = list_all_stems()
    return [s for i, s in enumerate(sorted(stems)) if i % 5 == 0]


def test_decode_sample_wavs_short_aggregate(ft8r_metrics):
    t0 = time.monotonic()
    # We ignore dt/freq and deduplicate strictly by payload bits,
    # while counting correctness by text match to golden.
    decoded_map: dict[str, str] = {}
    expected_set: set[str] = set()
    raw_decodes = 0
    hard_crc_total = 0
    stems = _short_sample_stems()
    for stem in stems:
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"

        audio = read_wav(str(wav_path))
        results = decode_full_period(audio, include_bits=True)
        raw_decodes += len(results)
        hard_crc_total += sum(1 for r in results if r.get("method") == "hard")
        # Deduplicate by payload bits; store corresponding text
        for r in results:
            key = r.get("bits") or r["message"]
            decoded_map.setdefault(key, r["message"])  # keep first text

        # Expected signals from golden text
        expected_records = parse_expected(txt_path)
        for (msg, _dt, _freq) in expected_records:
            expected_set.add(msg)

    assert len(expected_set) > 0, "No sample records found"
    total_decodes = len(decoded_map)
    total_signals = len(expected_set)
    # Count each unique decoded payload as correct if its text matches golden
    correct_decodes = sum(1 for txt in decoded_map.values() if txt in expected_set)
    false_decodes = total_decodes - correct_decodes
    decode_rate = correct_decodes / total_signals if total_signals else 0.0
    false_decode_rate = false_decodes / total_decodes if total_decodes else 0.0
    # update session-level metrics for CI summary
    ft8r_metrics["matched"] += correct_decodes
    ft8r_metrics["total"] += total_signals
    print(
        f"Short summary: raw={raw_decodes} unique={total_decodes} expected={total_signals} "
        f"correct={correct_decodes} false={false_decodes} hard_crc={hard_crc_total}"
    )
    print(
        f"Short metrics: decode_rate={decode_rate:.3f} false_decode_rate={false_decode_rate:.3f}"
    )
    # Persist detailed metrics for CI PR comment
    try:
        import json, os
        os.makedirs(".tmp", exist_ok=True)
        duration_sec = time.monotonic() - t0
        num_files = len(stems)
        avg_runtime = (duration_sec / num_files) if num_files else 0.0
        with open(".tmp/ft8r_short_metrics.json", "w") as f:
            json.dump({
                "total_decodes": int(total_decodes),
                "correct_decodes": int(correct_decodes),
                "false_decodes": int(false_decodes),
                "total_signals": int(total_signals),
                "decode_rate": float(decode_rate),
                "false_decode_rate": float(false_decode_rate),
                "duration_sec": float(duration_sec),
                "num_files": int(num_files),
                "avg_runtime_per_file_sec": float(avg_runtime),
            }, f, indent=2)
    except Exception:
        pass
    assert decode_rate >= SHORT_MIN_RATIO, f"Short aggregate decode rate {decode_rate:.3f} < {SHORT_MIN_RATIO:.3f}"
    assert false_decode_rate <= SHORT_MAX_FALSE_OVERLAP_RATIO, (
        f"Short false decode rate {false_decode_rate:.3f} > {SHORT_MAX_FALSE_OVERLAP_RATIO:.3f}"
    )
