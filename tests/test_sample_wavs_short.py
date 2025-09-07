import pytest
import time

from demod import decode_full_period
from utils import read_wav
from tests.test_sample_wavs import (
    DATA_DIR,
    parse_expected,
    list_all_stems,
    _strict_eps,
    assert_no_unexpected_duplicates,
)


# Minimum aggregate decode ratio required to pass for the short regression
# Tightened to track recent gains while allowing ~2% headroom
# Per-file normalized decode rate ratchet (counts only texts present in each file)
SHORT_MIN_RATIO = 0.69
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
    # Aggregate per-file metrics to ensure rates are bounded in [0,1]
    decoded_total = 0
    expected_total = 0
    correct_total = 0
    false_total = 0
    raw_decodes = 0
    hard_crc_total = 0
    stems = _short_sample_stems()
    # Additional breakdown among deduped correct decodes
    strict_dedup_first = 0
    text_only_dedup_first = 0
    # Track text-only dt/freq errors against the closest golden pair
    text_only_dt_errors: list[float] = []
    text_only_df_errors: list[float] = []
    # Accumulate golden dt/freq pairs per message across all sampled files
    expected_map: dict[str, list[tuple[float, float]]] = {}
    # Collect all decodes by payload bits to analyze dedup selection
    groups: dict[str, list[dict]] = {}
    # Flat list of all golden (dt,freq) pairs across files/messages
    expected_pairs_all: list[tuple[float, float]] = []
    for stem in stems:
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"

        audio = read_wav(str(wav_path))
        results = decode_full_period(audio, include_bits=True)
        assert_no_unexpected_duplicates(results)
        raw_decodes += len(results)
        hard_crc_total += sum(1 for r in results if r.get("method") == "hard")
        # Deduplicate by payload bits within this file
        decoded_map_file: dict[str, str] = {}
        for r in results:
            key = r.get("bits") or r["message"]
            decoded_map_file.setdefault(key, r["message"])  # keep first text
        # Expected signals for this file
        expected_records = parse_expected(txt_path)
        expected_texts = {msg for (msg, _dt, _freq) in expected_records}
        # Tally per-file
        texts = set(decoded_map_file.values())
        correct = len(texts & expected_texts)
        false = len(texts - expected_texts)
        correct_total += correct
        false_total += false
        decoded_total += len(texts)
        expected_total += len(expected_texts)
        # Collect for supplemental analysis only
        for (msg, _dt, _freq) in expected_records:
            expected_map.setdefault(msg, []).append((_dt, _freq))
        for r in results:
            bits = r.get("bits") or r["message"]
            groups.setdefault(bits, []).append(r)
        for (_m, _dt, _fq) in expected_records:
            expected_pairs_all.append((_dt, _fq))

    assert expected_total > 0, "No sample records found"
    total_decodes = decoded_total
    total_signals = expected_total
    correct_decodes = correct_total
    false_decodes = false_total
    decode_rate = correct_total / expected_total if expected_total else 0.0
    false_decode_rate = false_total / decoded_total if decoded_total else 0.0
    # update session-level metrics for CI summary
    ft8r_metrics["matched"] += correct_decodes
    ft8r_metrics["total"] += total_signals
    print(
        f"Short summary: "
        f"raw={raw_decodes} (pre-dedup decodes) "
        f"unique={total_decodes} (dedup by payload bits) "
        f"expected={total_signals} (distinct golden texts) "
        f"correct={correct_decodes} (unique with text in golden) "
        f"false={false_decodes} (unique - correct) "
        f"hard_crc={hard_crc_total} (#hard-decision CRC passes)"
    )
    print(
        f"Short metrics: decode_rate={decode_rate:.3f} false_decode_rate={false_decode_rate:.3f}"
    )
    # Classify deduped correct decodes using the first occurrence per payload group
    if groups:
        expected_set = set(expected_map.keys())
        dt_eps, fq_eps = _strict_eps()
        # Helper to test strictness against any golden (dt,freq) for a text
        def _is_strict(rec: dict) -> bool:
            pairs = expected_map.get(rec.get("message", "")) or []
            return any(
                abs(rec.get("dt", 0.0) - dt) < dt_eps and abs(rec.get("freq", 0.0) - fq) < fq_eps
                for (dt, fq) in pairs
            )
        # Helper to compute error vs closest golden pair (normalized by eps)
        def _closest_errors(rec: dict) -> tuple[float, float]:
            pairs = expected_map.get(rec.get("message", "")) or []
            if not pairs:
                return 0.0, 0.0
            best = None
            for dt, fq in pairs:
                ddt = abs(rec.get("dt", 0.0) - dt)
                dfq = abs(rec.get("freq", 0.0) - fq)
                score = (ddt / dt_eps) ** 2 + (dfq / fq_eps) ** 2
                if best is None or score < best[0]:
                    best = (score, ddt, dfq)
            return best[1], best[2]  # type: ignore[index]

        false_near = 0
        false_far = 0
        # Evaluate per unique payload group
        for bits, recs in groups.items():
            # Only consider groups that are counted as correct (text in expected)
            txt = recs[0].get("message")
            if txt not in expected_set:
                # false decode; assess proximity to any golden pair
                first = recs[0]
                near = any(
                    abs(first.get("dt", 0.0) - dt) < dt_eps
                    and abs(first.get("freq", 0.0) - fq) < fq_eps
                    for (dt, fq) in expected_pairs_all
                )
                if near:
                    false_near += 1
                else:
                    false_far += 1
                continue
            # First policy: keep first occurrence
            first = recs[0]
            first_strict = _is_strict(first)
            if first_strict:
                strict_dedup_first += 1
            else:
                text_only_dedup_first += 1
                ddt, dfq = _closest_errors(first)
                text_only_dt_errors.append(ddt)
                text_only_df_errors.append(dfq)

    if correct_decodes:
        total_correct = strict_dedup_first + text_only_dedup_first
        print(
            f"Short strict/text breakdown (among correct): "
            f"strict={strict_dedup_first}/{total_correct} ({(strict_dedup_first/total_correct):.3f}) "
            f"(text match + |dt|<{dt_eps:.3f}s & |df|<{fq_eps:.3f}Hz), "
            f"text_only={text_only_dedup_first}/{total_correct} ({(text_only_dedup_first/total_correct):.3f}) "
            f"(text match only; dt/freq outside eps)"
        )
        # Text-only error statistics
        if text_only_dedup_first:
            import numpy as _np
            dt_mean = float(_np.mean(text_only_dt_errors))
            dt_max = float(_np.max(text_only_dt_errors))
            df_mean = float(_np.mean(text_only_df_errors))
            df_max = float(_np.max(text_only_df_errors))
            print(
                f"Short text-only error: dt_mean={dt_mean:.3f}s dt_max={dt_max:.3f}s; "
                f"df_mean={df_mean:.3f}Hz df_max={df_max:.3f}Hz"
            )
    # False decode proximity breakdown
    if false_decodes:
        total_false = false_near + false_far
        if total_false:
            print(
                f"Short false breakdown: near={false_near}/{total_false} "
                f"(within |dt|<{dt_eps:.3f}s & |df|<{fq_eps:.3f}Hz), "
                f"far={false_far}/{total_false}"
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
                "strict_matches": int(strict_dedup_first),
                "text_only_matches": int(text_only_dedup_first),
                "false_near": int(false_near),
                "false_far": int(false_far),
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
