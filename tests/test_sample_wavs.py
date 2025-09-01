import re
import time
from pathlib import Path

import pytest

from demod import decode_full_period
from utils import read_wav, FT8_SYMBOL_LENGTH_IN_SEC, TONE_SPACING_IN_HZ
from tests.utils import DEFAULT_DT_EPS, DEFAULT_FREQ_EPS, ft8code_bits, resolve_wsjt_binary

# Directory containing sample WAVs and accompanying TXT files from ft8_lib
DATA_DIR = Path(__file__).resolve().parent.parent / "ft8_lib-2.0" / "test" / "wav"

# Minimum aggregate decode ratio required to pass across the full library
# Raised to reflect current performance with microsearch enabled
FULL_MIN_RATIO = 0.87
# Maximum false decode rate threshold (after deduping by payload)
FULL_MAX_FALSE_OVERLAP_RATIO = 0.25


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


def _strict_eps() -> tuple[float, float]:
    """Return (dt_eps, freq_eps) for strict classification.

    Relaxed to 1 symbol period and 1 tone bin.
    """
    return 1.0 * FT8_SYMBOL_LENGTH_IN_SEC, 1.0 * TONE_SPACING_IN_HZ


def find_wrong_text_decodes(results, expected):
    """Return list of decodes whose dt/freq match an expected record but text differs.

    A decode is considered overlapping an expected record if both ``dt`` and
    ``freq`` are within the default epsilons used by the tests. If such an
    overlap exists but the decoded ``message`` does not equal the expected
    ``msg``, the decode is flagged as a wrong-text decode.

    Returns a list of tuples ``(result, expected_record)`` where ``result`` is
    one element from ``results`` and ``expected_record`` is the matching
    ``(msg, dt, freq)`` from ``expected`` that it overlaps.
    """
    wrong: list[tuple[dict, tuple[str, float, float]]] = []
    for rec in results:
        for msg, dt, freq in expected:
            if (
                abs(rec["dt"] - dt) < DEFAULT_DT_EPS
                and abs(rec["freq"] - freq) < DEFAULT_FREQ_EPS
            ):
                if rec["message"] != msg:
                    wrong.append((rec, (msg, dt, freq)))
                # At most one expected overlap per decode
                break
    return wrong


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


def _bits_for_message(msg: str) -> str | None:
    # Internal-only path: this test no longer derives bits from external tools.
    return None


def test_decode_sample_wavs_aggregate():
    t0 = time.monotonic()
    decoded_map: dict[str, str] = {}
    expected_set: set[str] = set()
    raw_decodes = 0
    hard_crc_total = 0
    stems = list_all_stems()
    # Additional breakdown: classify unique (deduped) correct decodes into
    # strict vs text-only based on the dt/freq of the kept (first) instance,
    # and compare with an alternative "best" dedup policy.
    strict_dedup_first = 0
    text_only_dedup_first = 0
    # Track text-only dt/freq errors against the closest golden pair
    text_only_dt_errors: list[float] = []
    text_only_df_errors: list[float] = []
    # Accumulate all golden (dt,freq) pairs keyed by message text
    expected_map: dict[str, list[tuple[float, float]]] = {}
    # Group all decodes by payload for dedup analysis
    groups: dict[str, list[dict]] = {}
    # Flat list of all golden (dt,freq) pairs across files/messages
    expected_pairs_all: list[tuple[float, float]] = []

    for stem in stems:
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"

        audio = read_wav(str(wav_path))
        results = decode_full_period(audio, include_bits=True)

        raw_decodes += len(results)
        hard_crc_total += sum(1 for r in results if r.get("method") == "hard")
        for r in results:
            key = r.get("bits") or r["message"]
            decoded_map.setdefault(key, r["message"])  # keep first text

        expected_records = parse_expected(txt_path)
        for (msg, _dt, _freq) in expected_records:
            expected_set.add(msg)
            expected_map.setdefault(msg, []).append((_dt, _freq))
        # Capture all decodes for dedup analysis (grouped by payload)
        for r in results:
            bits = r.get("bits") or r["message"]
            groups.setdefault(bits, []).append(r)
        # Accumulate golden (dt,freq) pairs
        for (_m, _dt, _fq) in expected_records:
            expected_pairs_all.append((_dt, _fq))

    assert len(expected_set) > 0, "No sample records found"
    total_decodes = len(decoded_map)
    total_signals = len(expected_set)
    correct_decodes = sum(1 for txt in decoded_map.values() if txt in expected_set)
    false_decodes = total_decodes - correct_decodes
    decode_rate = correct_decodes / total_signals if total_signals else 0.0
    false_decode_rate = false_decodes / total_decodes if total_decodes else 0.0
    # Classify deduped correct decodes using the current (first-in-output) policy
    if decoded_map:
        dt_eps, fq_eps = _strict_eps()
        def _is_strict(rec: dict) -> bool:
            pairs = expected_map.get(rec.get("message", "")) or []
            return any(
                abs(rec.get("dt", 0.0) - dt) < dt_eps and abs(rec.get("freq", 0.0) - fq) < fq_eps
                for (dt, fq) in pairs
            )
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
        for bits, recs in groups.items():
            txt = recs[0].get("message")
            # If this group is false (text not expected), categorize by proximity
            if txt not in expected_set:
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
            first = recs[0]
            first_strict = _is_strict(first)
            if first_strict:
                strict_dedup_first += 1
            else:
                text_only_dedup_first += 1
                ddt, dfq = _closest_errors(first)
                text_only_dt_errors.append(ddt)
                text_only_df_errors.append(dfq)

    print(
        f"Full summary: "
        f"raw={raw_decodes} (pre-dedup decodes) "
        f"unique={total_decodes} (dedup by payload bits) "
        f"expected={total_signals} (distinct golden texts) "
        f"correct={correct_decodes} (unique with text in golden) "
        f"false={false_decodes} (unique - correct) "
        f"hard_crc={hard_crc_total} (#hard-decision CRC passes)"
    )
    print(
        f"Full metrics: decode_rate={decode_rate:.3f} false_decode_rate={false_decode_rate:.3f}"
    )
    # Supplemental breakdown among correct decodes after dedup
    if correct_decodes:
        total_correct = strict_dedup_first + text_only_dedup_first
        dt_eps, fq_eps = _strict_eps()
        print(
            f"Full strict/text breakdown (among correct): "
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
                f"Full text-only error: dt_mean={dt_mean:.3f}s dt_max={dt_max:.3f}s; "
                f"df_mean={df_mean:.3f}Hz df_max={df_max:.3f}Hz"
            )
    # False decode proximity breakdown
    if false_decodes:
        total_false = false_near + false_far
        if total_false:
            dt_eps, fq_eps = _strict_eps()
            print(
                f"Full false breakdown: near={false_near}/{total_false} "
                f"(within |dt|<{dt_eps:.3f}s & |df|<{fq_eps:.3f}Hz), "
                f"far={false_far}/{total_false}"
            )
    assert decode_rate >= FULL_MIN_RATIO, f"Aggregate decode rate {decode_rate:.3f} < {FULL_MIN_RATIO:.3f}"
    assert false_decode_rate <= FULL_MAX_FALSE_OVERLAP_RATIO, (
        f"False decode rate {false_decode_rate:.3f} > {FULL_MAX_FALSE_OVERLAP_RATIO:.3f}"
    )
    # Persist detailed metrics for CI PR comment
    try:
        import json, os
        os.makedirs(".tmp", exist_ok=True)
        duration_sec = time.monotonic() - t0
        num_files = len(stems)
        avg_runtime = (duration_sec / num_files) if num_files else 0.0
        with open(".tmp/ft8r_full_metrics.json", "w") as f:
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
