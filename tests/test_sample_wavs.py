import re
import time
from pathlib import Path

import pytest

from demod import decode_full_period
from utils import read_wav
from tests.utils import DEFAULT_DT_EPS, DEFAULT_FREQ_EPS, ft8code_bits, resolve_wsjt_binary

# Directory containing sample WAVs and accompanying TXT files from ft8_lib
DATA_DIR = Path(__file__).resolve().parent.parent / "ft8_lib-2.0" / "test" / "wav"

# Minimum aggregate decode ratio required to pass across the full library
FULL_MIN_RATIO = 0.65
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
    decoded_set: set[str] = set()
    expected_set: set[str] = set()
    decoded_texts: set[str] = set()
    raw_decodes = 0
    hard_crc_total = 0
    stems = list_all_stems()
    for stem in stems:
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"

        audio = read_wav(str(wav_path))
        results = decode_full_period(audio, include_bits=True)

        raw_decodes += len(results)
        hard_crc_total += sum(1 for r in results if r.get("method") == "hard")
        for r in results:
            decoded_set.add(r.get("bits") or r["message"])
            decoded_texts.add(r["message"]) 

        expected_records = parse_expected(txt_path)
        for (msg, _dt, _freq) in expected_records:
            expected_set.add(msg)

    assert len(expected_set) > 0, "No sample records found"
    total_decodes = len(decoded_set)
    total_signals = len(expected_set)
    correct_decodes = len(decoded_texts & expected_set)
    false_decodes = total_decodes - correct_decodes
    decode_rate = correct_decodes / total_signals if total_signals else 0.0
    false_decode_rate = false_decodes / total_decodes if total_decodes else 0.0
    print(
        f"Full summary: raw={raw_decodes} unique={total_decodes} expected={total_signals} "
        f"correct={correct_decodes} false={false_decodes} hard_crc={hard_crc_total}"
    )
    print(
        f"Full metrics: decode_rate={decode_rate:.3f} false_decode_rate={false_decode_rate:.3f}"
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
                "duration_sec": float(duration_sec),
                "num_files": int(num_files),
                "avg_runtime_per_file_sec": float(avg_runtime),
            }, f, indent=2)
    except Exception:
        pass
