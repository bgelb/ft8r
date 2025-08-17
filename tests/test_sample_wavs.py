import re
from pathlib import Path

import pytest

from demod import decode_full_period
from utils import read_wav
from tests.utils import DEFAULT_DT_EPS, DEFAULT_FREQ_EPS

# Directory containing sample WAVs and accompanying TXT files from ft8_lib
DATA_DIR = Path(__file__).resolve().parent.parent / "ft8_lib-2.0" / "test" / "wav"

# Minimum aggregate decode ratio required to pass across the full library
FULL_MIN_RATIO = 0.65
# Maximum proportion of wrong-text decodes among overlapping decodes (success+wrong)
FULL_MAX_FALSE_OVERLAP_RATIO = 0.18


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


def test_decode_sample_wavs_aggregate():
    matched_total = 0
    expected_total = 0
    wrong_total = 0
    produced_total = 0
    hard_crc_total = 0
    for stem in list_all_stems():
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"

        audio = read_wav(str(wav_path))
        results = decode_full_period(audio)

        expected_records = parse_expected(txt_path)
        matched = check_decodes(results, expected_records)
        wrong = find_wrong_text_decodes(results, expected_records)
        if wrong:
            stem = stem  # for clarity in print
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
    print(f"Full summary: produced={produced_total} expected={expected_total} successful={matched_total} false={wrong_total} hard_crc={hard_crc_total}")
    print(f"Full metrics: coverage_success_ratio={ratio:.3f} success_overlap_ratio={success_overlap_ratio:.3f} false_overlap_ratio={false_ratio:.3f}")
    assert ratio >= FULL_MIN_RATIO, f"Aggregate decode ratio {ratio:.3f} < {FULL_MIN_RATIO:.3f}"
    assert false_ratio <= FULL_MAX_FALSE_OVERLAP_RATIO, (
        f"False overlap ratio {false_ratio:.3f} > {FULL_MAX_FALSE_OVERLAP_RATIO:.3f}"
    )
