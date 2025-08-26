#!/usr/bin/env python3
"""
Experiment: analyze non-decoding expected FT8 signals in the sample WAV library.

For each expected decode in the golden TXT that our default pipeline fails to
decode, try two diagnostics:

1) Lower search threshold: re-run decode_full_period with smaller thresholds
   (e.g., 0.9, 0.8, 0.7, 0.6, 0.5) and see if the expected record appears.

2) Brute-force fine frequency seed: bypass search and, for the expected (dt,freq),
   run fine sync + demod + LDPC at frequencies in +/-1.0 Hz around the expected
   in 0.1 Hz steps. Succeeds if CRC passes and decoded text matches expected.

Outputs a summary to stdout and writes a JSON report under .tmp/.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from utils import read_wav
from demod import (
    decode_full_period,
    fine_sync_candidate,
    soft_demod,
    naive_hard_decode,
    ldpc_decode,
)
from utils import decode77, check_crc


# Where to find sample WAV/TXT pairs
DATA_DIR = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"

# Matching epsilons to associate decodes with expected records
DT_EPS = 0.2  # seconds
FREQ_EPS = 1.0  # Hz


def parse_expected(path: Path) -> List[Tuple[str, float, float]]:
    records: List[Tuple[str, float, float]] = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "<" in line:
            continue
        m = re.match(r"\d+\s+(-?\d+)\s+([\d.-]+)\s+(\d+)\s+~\s+(.*)", line)
        if not m:
            continue
        _snr, dt, freq, msg = m.groups()
        # Trim auxiliary fields after two+ spaces
        msg = re.sub(r"\s{2,}.*", "", msg).strip()
        records.append((msg, float(dt), float(freq)))
    return records


def has_record(results: List[dict], target: Tuple[str, float, float]) -> bool:
    msg_t, dt_t, fq_t = target
    for r in results:
        if r.get("message") != msg_t:
            continue
        if abs(float(r.get("dt", 0.0)) - dt_t) < DT_EPS and abs(float(r.get("freq", 0.0)) - fq_t) < FREQ_EPS:
            return True
    return False


def seeded_decode(audio, dt: float, freq: float) -> bool:
    """Bypass search; seed with (dt,freq) and attempt CRC-passing decode."""
    try:
        bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
        llrs = soft_demod(bb)
        hard = naive_hard_decode(llrs)
        if check_crc(hard):
            return True
        soft_bits = ldpc_decode(llrs)
        return check_crc(soft_bits)
    except Exception:
        return False


@dataclass
class FailCase:
    stem: str
    expected: Tuple[str, float, float]
    threshold_recovered: float | None = None
    brute_recovered: bool = False


def main() -> int:
    if not DATA_DIR.exists():
        print(f"No sample directory found at {DATA_DIR}")
        return 2

    wavs = sorted(DATA_DIR.rglob("*.wav"))
    if not wavs:
        print(f"No .wav files under {DATA_DIR}. Populate the sample set first.")
        return 2

    total_expected = 0
    total_default_ok = 0
    fail_cases: List[FailCase] = []

    for wav in wavs:
        txt = wav.with_suffix(".txt")
        exp = parse_expected(txt)
        if not exp:
            continue
        audio = read_wav(str(wav))
        base_results = decode_full_period(audio)
        for rec in exp:
            total_expected += 1
            if has_record(base_results, rec):
                total_default_ok += 1
                continue
            fail_cases.append(FailCase(stem=str(wav.relative_to(DATA_DIR)), expected=rec))

    print(f"Total expected: {total_expected}")
    print(f"Decoded at default: {total_default_ok}")
    print(f"Failing at default: {len(fail_cases)}")

    # Diagnostics per failing record
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    brute_steps = [round(x * 0.1, 1) for x in range(-10, 11)]  # -1.0 .. +1.0 step 0.1

    recovered_by_thresh = 0
    recovered_by_brute = 0

    for fc in fail_cases:
        wav_path = DATA_DIR / fc.stem
        audio = read_wav(str(wav_path))
        msg_t, dt_t, fq_t = fc.expected
        # Threshold sweep
        for t in thresholds:
            res = decode_full_period(audio, threshold=t)
            if has_record(res, fc.expected):
                fc.threshold_recovered = float(t)
                recovered_by_thresh += 1
                break
        # Brute-force seed around expected freq
        if fc.threshold_recovered is None:
            ok = False
            for d in brute_steps:
                if seeded_decode(audio, dt_t, fq_t + d):
                    ok = True
                    break
            fc.brute_recovered = ok
            if ok:
                recovered_by_brute += 1

    print(f"Recovered by lowering threshold: {recovered_by_thresh}")
    print(f"Recovered by brute-force freq seed: {recovered_by_brute}")
    # Overlap stats
    both = sum(1 for fc in fail_cases if fc.threshold_recovered is not None and fc.brute_recovered)
    only_thresh = sum(1 for fc in fail_cases if fc.threshold_recovered is not None and not fc.brute_recovered)
    only_brute = sum(1 for fc in fail_cases if fc.threshold_recovered is None and fc.brute_recovered)
    none = sum(1 for fc in fail_cases if fc.threshold_recovered is None and not fc.brute_recovered)
    print(f"Breakdown: both={both} only_thresh={only_thresh} only_brute={only_brute} none={none}")

    # Persist JSON report
    out = {
        "total_expected": total_expected,
        "decoded_default": total_default_ok,
        "failing_default": len(fail_cases),
        "recovered_by_threshold": recovered_by_thresh,
        "recovered_by_bruteforce": recovered_by_brute,
        "both": both,
        "only_threshold": only_thresh,
        "only_bruteforce": only_brute,
        "none": none,
        "details": [
            {
                "stem": fc.stem,
                "expected": {
                    "message": fc.expected[0],
                    "dt": fc.expected[1],
                    "freq": fc.expected[2],
                },
                "threshold_recovered": fc.threshold_recovered,
                "bruteforce_recovered": fc.brute_recovered,
            }
            for fc in fail_cases
        ],
    }
    Path(".tmp").mkdir(exist_ok=True)
    with open(".tmp/ft8r_wav_failures_experiment.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

