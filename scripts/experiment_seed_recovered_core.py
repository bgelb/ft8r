#!/usr/bin/env python3
"""
Quick experiment: find expected TXT records that do NOT decode via the default
pipeline but DO decode when seeded at the expected (dt,freq) with a small
brute-force frequency dither.

Outputs .tmp/ft8r_seed_recovered_core.json with details for downstream analysis.
"""
from __future__ import annotations

import json
import re
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
from utils import check_crc


DATA_DIR = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"
DT_EPS = 0.2
FREQ_EPS = 1.0


def parse_expected(path: Path) -> List[Tuple[str, float, float]]:
    recs: List[Tuple[str, float, float]] = []
    if not path.exists():
        return recs
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "<" in line:
            continue
        m = re.match(r"\d+\s+(-?\d+)\s+([\d.-]+)\s+(\d+)\s+~\s+(.*)", line)
        if not m:
            continue
        _snr, dt, freq, msg = m.groups()
        msg = re.sub(r"\s{2,}.*", "", msg).strip()
        recs.append((msg, float(dt), float(freq)))
    return recs


def has_record(results: List[dict], msg: str, dt: float, freq: float) -> bool:
    for r in results:
        if r.get("message") != msg:
            continue
        if abs(float(r.get("dt", 0.0)) - dt) < DT_EPS and abs(float(r.get("freq", 0.0)) - freq) < FREQ_EPS:
            return True
    return False


def seeded_ok(audio, dt: float, freq: float) -> bool:
    try:
        bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
        llrs = soft_demod(bb)
        hard = naive_hard_decode(llrs)
        if check_crc(hard):
            return True
        soft = ldpc_decode(llrs)
        return check_crc(soft)
    except Exception:
        return False


def main() -> int:
    if not DATA_DIR.exists():
        print(f"No sample directory at {DATA_DIR}")
        return 2

    wavs = sorted(DATA_DIR.rglob("*.wav"))
    out_details = []
    total_expected = 0
    total_default_ok = 0
    total_failing = 0
    total_seed_recovered = 0

    for wav in wavs:
        txt = wav.with_suffix(".txt")
        expected = parse_expected(txt)
        if not expected:
            continue
        audio = read_wav(str(wav))
        base_results = decode_full_period(audio)
        for msg, dt, freq in expected:
            total_expected += 1
            if has_record(base_results, msg, dt, freq):
                total_default_ok += 1
                continue
            total_failing += 1
            ok = False
            for d in [round(x * 0.1, 1) for x in range(-10, 11)]:
                if seeded_ok(audio, dt, freq + d):
                    ok = True
                    break
            if ok:
                total_seed_recovered += 1
                out_details.append({
                    "stem": str(wav.relative_to(DATA_DIR)),
                    "expected": {"message": msg, "dt": dt, "freq": freq},
                })

    Path(".tmp").mkdir(exist_ok=True)
    out = {
        "total_expected": total_expected,
        "decoded_default": total_default_ok,
        "failing_default": total_failing,
        "seed_recovered": total_seed_recovered,
        "details": out_details,
    }
    (Path(".tmp") / "ft8r_seed_recovered_core.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "details"}, indent=2))
    print(f"Saved details: .tmp/ft8r_seed_recovered_core.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

