#!/usr/bin/env python3
"""
Evaluate coarse whitening/normalization on default-failing cases.

Steps:
 1) Build failing set with default settings.
 2) Re-run with FT8R_WHITEN_ENABLE=1 and report recovered count.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Tuple

from utils import read_wav
from demod import decode_full_period


DATA_DIR = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"
DT_EPS = 0.2
FREQ_EPS = 1.0


def parse_expected(path: Path) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "<" in line:
            continue
        m = re.match(r"\d+\s+(-?\d+)\s+([\d.-]+)\s+(\d+)\s+~\s+(.*)", line)
        if not m:
            continue
        _snr, dt, freq, msg = m.groups()
        msg = re.sub(r"\s{2,}.*", "", msg).strip()
        out.append((msg, float(dt), float(freq)))
    return out


def has_record(results: List[dict], msg: str, dt: float, freq: float) -> bool:
    for r in results:
        if r.get("message") != msg:
            continue
        if abs(float(r.get("dt", 0.0)) - dt) < DT_EPS and abs(float(r.get("freq", 0.0)) - freq) < FREQ_EPS:
            return True
    return False


def main() -> int:
    wavs = sorted(DATA_DIR.rglob("*.wav"))
    failing = []
    total_expected = 0
    default_ok = 0

    # Baseline set of failures
    os.environ.pop("FT8R_WHITEN_ENABLE", None)
    for wav in wavs:
        exp = parse_expected(wav.with_suffix('.txt'))
        if not exp:
            continue
        audio = read_wav(str(wav))
        base = decode_full_period(audio)
        for msg, dt, freq in exp:
            total_expected += 1
            if has_record(base, msg, dt, freq):
                default_ok += 1
            else:
                failing.append((str(wav.relative_to(DATA_DIR)), msg, dt, freq))

    # Enable whitening
    os.environ['FT8R_WHITEN_ENABLE'] = '1'
    recovered = 0
    for stem, msg, dt, freq in failing:
        audio = read_wav(str(DATA_DIR / stem))
        res = decode_full_period(audio)
        if has_record(res, msg, dt, freq):
            recovered += 1

    out = {
        'total_expected': total_expected,
        'default_ok': default_ok,
        'default_failed': len(failing),
        'recovered_with_whitening': recovered,
        'pct_recovered_from_failures': round(100.0 * recovered / max(1, len(failing)), 1),
        'improved_total_ok': default_ok + recovered,
        'pct_improved_total': round(100.0 * (default_ok + recovered) / max(1, total_expected), 1),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

