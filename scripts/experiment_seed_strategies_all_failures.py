#!/usr/bin/env python3
"""
Across all default-failing expected records, compare seeded decode success for:
  A) Exact golden (dt_target, freq_target)
  B) Nearest coarse bin (dt_target, round(freq_target/6.25)*6.25)

Reports overall counts and percentages.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

from utils import read_wav, TONE_SPACING_IN_HZ
from demod import fine_sync_candidate, soft_demod, naive_hard_decode, ldpc_decode, decode_full_period
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


def crc_from_seed(audio, dt: float, freq: float) -> bool:
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
    wavs = sorted(DATA_DIR.rglob("*.wav"))
    total_expected = 0
    default_ok = 0
    failing = []  # (stem, msg, dt, freq)
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

    A_ok = 0
    B_ok = 0
    for stem, msg, dt, freq in failing:
        audio = read_wav(str(DATA_DIR / stem))
        # A) exact golden
        if crc_from_seed(audio, dt, freq):
            A_ok += 1
        # B) nearest coarse bin
        coarse = round(freq / TONE_SPACING_IN_HZ) * TONE_SPACING_IN_HZ
        if crc_from_seed(audio, dt, coarse):
            B_ok += 1

    out = {
        'total_expected': total_expected,
        'default_decoded': default_ok,
        'default_failed': len(failing),
        'seed_exact_ok': A_ok,
        'pct_seed_exact_ok': round(100.0 * A_ok / max(1, len(failing)), 1),
        'seed_coarse_ok': B_ok,
        'pct_seed_coarse_ok': round(100.0 * B_ok / max(1, len(failing)), 1),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

