#!/usr/bin/env python3
"""
Experiment: Among seed-recovered cases (default failed; brute-force seeding succeeded),
what fraction would decode if the coarse search had included the correct coarse
frequency bin near the expected (dt,freq)?

Method: For each case in .tmp/ft8r_seed_recovered_core.json, run the refined
pipeline starting from (dt_target, coarse_freq = round(freq_target/6.25)*6.25)
and check CRC. Aggregate overall and for the CoarseMiss subset.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import read_wav, TONE_SPACING_IN_HZ
from demod import fine_sync_candidate, soft_demod, naive_hard_decode, ldpc_decode
from utils import check_crc


DATA_DIR = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"
SEED_JSON = Path(".tmp/ft8r_seed_recovered_core.json")
SUMMARY_TSV = Path(".tmp/fine_sync_core_summary.tsv")


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


def load_coarse_miss_flags() -> Dict[Tuple[str, float, float], bool]:
    flags: Dict[Tuple[str, float, float], bool] = {}
    if not SUMMARY_TSV.exists():
        return flags
    with SUMMARY_TSV.open() as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            stem = row['stem']
            dt = float(row['dt'])
            freq = float(row['freq'])
            labels = row.get('labels', '')
            flags[(stem, dt, freq)] = 'CoarseMiss' in labels
    return flags


def main() -> int:
    if not SEED_JSON.exists():
        raise SystemExit("Missing .tmp/ft8r_seed_recovered_core.json. Run experiment_seed_recovered_core.py first.")
    data = json.loads(SEED_JSON.read_text())
    details = data.get('details', [])
    coarse_miss = load_coarse_miss_flags()

    total = 0
    ok_total = 0
    miss_total = 0
    miss_ok = 0

    for d in details:
        stem = d['stem']
        exp = d['expected']
        dt = float(exp['dt'])
        freq = float(exp['freq'])
        wav = DATA_DIR / stem
        audio = read_wav(str(wav))

        coarse_freq = round(freq / TONE_SPACING_IN_HZ) * TONE_SPACING_IN_HZ
        ok = crc_from_seed(audio, dt, coarse_freq)

        total += 1
        ok_total += int(ok)
        if coarse_miss.get((stem, dt, freq), True):
            miss_total += 1
            miss_ok += int(ok)

    print(json.dumps({
        'total_cases': total,
        'ok_if_coarse_picked': ok_total,
        'pct_ok_if_coarse_picked': round(100.0 * ok_total / max(1, total), 1),
        'coarse_miss_cases': miss_total,
        'coarse_miss_ok_if_picked': miss_ok,
        'pct_coarse_miss_ok_if_picked': round(100.0 * miss_ok / max(1, miss_total), 1),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

