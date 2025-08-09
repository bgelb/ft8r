#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import importlib

from utils import read_wav, TONE_SPACING_IN_HZ, COSTAS_START_OFFSET_SEC
from search import find_candidates
from tests.test_sample_wavs import DATA_DIR, SAMPLES, parse_expected, check_decodes


def candidate_count_for_wav(wav_path: Path) -> int:
    audio = read_wav(str(wav_path))
    sample_rate = audio.sample_rate_in_hz
    sym_len = int(sample_rate / TONE_SPACING_IN_HZ)
    max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
    max_dt_samples = len(audio.samples) - int(sample_rate * COSTAS_START_OFFSET_SEC)
    max_dt_symbols = -(-max_dt_samples // sym_len)
    candidates = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=1.0)
    return len(candidates)


def matched_decodes_for_cap(wav_path: Path, cap: int) -> Tuple[int, int, int]:
    # Reload demod to pick up cap change
    import demod as demod_mod

    if cap <= 0:
        demod_mod._MAX_CANDIDATES = 0
    else:
        demod_mod._MAX_CANDIDATES = int(cap)
    demod_mod._DISABLE_LEGACY = False

    audio = read_wav(str(wav_path))
    results = demod_mod.decode_full_period(audio)
    txt_path = wav_path.with_suffix('.txt')
    expected_records = parse_expected(txt_path)
    matched = check_decodes(results, expected_records)
    return matched, len(expected_records), len(results)


def main():
    stems = list(SAMPLES.keys())
    summary: Dict[str, Dict] = {}

    print("Computing candidate counts...")
    counts: List[Tuple[str, int]] = []
    for stem in stems:
        wav_path = DATA_DIR / f"{stem}.wav"
        if not wav_path.exists():
            continue
        cnt = candidate_count_for_wav(wav_path)
        counts.append((stem, cnt))
        summary[stem] = {"candidates": cnt}

    counts.sort(key=lambda x: x[1], reverse=True)
    all_counts = np.array([c for _, c in counts], dtype=int)
    pct95 = int(np.percentile(all_counts, 95)) if len(all_counts) else 0
    pct99 = int(np.percentile(all_counts, 99)) if len(all_counts) else 0
    mx = int(all_counts.max()) if len(all_counts) else 0

    print(f"Candidate count stats: 95th={pct95}, 99th={pct99}, max={mx}")

    busiest = counts[:5]
    caps = [pct95, pct99, max(200, pct95 + 50), max(300, pct99 + 50), mx]
    caps = sorted(set(c for c in caps if c > 0))

    print("\nValidating decode match on busiest samples for caps:", caps)
    for stem, _ in busiest:
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = wav_path.with_suffix('.txt')
        summary[stem]["caps"] = {}
        for cap in caps:
            # Ensure fresh import each time
            if 'demod' in globals():
                importlib.reload(globals()['demod'])
            matched, expected_total, produced = matched_decodes_for_cap(wav_path, cap)
            summary[stem]["caps"][str(cap)] = {
                "matched": matched,
                "expected_total": expected_total,
                "produced": produced,
            }
            print(f"{stem}: cap={cap} matched={matched}/{expected_total} produced={produced}")

    out = Path("candidate_cap_eval.json")
    out.write_text(json.dumps({
        "counts_sorted": counts,
        "stats": {"p95": pct95, "p99": pct99, "max": mx},
        "details": summary,
    }, indent=2))
    print(f"\nSaved detailed results to {out}")


if __name__ == "__main__":
    main()


