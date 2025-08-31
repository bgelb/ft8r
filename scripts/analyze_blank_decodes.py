#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, List

import numpy as np

from utils import read_wav, TONE_SPACING_IN_HZ, FT8_SYMBOL_LENGTH_IN_SEC, COSTAS_START_OFFSET_SEC
from demod import decode_full_period, fine_sync_candidate, soft_demod, ldpc_decode, BASEBAND_RATE_HZ
from tests.test_sample_wavs import parse_expected
from utils import check_crc, decode77


DATA_DIR = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"


def nearest_expected(expected: List[Tuple[str, float, float]], dt: float, fq: float) -> Tuple[str | None, float, float, float, float]:
    best = None
    for msg, e_dt, e_fq in expected:
        ddt = abs(dt - e_dt)
        dfq = abs(fq - e_fq)
        score = (ddt / FT8_SYMBOL_LENGTH_IN_SEC) ** 2 + (dfq / TONE_SPACING_IN_HZ) ** 2
        if best is None or score < best[0]:
            best = (score, msg, ddt, dfq, e_dt, e_fq)
    if best is None:
        return None, math.inf, math.inf, math.inf, math.inf
    return best[1], best[2], best[3], best[4], best[5]


def golden_seed_redecode(audio, e_dt: float, e_fq: float) -> Tuple[bool, str | None]:
    # Ensure bounds similar to main pipeline
    sample_rate = audio.sample_rate_in_hz
    sym_len = int(sample_rate / TONE_SPACING_IN_HZ)
    start = int(round((e_dt + COSTAS_START_OFFSET_SEC) * sample_rate))
    end = start + sym_len * 79
    margin = int(round(10 * sample_rate / BASEBAND_RATE_HZ))
    if start - margin < 0 or end + margin > len(audio.samples):
        return False, None
    bb, dt_f, fq_f = fine_sync_candidate(audio, e_fq, e_dt)
    llrs = soft_demod(bb)
    bits = ldpc_decode(llrs)
    if not check_crc(bits):
        return False, None
    text = decode77(bits[:77])
    return (text.strip() != ""), text


def main():
    blanks = []
    files = sorted(DATA_DIR.rglob("*.wav"))
    for wav in files:
        txt = wav.with_suffix('.txt')
        if not txt.exists():
            continue
        expected = parse_expected(txt)
        audio = read_wav(str(wav))
        decs = decode_full_period(audio, include_bits=True)
        texts = set((d.get('message') or '') for d in decs)
        for r in decs:
            if (r.get('message') or '') == '':
                msg_near, ddt, dfq, e_dt, e_fq = nearest_expected(expected, r['dt'], r['freq'])
                ok, text = (False, None)
                # Only try golden-seeded retry when we have a nearby expected
                if math.isfinite(e_dt) and math.isfinite(e_fq):
                    ok, text = golden_seed_redecode(audio, e_dt, e_fq)
                blanks.append({
                    'file': wav.name,
                    'score': float(r.get('score', 0.0)),
                    'dt': float(r['dt']),
                    'freq': float(r['freq']),
                    'near_msg': msg_near,
                    'ddt': float(ddt),
                    'dfq': float(dfq),
                    'e_dt': float(e_dt),
                    'e_fq': float(e_fq),
                    'has_correct_in_results': bool(msg_near in texts if msg_near else False),
                    'seed_recovered': bool(ok),
                    'seed_text': text,
                })

    total = len(blanks)
    near_eps_dt = FT8_SYMBOL_LENGTH_IN_SEC
    near_eps_fq = TONE_SPACING_IN_HZ
    near = [b for b in blanks if b['ddt'] < near_eps_dt and b['dfq'] < near_eps_fq]
    far = [b for b in blanks if not (b['ddt'] < near_eps_dt and b['dfq'] < near_eps_fq)]
    print(f"blank_total={total} near={len(near)} far={len(far)} (eps: dt<{near_eps_dt:.3f}s df<{near_eps_fq:.3f}Hz)")
    if near:
        dt_mean = float(np.mean([b['ddt'] for b in near])); dt_max = float(np.max([b['ddt'] for b in near]))
        df_mean = float(np.mean([b['dfq'] for b in near])); df_max = float(np.max([b['dfq'] for b in near]))
        print(f"near stats: dt_mean={dt_mean:.3f}s dt_max={dt_max:.3f}s; df_mean={df_mean:.3f}Hz df_max={df_max:.3f}Hz")
    recovered = [b for b in blanks if b['seed_recovered']]
    print(f"golden-seeded recovery: {len(recovered)}/{total}")
    for b in blanks[:8]:
        print(f"ex: {b['file']} dt={b['dt']:.3f} freq={b['freq']:.1f} score={b['score']:.2f} near='{b['near_msg']}' "
              f"e_dt={b['e_dt']:.3f} e_fq={b['e_fq']:.1f} ddt={b['ddt']:.3f}s dfq={b['dfq']:.3f}Hz "
              f"in_results={b['has_correct_in_results']} recovered={b['seed_recovered']} txt='{b['seed_text']}'")


if __name__ == '__main__':
    main()
