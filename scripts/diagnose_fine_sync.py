#!/usr/bin/env python3
"""
Diagnose why fine sync missed for a case that brute-force seeding could decode.

Inputs:
  -W/--wav-stem: Relative path under ft8_lib-2.0/test/wav (e.g., websdr_test4.wav)
  -d/--dt: Expected dt (seconds)
  -f/--freq: Expected freq (Hz)

Reports:
  - Whether coarse search produced a nearby candidate, and its rank/score
  - Fine-time/fine-freq results starting from the nearest coarse bin
  - Frequency energy curve around refined dt
  - deltas to expected, and whether LDPC passes at selected vs expected
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np

from utils import read_wav, TONE_SPACING_IN_HZ, COSTAS_START_OFFSET_SEC
from demod import (
    decode_full_period,
    downsample_to_baseband,
    fine_time_sync,
    fine_freq_sync,
    fine_sync_candidate,
    soft_demod,
    naive_hard_decode,
    ldpc_decode,
)
from search import find_candidates
from utils import check_crc


DATA_DIR = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"
DT_EPS = 0.2
FREQ_EPS = 1.0


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
    p = argparse.ArgumentParser()
    p.add_argument("--wav-stem", "-W", required=True)
    p.add_argument("--dt", "-d", type=float, required=True)
    p.add_argument("--freq", "-f", type=float, required=True)
    args = p.parse_args()

    wav = (DATA_DIR / args.wav_stem).with_suffix("")
    # Allow passing either with or without extension
    if wav.suffix == "":
        wav = wav.with_suffix(".wav")
    if not wav.exists():
        raise SystemExit(f"WAV not found: {wav}")

    audio = read_wav(str(wav))
    sample_rate = audio.sample_rate_in_hz

    # 1) Coarse candidate presence and rank
    sym_len = int(sample_rate / TONE_SPACING_IN_HZ)
    max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
    max_dt_samples = len(audio.samples) - int(sample_rate * COSTAS_START_OFFSET_SEC)
    max_dt_symbols = -(-max_dt_samples // sym_len)
    candidates = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=1.0)
    near: List[Tuple[int, float, float, float]] = []  # (rank, score, dt, freq)
    for i, (score, dt, freq) in enumerate(candidates):
        if abs(dt - args.dt) < DT_EPS and abs(freq - args.freq) < FREQ_EPS:
            near.append((i + 1, float(score), float(dt), float(freq)))

    # 2) Fine sync starting from nearest coarse bin to expected
    coarse_freq = round(args.freq / TONE_SPACING_IN_HZ) * TONE_SPACING_IN_HZ
    bb = downsample_to_baseband(audio, coarse_freq)
    dt_ref = fine_time_sync(bb, args.dt, search=10)
    df_ref = fine_freq_sync(bb, dt_ref, search_hz=5.0, step_hz=0.25)
    freq_ref = coarse_freq + df_ref

    # Refine again after retuning baseband (matches pipeline)
    bb2 = downsample_to_baseband(audio, freq_ref)
    dt_ref2 = fine_time_sync(bb2, dt_ref, search=4)

    # 3) Frequency energy curve around refined dt
    freqs = np.arange(-5.0, 5.0 + 0.25 / 2, 0.25)
    time_idx = np.arange(int(sample_rate * (1.0 / TONE_SPACING_IN_HZ))) / sample_rate
    # Compute energies by reusing fine_freq_sync implementation indirectly via sweeping
    # Use the already computed bb2 centered at freq_ref and dt_ref2
    energies = []
    for df in freqs:
        e = fine_freq_sync(bb2, dt_ref2, search_hz=0.0, step_hz=1.0)  # trivial call for structure
        # Overwrite with direct tone computation to avoid extra allocations
        # Here, we measure peak selection by asking fine_freq_sync at a single df
        energies.append(df)  # placeholder to keep lengths consistent
    # Instead of abusing internals, re-call fine_freq_sync over the whole grid
    energies = []
    # fine_freq_sync returns the best offset; to get full curve we duplicate its body inline
    sym_len = int(bb2.sample_rate_in_hz / TONE_SPACING_IN_HZ)
    seg = (bb2.samples[int(round((dt_ref2 + COSTAS_START_OFFSET_SEC) * bb2.sample_rate_in_hz))
            : int(round((dt_ref2 + COSTAS_START_OFFSET_SEC) * bb2.sample_rate_in_hz)) + sym_len * 79]
           ).reshape(79, sym_len)
    bases0 = np.exp(-2j * np.pi * (np.arange(8) * TONE_SPACING_IN_HZ)[:, None] * (np.arange(sym_len) / bb2.sample_rate_in_hz)[None, :])
    pos_idx = np.array([*range(7), *range(36, 43), *range(72, 79)])
    tone_idx = np.array([2,5,6,1,0,3,4] * 3)
    time_idx2 = np.arange(sym_len) / bb2.sample_rate_in_hz
    for df in freqs:
        shifts = np.exp(-2j * np.pi * df * time_idx2)
        bases = bases0 * shifts[None, :]
        resp = np.abs(bases @ seg.T) ** 2
        energies.append(float(resp[tone_idx, :][:, pos_idx].sum()))
    energies = np.array(energies)
    max_i = int(np.argmax(energies))
    peak_df = float(freqs[max_i])
    peak_ratio = float(energies[max_i] / (energies.mean() + 1e-12))

    # 4) CRC results at refined vs expected
    ok_ref = seeded_ok(audio, dt_ref2, freq_ref)
    ok_expected = seeded_ok(audio, args.dt, args.freq)

    # 5) Print summary
    print("WAV:", str(wav.relative_to(DATA_DIR)))
    print(f"Expected: dt={args.dt:.3f}  freq={args.freq:.2f}")
    if near:
        rank, score, dt_n, fq_n = near[0]
        print(f"Coarse:   found near expected  rank={rank} score={score:.3f} dt={dt_n:.3f} freq={fq_n:.2f}")
    else:
        print("Coarse:   no candidate within eps of expected (dt,freq)")
    print(f"FineRef:  dt={dt_ref2:.3f}  freq={freq_ref:.2f}  CRC={'OK' if ok_ref else 'FAIL'}  Δdt={dt_ref2-args.dt:+.3f}  Δf={freq_ref-args.freq:+.2f}")
    print(f"FreqCurve: peak_df={peak_df:+.2f} Hz  peak/mean={peak_ratio:.2f}")
    print(f"Expected-seeded CRC: {'OK' if ok_expected else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

