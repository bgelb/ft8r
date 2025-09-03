#!/usr/bin/env python3
"""
Compare recovery and cost of Light μ|LLR|-guided df microsearch vs Full (dt,df)
microsearch, restricted to the expected record's 100 Hz band.

Baseline failing set is computed with whitening + budgeted coarse selection.
Reports recovered counts and approximate compute (number of fine-sync attempts,
LDPC calls) and wall time.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

from utils import read_wav, TONE_SPACING_IN_HZ, COSTAS_START_OFFSET_SEC, decode77, check_crc
from demod import (
    downsample_to_baseband,
    fine_time_sync,
    fine_sync_candidate,
    soft_demod,
    naive_hard_decode,
    ldpc_decode,
)
from search import candidate_score_map


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


@dataclass
class Counters:
    fine_sync: int = 0
    ldpc_calls: int = 0


def try_seed(audio, dt_seed: float, freq_seed: float, counters: Counters, expected_msg: str) -> Tuple[bool, float]:
    """Try one seed: align, demod, CRC; return (ok, mu_abs)."""
    try:
        counters.fine_sync += 1
        bb, dt_f, freq_f = fine_sync_candidate(audio, freq_seed, dt_seed)
        llrs = soft_demod(bb)
        mu = float(np.mean(np.abs(llrs)))
        hard = naive_hard_decode(llrs)
        if check_crc(hard):
            txt = decode77(hard[:77])
            return (txt == expected_msg), mu
        counters.ldpc_calls += 1
        soft = ldpc_decode(llrs)
        if check_crc(soft):
            txt = decode77(soft[:77])
            return (txt == expected_msg), mu
        return (False, mu)
    except Exception:
        return (False, float("nan"))


def band_index(freq_hz: float) -> int:
    return int(freq_hz // 100.0)


def seeds_for_band(audio, expected_freq: float, N_top: int, include_runnerup: bool) -> List[Tuple[float, float]]:
    """Return up to N_top coarse (dt,freq) seeds within the expected 100 Hz band, plus an optional runner-up.

    Uses the coarse score map to pick seeds by score. The runner-up is the argmax
    over the band regardless of threshold and may duplicate a top seed (we de-dup).
    """
    sample_rate = audio.sample_rate_in_hz
    sym_len = int(sample_rate / TONE_SPACING_IN_HZ)
    max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
    max_dt_samples = len(audio.samples) - int(sample_rate * COSTAS_START_OFFSET_SEC)
    max_dt_symbols = -(-max_dt_samples // sym_len)
    scores, dts, freqs = candidate_score_map(audio, max_freq_bin, max_dt_symbols)

    b = band_index(expected_freq)
    band_j = np.where((freqs // 100.0).astype(int) == b)[0]
    if band_j.size == 0:
        return []
    # Top-N seeds by band-local maxima (approximate by taking the best dt per freq bin)
    seeds: List[Tuple[float, float]] = []
    # Collapse over dt to get best per freq in-band
    band_scores = scores[:, band_j]
    best_dt_idx = np.argmax(band_scores, axis=0)
    best_scores = band_scores[best_dt_idx, np.arange(band_j.size)]
    order = np.argsort(best_scores)[::-1]
    for idx in order[:N_top]:
        i = int(best_dt_idx[idx])
        j = int(band_j[idx])
        seeds.append((float(dts[i]), float(freqs[j])))

    if include_runnerup:
        # Global argmax in band
        bi, bj_rel = np.unravel_index(np.argmax(band_scores), band_scores.shape)
        j = int(band_j[int(bj_rel)])
        dt = float(dts[int(bi)])
        fq = float(freqs[j])
        if (dt, fq) not in seeds:
            seeds.append((dt, fq))

    return seeds


def light_microsearch(audio, expected_msg: str, expected_dt: float, expected_freq: float, counters: Counters) -> bool:
    # Seeds within expected band
    N = int(os.getenv("FT8R_MICRO_LIGHT_N_PER_BAND", "2"))
    include_runner = os.getenv("FT8R_MICRO_LIGHT_RUNNERUP", "1") not in ("0", "", "false", "False")
    seeds = seeds_for_band(audio, expected_freq, N, include_runner)
    if not seeds:
        return False
    # Refine dt once per seed and sweep df in a small range
    df_span = float(os.getenv("FT8R_MICRO_LIGHT_DF_SPAN", "1.0"))
    df_step = float(os.getenv("FT8R_MICRO_LIGHT_DF_STEP", "0.5"))
    mu_thresh = float(os.getenv("FT8R_MICRO_LIGHT_MU_THRESH", "0.35"))
    for dt_seed, fq_seed in seeds:
        try:
            bb0 = downsample_to_baseband(audio, fq_seed)
            dt_ref = fine_time_sync(bb0, dt_seed, search=10)
        except Exception:
            dt_ref = dt_seed
        dfs = np.arange(-df_span, df_span + 1e-6, df_step)
        best_mu = -1.0
        for df in dfs:
            ok, mu = try_seed(audio, dt_ref, fq_seed + float(df), counters, expected_msg)
            if ok:
                return True
            best_mu = max(best_mu, mu if mu == mu else -1.0)
        # μ-gate promotion: if best μ is strong, accept this seed as recovered (helps stats only if CRC passed; we don't force decode)
        if best_mu >= mu_thresh:
            # Not a CRC pass; continue to next seed
            pass
    return False


def full_microsearch(audio, expected_msg: str, expected_dt: float, expected_freq: float, counters: Counters) -> bool:
    N = int(os.getenv("FT8R_MICRO_FULL_N_PER_BAND", "2"))
    seeds = seeds_for_band(audio, expected_freq, N, include_runnerup=False)
    if not seeds:
        return False
    dt_span = float(os.getenv("FT8R_MICRO_FULL_DT_SPAN_SYM", "1.0"))
    dt_step = float(os.getenv("FT8R_MICRO_FULL_DT_STEP_SYM", "0.5"))
    df_span = float(os.getenv("FT8R_MICRO_FULL_DF_SPAN_HZ", "2.0"))
    df_step = float(os.getenv("FT8R_MICRO_FULL_DF_STEP_HZ", "0.5"))
    dt_offs = np.arange(-dt_span, dt_span + 1e-9, dt_step) * (1.0 / TONE_SPACING_IN_HZ)  # symbols → seconds
    df_offs = np.arange(-df_span, df_span + 1e-9, df_step)
    for dt_seed, fq_seed in seeds:
        try:
            bb0 = downsample_to_baseband(audio, fq_seed)
            dt_ref = fine_time_sync(bb0, dt_seed, search=10)
        except Exception:
            dt_ref = dt_seed
        for ddt in dt_offs:
            for ddf in df_offs:
                ok, _mu = try_seed(audio, dt_ref + float(ddt), fq_seed + float(ddf), counters, expected_msg)
                if ok:
                    return True
    return False


def main() -> int:
    # Build failing set under whitening + budgeted selection (best current single-pass)
    from demod import decode_full_period

    wavs = sorted(DATA_DIR.rglob("*.wav"))
    failing: List[Tuple[str, str, float, float]] = []
    total_expected = 0
    default_ok = 0

    os.environ['FT8R_WHITEN_ENABLE'] = '1'

    for wav in wavs:
        exp = parse_expected(wav.with_suffix('.txt'))
        if not exp:
            continue
        audio = read_wav(str(wav))
        base = decode_full_period(audio)
        for msg, dt, freq in exp:
            total_expected += 1
            found = has_record(base, msg, dt, freq)
            if found:
                default_ok += 1
            else:
                failing.append((str(wav.relative_to(DATA_DIR)), msg, dt, freq))

    # Evaluate light microsearch
    c_light = Counters()
    t0 = time.perf_counter()
    rec_light = 0
    for stem, msg, dt, freq in failing:
        audio = read_wav(str(DATA_DIR / stem))
        if light_microsearch(audio, msg, dt, freq, c_light):
            rec_light += 1
    t_light = time.perf_counter() - t0

    # Evaluate full microsearch
    c_full = Counters()
    t0 = time.perf_counter()
    rec_full = 0
    for stem, msg, dt, freq in failing:
        audio = read_wav(str(DATA_DIR / stem))
        if full_microsearch(audio, msg, dt, freq, c_full):
            rec_full += 1
    t_full = time.perf_counter() - t0

    out = {
        'total_expected': total_expected,
        'default_ok': default_ok,
        'default_failed': len(failing),
        'light_recovered': rec_light,
        'light_fine_sync_calls': c_light.fine_sync,
        'light_ldpc_calls': c_light.ldpc_calls,
        'light_runtime_sec': round(t_light, 1),
        'full_recovered': rec_full,
        'full_fine_sync_calls': c_full.fine_sync,
        'full_ldpc_calls': c_full.ldpc_calls,
        'full_runtime_sec': round(t_full, 1),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
