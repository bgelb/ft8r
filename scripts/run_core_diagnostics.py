#!/usr/bin/env python3
"""
Run the core fine-sync diagnostics over all cases that failed default decode
but succeeded with brute-force seeding around known (dt,freq).

Inputs: .tmp/ft8r_seed_recovered_core.json (from scripts/experiment_seed_recovered_core.py)
Outputs:
  - Per-case JSON under .tmp/fine_sync_diag_core/
  - Summary TSV at .tmp/fine_sync_core_summary.tsv
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import (
    read_wav,
    COSTAS_START_OFFSET_SEC,
    TONE_SPACING_IN_HZ,
    FT8_SYMBOL_LENGTH_IN_SEC,
    COSTAS_SEQUENCE,
)
from search import find_candidates
from demod import (
    downsample_to_baseband,
    fine_time_sync,
    soft_demod,
    fine_sync_candidate,
    naive_hard_decode,
    ldpc_decode,
)
from utils import check_crc


DATA_DIR = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"
REPORT = Path(".tmp/ft8r_seed_recovered_core.json")
OUT_DIR = Path(".tmp/fine_sync_diag_core")
SUMMARY_TSV = Path(".tmp/fine_sync_core_summary.tsv")

DT_EPS = 0.2
FREQ_EPS = 1.0


def _symbol_len(sample_rate: float) -> int:
    return int(round(sample_rate * FT8_SYMBOL_LENGTH_IN_SEC))


def _costas_positions() -> np.ndarray:
    # 3x 7-symbol Costas pilots at symbols 0..6, 36..42, 72..78
    return np.array([*range(7), *range(36, 43), *range(72, 79)], dtype=int)


def _freq_curve(bb_samples, dt_ref: float, df_span: float = 5.0, df_step: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Return (df_grid, energy) around dt_ref using Costas energy."""
    sample_rate = bb_samples.sample_rate_in_hz
    sym_len = _symbol_len(sample_rate)
    start = int(round((dt_ref + COSTAS_START_OFFSET_SEC) * sample_rate))
    seg = bb_samples.samples[start : start + sym_len * 79].reshape(79, sym_len)

    time_idx = np.arange(sym_len) / sample_rate
    bases0 = np.exp(-2j * np.pi * (np.arange(8) * TONE_SPACING_IN_HZ)[:, None] * time_idx[None, :])
    df = np.arange(-df_span, df_span + df_step / 2.0, df_step)
    shifts = np.exp(-2j * np.pi * df[:, None] * time_idx[None, :])  # (F, S)
    bases = bases0[None, :, :] * shifts[:, None, :]                 # (F, 8, S)
    resp = np.abs(bases @ seg.T) ** 2                               # (F, 8, 79)
    tones = np.array(COSTAS_SEQUENCE * 3)
    pos = _costas_positions()
    energy = resp[:, tones, :][:, :, pos].sum(axis=(1, 2))
    return df, energy.astype(float)


def _parabolic_refine(y: np.ndarray, x: np.ndarray, idx: int) -> float:
    if 0 < idx < len(y) - 1:
        y1, y2, y3 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-18:
            frac = 0.5 * (y1 - y3) / denom
            frac = float(np.clip(frac, -0.5, 0.5))
            return float(x[idx] + frac * (x[1] - x[0]))
    return float(x[idx])


def _fwhm(x: np.ndarray, y: np.ndarray) -> float:
    y_max = float(y.max())
    if y_max <= 0:
        return float("nan")
    half = 0.5 * y_max
    mask = y >= half
    if not mask.any():
        return float("nan")
    idx = np.where(mask)[0]
    return float(x[idx[-1]] - x[idx[0]])


def _crc_and_llr(audio, dt: float, freq: float) -> Tuple[bool, float]:
    try:
        bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
        llrs = soft_demod(bb)
        mu = float(np.mean(np.abs(llrs)))
        hard = naive_hard_decode(llrs)
        if check_crc(hard):
            return True, mu
        soft = ldpc_decode(llrs)
        return check_crc(soft), mu
    except Exception:
        return False, float("nan")


def diag_one(stem: str, msg: str, dt_t: float, fq_t: float) -> Dict:
    wav = DATA_DIR / stem
    audio = read_wav(str(wav))

    # Coarse candidates
    sample_rate = audio.sample_rate_in_hz
    sym_len = int(sample_rate / TONE_SPACING_IN_HZ)
    max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
    max_dt_samples = len(audio.samples) - int(sample_rate * COSTAS_START_OFFSET_SEC)
    max_dt_symbols = -(-max_dt_samples // sym_len)
    candidates = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=1.0)

    near_rank = None
    near_score = None
    near = None
    for i, (score, dt, freq) in enumerate(candidates):
        if abs(dt - dt_t) < DT_EPS and abs(freq - fq_t) < FREQ_EPS:
            near_rank = i + 1
            near_score = float(score)
            near = (dt, freq)
            break

    neighbor_count = 0
    top_neighbor_score = 0.0
    for i, (score, dt, freq) in enumerate(candidates):
        if abs(dt - dt_t) <= 0.3 and abs(freq - fq_t) <= 25.0:
            if near is not None and abs(dt - near[0]) < 1e-9 and abs(freq - near[1]) < 1e-9:
                continue
            neighbor_count += 1
            if score > top_neighbor_score:
                top_neighbor_score = float(score)

    # Frequency sweep at refined dt (time-only refine)
    coarse_freq = round(fq_t / TONE_SPACING_IN_HZ) * TONE_SPACING_IN_HZ
    bb0 = downsample_to_baseband(audio, coarse_freq)
    dt_ref = fine_time_sync(bb0, dt_t, search=10)
    df_grid, energy = _freq_curve(bb0, dt_ref, df_span=5.0, df_step=0.25)
    best = int(np.argmax(energy))
    df_peak = _parabolic_refine(energy, df_grid, best)
    snr_f = float((energy.max() - np.median(energy)) / (np.std(energy) + 1e-12))
    fwhm_f = _fwhm(df_grid, energy)
    off_grid_half = 2.5 <= abs(df_peak) <= 3.75

    # CRC and LLR margins
    crc_exp, mu_exp = _crc_and_llr(audio, dt_t, fq_t)
    crc_peak, mu_peak = _crc_and_llr(audio, dt_t, fq_t + df_peak)

    # Labels
    labels = []
    if near_rank is None:
        labels.append("CoarseMiss")
    if snr_f < 3.0 or (not math.isnan(fwhm_f) and fwhm_f > 0.8):
        labels.append("AmbiguousFreq")
    if off_grid_half:
        labels.append("OffGridHalfTone")
    if neighbor_count >= 1:
        base = near_score if near_score is not None else top_neighbor_score
        if base > 0 and top_neighbor_score >= 0.8 * base:
            labels.append("InterferenceSuspected")
    if (not crc_exp) and crc_peak and (not math.isnan(mu_exp)) and (not math.isnan(mu_peak)) and (mu_exp < 0.25 <= mu_peak):
        labels.append("DecoderMargin")

    # Minimal JSON
    return {
        "stem": stem,
        "expected": {"message": msg, "dt": dt_t, "freq": fq_t},
        "coarse": {
            "near_rank": near_rank,
            "near_score": near_score,
            "neighbor_count_pm25Hz": neighbor_count,
            "top_neighbor_score": top_neighbor_score,
        },
        "freq_sweep": {
            "df_peak": df_peak,
            "snr_f": snr_f,
            "fwhm_f": fwhm_f,
            "off_grid_half_tone": off_grid_half,
            "df_grid": df_grid.tolist(),
            "energy": energy.tolist(),
        },
        "decode": {
            "crc_expected": crc_exp,
            "mu_llr_expected": mu_exp,
            "crc_df_peak": crc_peak,
            "mu_llr_df_peak": mu_peak,
        },
        "labels": labels,
    }


def sanitize_name(stem: str, dt: float, freq: float) -> str:
    s = stem.replace("/", "__").replace(".wav", "")
    return f"{s}__{int(round(dt*1000)):d}ms__{int(round(freq))}Hz.json"


def main() -> int:
    if not REPORT.exists():
        raise SystemExit("Missing .tmp/ft8r_seed_recovered_core.json. Run experiment_seed_recovered_core.py first.")
    data = json.loads(REPORT.read_text())
    details = data.get("details", [])
    if not details:
        print("No seed-recovered cases to diagnose.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []
    header = [
        "stem","dt","freq","near_rank","near_score","neighbor_count",
        "df_peak","snr_f","fwhm_f","crc_expected","crc_df_peak","mu_llr_expected","mu_llr_df_peak","labels"
    ]
    rows.append("\t".join(header))

    for d in details:
        stem = d["stem"]
        exp = d["expected"]
        msg = exp["message"]
        dt_t = float(exp["dt"])
        fq_t = float(exp["freq"])

        r = diag_one(stem, msg, dt_t, fq_t)
        out_name = sanitize_name(stem, dt_t, fq_t)
        (OUT_DIR / out_name).write_text(json.dumps(r, indent=2))

        near_rank = r["coarse"]["near_rank"]
        near_score = r["coarse"]["near_score"]
        neighbor_count = r["coarse"]["neighbor_count_pm25Hz"]
        df_peak = r["freq_sweep"]["df_peak"]
        snr_f = r["freq_sweep"]["snr_f"]
        fwhm_f = r["freq_sweep"]["fwhm_f"]
        crc_expected = r["decode"]["crc_expected"]
        crc_df_peak = r["decode"]["crc_df_peak"]
        mu_exp = r["decode"]["mu_llr_expected"]
        mu_peak = r["decode"]["mu_llr_df_peak"]
        labels = ",".join(r["labels"]) if r["labels"] else ""
        rows.append("\t".join([
            stem,
            f"{dt_t:.3f}", f"{fq_t:.2f}",
            str(near_rank) if near_rank is not None else "",
            f"{near_score:.3f}" if near_score is not None else "",
            str(neighbor_count),
            f"{df_peak:+.2f}", f"{snr_f:.2f}", f"{fwhm_f:.2f}" if not math.isnan(fwhm_f) else "nan",
            "1" if crc_expected else "0",
            "1" if crc_df_peak else "0",
            f"{mu_exp:.3f}" if not math.isnan(mu_exp) else "nan",
            f"{mu_peak:.3f}" if not math.isnan(mu_peak) else "nan",
            labels,
        ]))

    SUMMARY_TSV.write_text("\n".join(rows) + "\n")
    print(f"Wrote per-case JSON: {OUT_DIR}")
    print(f"Wrote summary TSV: {SUMMARY_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

