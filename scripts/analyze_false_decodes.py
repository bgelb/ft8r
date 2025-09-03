#!/usr/bin/env python3
"""Analyze and bucket false decodes against golden labels.

For each WAV (with adjacent WSJT-X .txt truth), runs the decoder and classifies
each decode record by:
- CRC status (pass/fail)
- All-zero bitstring vs not
- Near a golden transmission (dt/df tolerance) vs far
- Hamming distance to expected bits from ft8code (if available)
- Costas gate matches at (dt,freq)

Emits a summary to stdout and optionally a JSONL file with per-record details.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from utils import read_wav, RealSamples, check_crc, FT8_SYMBOL_LENGTH_IN_SEC, COSTAS_START_OFFSET_SEC


def _iter_wavs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.wav")):
        yield p


@dataclass
class Golden:
    dt: float
    df: float
    snr_db: float | None
    msg: str


def _parse_wsjt_txt_line(line: str) -> Optional[Golden]:
    line = line.strip()
    if not line:
        return None
    try:
        # Expected like: "000000  -5  1.1  809 ~  SQ5FBI G3NDC IO91"
        if "~" in line:
            left, msg = line.split("~", 1)
            msg = msg.strip()
            lparts = left.split()
        else:
            parts = line.split(maxsplit=5)
            msg = parts[5].strip() if len(parts) >= 6 else ""
            lparts = parts[:5]
        snr_db = float(lparts[1]) if len(lparts) > 1 else None
        dt = float(lparts[2]) if len(lparts) > 2 else 0.0
        df = float(lparts[3]) if len(lparts) > 3 else 0.0
        return Golden(dt=dt, df=df, snr_db=snr_db, msg=msg)
    except Exception:
        return None


def _read_golden(txt_path: Path) -> list[Golden]:
    rows: list[Golden] = []
    if not txt_path.exists():
        return rows
    for line in txt_path.read_text().splitlines():
        g = _parse_wsjt_txt_line(line)
        if g is not None:
            rows.append(g)
    return rows


def _normalize_msg(msg: str) -> str:
    """Normalize WSJT-X message text by removing appended metadata and squashing spaces.

    Many .txt lines include extra info after the message, e.g., "CQ TA6CQ KN70      AS Turkey".
    We split at runs of 2+ spaces and keep the left part, then collapse internal whitespace.
    """
    import re
    s = (msg or "").strip()
    # Remove everything after the first run of 2+ spaces
    s = re.split(r"\s{2,}", s)[0]
    # Collapse remaining whitespace to single spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _resolve_ft8code_bits() -> Optional[callable[[str], str]]:
    """Return a function message->174-bit using ft8code if available, else None."""
    try:
        from tests.utils import ft8code_bits  # type: ignore
        # Probe availability (raises if ft8code missing)
        _ = ft8code_bits("K1ABC FN31")
        # Return a thin wrapper to avoid re-import costs
        return ft8code_bits
    except Exception:
        return None


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _costas_gate_matches(audio: RealSamples, dt: float, freq_hz: float) -> Optional[int]:
    """Compute number of Costas gate matches (0..21) at dt,freq. Returns None on error."""
    try:
        from demod import fine_sync_candidate, TONE_SPACING_IN_HZ, FT8_SYMBOLS_PER_MESSAGE
        bb, dt_ref, freq_ref = fine_sync_candidate(audio, freq_hz, dt)
        sr = bb.sample_rate_in_hz
        sym_len = int(round(sr * FT8_SYMBOL_LENGTH_IN_SEC))
        seg = bb.samples[: sym_len * FT8_SYMBOLS_PER_MESSAGE].reshape(FT8_SYMBOLS_PER_MESSAGE, sym_len)
        time_idx = np.arange(sym_len) / sr
        bases = np.exp(-2j * np.pi * (np.arange(8)[:, None] * TONE_SPACING_IN_HZ) * time_idx[None, :])
        resp = np.abs(bases @ seg.T) ** 2  # (8, 79)
        costas_pos = list(range(7)) + list(range(36, 43)) + list(range(72, 79))
        from utils import COSTAS_SEQUENCE
        tones = COSTAS_SEQUENCE * 3
        max_idx = np.argmax(resp[:, costas_pos], axis=0)
        matches = int((max_idx == np.array(tones)).sum())
        return matches
    except Exception:
        return None


def analyze_wav(
    wav_path: Path,
    threshold: float,
    dt_tol: float,
    df_tol: float,
    get_bits_fn: Optional[callable[[str], str]],
) -> list[dict]:
    from demod import decode_full_period
    audio = read_wav(str(wav_path))
    gold = _read_golden(wav_path.with_suffix(".txt"))

    # Build golden helpers
    golden_msgs = {_normalize_msg(g.msg) for g in gold}
    golden_tbl = [(g.dt, g.df, _normalize_msg(g.msg)) for g in gold]
    golden_bits: dict[str, str] = {}
    if get_bits_fn is not None:
        for g in gold:
            try:
                nm = _normalize_msg(g.msg)
                if nm:
                    golden_bits[nm] = get_bits_fn(nm)
            except Exception:
                continue

    recs = decode_full_period(audio, threshold=threshold, include_bits=True)

    out: list[dict] = []
    for r in recs:
        bits: str = r.get("bits", "")
        crc_ok = (len(bits) == 174 and check_crc(bits))
        all_zero = (bits == ("0" * 174))
        msg = _normalize_msg(r.get("message", ""))
        dt = float(r.get("dt", 0.0))
        freq = float(r.get("freq", 0.0))
        # proximity to nearest golden
        near = False
        near_msg: Optional[str] = None
        near_dt = None
        near_df = None
        for gdt, gdf, gmsg in golden_tbl:
            if abs(dt - gdt) <= dt_tol and abs(freq - gdf) <= df_tol:
                near = True
                near_msg = gmsg
                near_dt = gdt
                near_df = gdf
                break

        # Expected bits if available
        ham174 = None
        ham91 = None
        ham77 = None
        exp_bits = None
        if near and near_msg and bits and len(bits) == 174:
            exp_bits = golden_bits.get(near_msg)
            if exp_bits and len(exp_bits) == 174:
                ham174 = _hamming(bits, exp_bits)
                ham91 = _hamming(bits[:91], exp_bits[:91])
                ham77 = _hamming(bits[:77], exp_bits[:77])

        # Costas gate matches
        gate = _costas_gate_matches(audio, dt, freq)

        # Classification
        if crc_ok:
            in_golden = (msg in golden_msgs)
            if in_golden and near:
                bucket = "true_positive"
            else:
                bucket = "crc_pass_not_in_golden"
        else:
            bucket = ("crc_fail_near" if near else "crc_fail_far")
            if all_zero:
                bucket += ":all_zero"

        out.append(
            {
                "wav": wav_path.name,
                "bucket": bucket,
                "message": msg,
                "crc_ok": crc_ok,
                "all_zero": all_zero,
                "dt": dt,
                "freq": freq,
                "score": float(r.get("score", 0.0)),
                "llr": float(r.get("llr", 0.0)),
                "method": r.get("method", ""),
                "near": near,
                "near_msg": near_msg,
                "near_dt": near_dt,
                "near_df": near_df,
                "gate_matches": gate,
                "ham174": ham174,
                "ham91": ham91,
                "ham77": ham77,
            }
        )
    return out


def summarize(rows: list[dict]) -> dict:
    from collections import Counter
    c = Counter(r["bucket"] for r in rows)
    # Breakdown of crc_fail buckets by all_zero and near
    cz = sum(1 for r in rows if str(r["bucket"]).startswith("crc_fail") and r.get("all_zero"))
    cn = sum(1 for r in rows if str(r["bucket"]).startswith("crc_fail_near"))
    cf = sum(1 for r in rows if str(r["bucket"]).startswith("crc_fail_far"))
    # Phantom decodes: crc_pass_not_in_golden
    ph = c.get("crc_pass_not_in_golden", 0)
    return {
        "total": len(rows),
        "counts": dict(c),
        "crc_fail_all_zero": cz,
        "crc_fail_near": cn,
        "crc_fail_far": cf,
        "phantom_crc_pass": ph,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze and bucket false decodes vs golden")
    ap.add_argument("wav_root", nargs="?", default="ft8_lib-2.0/test/wav", help="Directory with WAVs and WSJT-X .txt truth files")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--dt-tol", type=float, default=0.08, help="Seconds tolerance for near-golden")
    ap.add_argument("--df-tol", type=float, default=3.125, help="Hz tolerance for near-golden")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of WAVs")
    ap.add_argument("--out", type=str, default="", help="Optional JSONL output path with per-record details")
    args = ap.parse_args(argv)

    wav_root = Path(args.wav_root)
    # Ensure demod includes CRC-fail records
    import demod as _demod
    try:
        _demod._ALLOW_CRC_FAIL = True  # type: ignore[attr-defined]
    except Exception:
        pass

    get_bits_fn = _resolve_ft8code_bits()
    if get_bits_fn is None:
        print("Note: ft8code not available; expected-bit distances disabled")

    rows_all: list[dict] = []
    count_wav = 0
    for wav in _iter_wavs(wav_root):
        rows = analyze_wav(wav, args.threshold, args.dt_tol, args.df_tol, get_bits_fn)
        rows_all.extend(rows)
        count_wav += 1
        if args.limit and count_wav >= args.limit:
            break

    summ = summarize(rows_all)
    print(json.dumps(summ, indent=2))
    if args.out:
        outp = Path(args.out)
        with outp.open("w") as f:
            for r in rows_all:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows_all)} records to {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
