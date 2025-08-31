#!/usr/bin/env python3
"""Experiment: retry LDPC on blank decodes with a small LLR bias.

Scans the bundled sample WAVs, finds decodes where message == "",
reconstructs the aligned baseband at the reported dt/freq, recomputes LLRs,
and retries LDPC with a tiny bias on the 77 message bits (and optionally the
mode bits) to see if any blanks recover to non-blank messages (still CRC-gated).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

from utils import read_wav, check_crc, decode77
from demod import fine_sync_candidate, soft_demod
import ldpc as _ldpc
from utils import LDPC_174_91_H as _H


DATA_DIR = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"


def _ldpc_decode_with_order(llrs: np.ndarray, order: int = 2) -> str:
    # Mirror demod.ldpc_decode but allow custom OSD order
    hard = (llrs > 0).astype(np.uint8)
    error_prob = 1.0 / (np.exp(7.0 * np.abs(llrs)) + 1.0)
    dec = _ldpc.BpOsdDecoder(
        _H,
        error_rate=0.1,
        input_vector_type="received_vector",
        osd_method="OSD_CS",
        osd_order=int(order),
    )
    dec.update_channel_probs(error_prob)
    syndrome = (_H @ hard) % 2
    syndrome = syndrome.astype(np.uint8)
    err_est = dec.decode(syndrome)
    corrected = np.bitwise_xor(err_est.astype(np.uint8), hard)
    bits = "".join("1" if b else "0" for b in corrected.astype(int))
    return bits


def retry_blank(audio, dt: float, freq: float, *, bias_msg: float = 0.1, bias_mode: float = 0.2) -> Tuple[bool, str | None]:
    """Return (recovered, new_text) by retrying LDPC with small bias.

    Strategy: add a tiny positive bias to the first 77 message LLRs to break
    the all-zero fixed point; add a slightly larger bias to the 6 mode bits
    (positions 71..76) to discourage the free-text path (i3=0,n3=0).
    """
    bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
    llrs = soft_demod(bb)
    if llrs.shape[0] != 174:
        # Unexpected; give up
        return False, None
    # Compute avg |LLR| to gate attempts in very weak conditions only
    llr_avg = float(np.mean(np.abs(llrs)))
    # Only bother if evidence is weak
    # If evidence is already strong, don't retry
    if llr_avg > 0.6:
        return False, None

    # Bias copy
    b = llrs.copy()
    # Bias all 77 payload bits slightly toward '1'
    b[:77] += bias_msg
    # Extra bias to mode bits (n3,i3): positions 71..76 in the 77-bit payload
    b[71:77] += bias_mode

    # Try increasing OSD order variants
    for order in (2, 3, 4):
        new_bits = _ldpc_decode_with_order(b, order=order)
        if not check_crc(new_bits):
            continue
        text = decode77(new_bits[:77])
        if text.strip() == "":
            continue
        return True, text
    return False, None


def main(limit_files: int | None = None, max_retries: int = 50):
    blanks = 0
    recovered = 0
    attempts = 0
    examples = []
    bias_schemes = [
        (0.10, 0.20),
        (0.15, 0.30),
        (0.00, 0.30),
        (0.20, 0.50),
        (-0.10, 0.20),  # mild negative on payload, positive on mode bits
        (0.30, 0.80),
        (0.50, 1.00),
    ]
    files = sorted(DATA_DIR.rglob("*.wav"))
    if limit_files:
        files = files[:limit_files]
    for wav in files:
        audio = read_wav(str(wav))
        from demod import decode_full_period
        decs = decode_full_period(audio)
        for r in decs:
            if r.get("message", "") == "":
                blanks += 1
                if attempts < max_retries:
                    # Try multiple bias schemes until one recovers or limit hit
                    for bm, bi in bias_schemes:
                        ok, txt = retry_blank(audio, r["dt"], r["freq"], bias_msg=bm, bias_mode=bi)
                        attempts += 1
                        if ok:
                            recovered += 1
                            examples.append((wav.name, r, txt))
                            break
                        if attempts >= max_retries:
                            break
        # Early stop if enough attempts
        if attempts >= max_retries:
            break

    print(f"blank_total={blanks} attempts={attempts} recovered={recovered}")
    for name, rec, txt in examples[:5]:
        print(f"example: {name} dt={rec['dt']:.3f}s freq={rec['freq']:.1f}Hz -> '{txt}'")


if __name__ == "__main__":
    # Allow tuning via env vars
    limit = int(os.getenv("FT8R_EXPT_LIMIT", "0") or 0) or None
    tries = int(os.getenv("FT8R_EXPT_TRIES", "50") or 50)
    main(limit_files=limit, max_retries=tries)
