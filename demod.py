import numpy as np
from typing import List

import ldpc

from utils import (
    RealSamples,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
    FT8_SYMBOL_LENGTH_IN_SEC,
    FT8_SYMBOLS_PER_MESSAGE,
    LDPC_174_91_H,
)

# Symbol positions occupied by the three 7-symbol Costas sequences.
COSTAS_POSITIONS = list(range(7)) + list(range(36, 43)) + list(range(72, 79))


# Map tone indices to 3-bit values using Gray coding.
GRAY_MAP = [
    0b000,  # tone 0
    0b001,  # tone 1
    0b011,  # tone 2
    0b010,  # tone 3
    0b110,  # tone 4
    0b100,  # tone 5
    0b101,  # tone 6
    0b111,  # tone 7
]

# LDPC parity matrix copied from the WSJT-X source tree
# (see ``utils/ldpc_matrix.py``).  Embedding it here avoids relying on the
# WSJT-X archive at runtime.

_LDPC_DECODER = ldpc.BpDecoder(
    LDPC_174_91_H, error_rate=0.1, input_vector_type="received_vector"
)


def soft_demod(samples_in: RealSamples, freq: float, dt: float) -> np.ndarray:
    """Return log-likelihood ratios for each payload bit.

    Parameters
    ----------
    samples_in:
        Input audio samples with sample rate.
    freq:
        Base frequency in Hz returned by :func:`search.find_candidates`.
    dt:
        Start time offset from :func:`search.find_candidates`.
    """

    samples = samples_in.samples
    sample_rate = samples_in.sample_rate_in_hz
    sym_len = int(round(sample_rate * FT8_SYMBOL_LENGTH_IN_SEC))
    start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
    time_idx = np.arange(sym_len) / sample_rate
    bases = np.exp(
        -2j * np.pi * (freq + np.arange(8) * TONE_SPACING_IN_HZ)[:, None] * time_idx
    )

    # Arrange the input samples into one matrix containing every symbol.  Each
    # row corresponds to one symbol worth of data.  This allows the tone
    # responses for all symbols to be computed in a single matrix
    # multiplication.
    seg = samples[start : start + sym_len * FT8_SYMBOLS_PER_MESSAGE]
    seg = seg.reshape(FT8_SYMBOLS_PER_MESSAGE, sym_len)

    # ``resp`` has shape ``(8, FT8_SYMBOLS_PER_MESSAGE)`` and contains the
    # magnitude response of each tone for every symbol.
    resp = np.abs(bases @ seg.T)

    # Remove the Costas symbols used for synchronization.
    payload_resp = np.delete(resp, COSTAS_POSITIONS, axis=1)

    # Normalize to per-symbol probabilities.
    probs = payload_resp / payload_resp.sum(axis=0, keepdims=True)

    # Pre-build Gray-code bit masks to compute log-likelihood ratios with
    # broadcasting. ``gray_bits`` has shape ``(3, 8)`` where each row selects the
    # tones contributing a ``1`` for that bit position.
    gray_bits = np.array(
        [[(g >> (2 - b)) & 1 for g in GRAY_MAP] for b in range(3)], dtype=bool
    )

    mask = gray_bits[:, :, None]
    ones = np.where(mask, probs[None, :, :], 0).sum(axis=1)
    zeros = np.where(~mask, probs[None, :, :], 0).sum(axis=1)
    llrs = np.log(ones + 1e-12) - np.log(zeros + 1e-12)

    return llrs.T.ravel()


def naive_hard_decode(llrs: np.ndarray) -> str:
    """Return a hard-decision bitstring from log-likelihood ratios."""

    bits = ["1" if llr > 0 else "0" for llr in llrs]
    return "".join(bits)


def ldpc_decode(llrs: np.ndarray) -> str:
    """LDPC decode using soft information ``llrs``.

    Parameters
    ----------
    llrs:
        Log-likelihood ratios for each of the 174 coded bits where positive
        values favour ``1`` and negative values favour ``0``.
    """

    hard = (llrs > 0).astype(np.uint8)
    decoded = _LDPC_DECODER.decode(hard)
    bits = "".join("1" if b else "0" for b in decoded.astype(int))
    return bits
