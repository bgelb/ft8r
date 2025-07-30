import numpy as np

from . import (
    COSTAS_SEQUENCE,
    COSTAS_START_OFFSET_SEC,
    FT8_SYMBOL_LENGTH_IN_SEC,
    FT8_SYMBOLS_PER_MESSAGE,
    TONE_SPACING_IN_HZ,
)

# Gray-code map used for FT8 tone assignments
_GRAY_MAP = [
    0b000,
    0b001,
    0b011,
    0b010,
    0b110,
    0b100,
    0b101,
    0b111,
]

# Pre-computed mapping from symbol index to Costas tone if the position is part
# of one of the three 7-symbol sync sequences.
_COSTAS_TONES = {i: t for i, t in zip(
    list(range(7)) + list(range(36, 43)) + list(range(72, 79)),
    COSTAS_SEQUENCE * 3,
)}


def bits_to_tones(bitstring: str) -> np.ndarray:
    """Return the FT8 tone sequence encoded by ``bitstring``.

    ``bitstring`` must contain the 174 FEC bits of an FT8 codeword.
    """
    if len(bitstring) != 174:
        raise ValueError("bitstring must contain 174 bits")

    tones = np.zeros(FT8_SYMBOLS_PER_MESSAGE, dtype=int)
    bit_idx = 0
    for sym in range(FT8_SYMBOLS_PER_MESSAGE):
        if sym in _COSTAS_TONES:
            tones[sym] = _COSTAS_TONES[sym]
        else:
            bits = bitstring[bit_idx : bit_idx + 3]
            bit_idx += 3
            val = int(bits, 2)
            tones[sym] = _GRAY_MAP[val]
    if bit_idx != len(bitstring):
        raise ValueError("bitstring length mismatch")
    return tones


def synth_waveform(
    tones: np.ndarray,
    sample_rate: int,
    base_freq: float,
    start_sec: float,
    amplitude: float = 1.0,
    total_len: int | None = None,
) -> np.ndarray:
    """Return a real-valued waveform for ``tones`` starting at ``start_sec``."""
    sym_len = int(round(sample_rate * FT8_SYMBOL_LENGTH_IN_SEC))
    if total_len is None:
        total_len = int(round(15.0 * sample_rate))
    wave = np.zeros(total_len, dtype=float)

    t = np.arange(sym_len) / sample_rate
    base_start = int(round((start_sec + COSTAS_START_OFFSET_SEC) * sample_rate))
    for i, tone in enumerate(tones):
        freq = base_freq + tone * TONE_SPACING_IN_HZ
        segment = amplitude * np.cos(2 * np.pi * freq * t)
        start = base_start + i * sym_len
        end = start + sym_len
        if start >= total_len:
            break
        seg_end = min(end, total_len)
        wave[start:seg_end] += segment[: seg_end - start]
    return wave
