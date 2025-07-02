import numpy as np
from typing import List, Tuple

from utils import (
    RealSamples,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
    FT8_SYMBOL_LENGTH_IN_SEC,
    FT8_SYMBOLS_PER_MESSAGE,
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


def naive_demod(samples_in: RealSamples, freq: float, dt: float) -> Tuple[str, List[int]]:
    """Very naive FT8 demodulator.

    Parameters
    ----------
    samples_in:
        Input audio samples with sample rate.
    freq:
        Base frequency in Hz returned by :func:`search.find_candidates`.
    dt:
        Start time offset from :func:`search.find_candidates`.

    Returns
    -------
    Tuple[str, List[int]]
        Bit string for the message payload and list of tone indices.
    """
    samples = samples_in.samples
    sample_rate = samples_in.sample_rate_in_hz
    sym_len = int(round(sample_rate * FT8_SYMBOL_LENGTH_IN_SEC))
    # ``dt`` comes from ``find_candidates`` which subtracts
    # ``COSTAS_START_OFFSET_SEC`` from the true signal start time.
    # Add it back here and convert to a sample index.
    start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
    time_idx = np.arange(sym_len) / sample_rate
    bases = np.exp(
        -2j
        * np.pi
        * (freq + np.arange(8) * TONE_SPACING_IN_HZ)[:, None]
        * time_idx
    )

    tones: List[int] = []
    for sym in range(FT8_SYMBOLS_PER_MESSAGE):
        seg = samples[start + sym * sym_len : start + (sym + 1) * sym_len]
        assert len(seg) == sym_len, "ran out of samples"
        tones.append(int(np.argmax(np.abs(np.matmul(bases, seg)))))

    bits: List[str] = []
    for idx, tone in enumerate(tones):
        if idx in COSTAS_POSITIONS:
            continue
        val = GRAY_MAP[tone]
        bits.append(f"{val:03b}")
    bitstring = "".join(bits)
    return bitstring, tones
