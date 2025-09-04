import numpy as np

# Inverse of demod.GRAY_MAP
_GRAY_MAP = [
    0b000,  # tone 0
    0b001,  # tone 1
    0b011,  # tone 2
    0b010,  # tone 3
    0b110,  # tone 4
    0b100,  # tone 5
    0b101,  # tone 6
    0b111,  # tone 7
]
_INV_GRAY = {code: tone for tone, code in enumerate(_GRAY_MAP)}


def tones_from_bits(bits174: str) -> list[int]:
    """Map a 174-bit FT8 codeword to 79 tone indices (0..7).

    - Costas sync tones at positions 0-6, 36-42, 72-78.
    - Remaining 58 positions are payload: map consecutive 3-bit groups via inverse Gray code.
    """
    if len(bits174) != 174:
        raise ValueError("bits174 must have length 174")
    # Local import to avoid circular dependency during utils package init
    from . import COSTAS_SEQUENCE, FT8_SYMBOLS_PER_MESSAGE
    costas_pos = list(range(7)) + list(range(36, 43)) + list(range(72, 79))
    tones = [0] * FT8_SYMBOLS_PER_MESSAGE
    for i, p in enumerate(costas_pos):
        tones[p] = COSTAS_SEQUENCE[i % 7]
    # Fill payload symbols
    payload_positions = [i for i in range(FT8_SYMBOLS_PER_MESSAGE) if i not in costas_pos]
    assert len(payload_positions) == 58
    for k, pos in enumerate(payload_positions):
        b3 = bits174[3 * k : 3 * k + 3]
        val = int(b3, 2)
        tone = _INV_GRAY.get(val)
        if tone is None:
            raise ValueError("invalid 3-bit Gray code")
        tones[pos] = tone
    return tones


def generate_ft8_waveform(
    bits174: str,
    sample_rate: int = 12000,
    base_freq_hz: float = 1500.0,
    *,
    start_offset_sec: float = None,
    total_duration_sec: float = 15.0,
    amplitude: float = 0.9,
):
    """Synthesize a mono FT8 audio period containing a single transmission.

    Generates 0.5 s of pre-gap by default, followed by 79 symbols of length
    1/TONE_SPACING_IN_HZ with continuous phase, then trailing silence to reach
    ``total_duration_sec``.
    """
    # Local import to avoid circulars
    from . import (
        COSTAS_START_OFFSET_SEC,
        FT8_SYMBOL_LENGTH_IN_SEC,
        FT8_SYMBOLS_PER_MESSAGE,
        TONE_SPACING_IN_HZ,
        RealSamples,
    )
    if start_offset_sec is None:
        start_offset_sec = COSTAS_START_OFFSET_SEC
    sym_len = int(round(sample_rate * FT8_SYMBOL_LENGTH_IN_SEC))
    tones = tones_from_bits(bits174)

    # Time indices per symbol
    sig = np.zeros(int(total_duration_sec * sample_rate), dtype=float)
    start_idx = int(round(start_offset_sec * sample_rate))
    n_sym_total = FT8_SYMBOLS_PER_MESSAGE
    phase = 0.0
    two_pi = 2.0 * np.pi

    for i in range(n_sym_total):
        f = base_freq_hz + tones[i] * TONE_SPACING_IN_HZ
        n0 = start_idx + i * sym_len
        n1 = n0 + sym_len
        t = (np.arange(sym_len) / sample_rate)
        # continuous-phase tone for this symbol starting at accumulated phase
        phi = phase + two_pi * f * t
        sig[n0:n1] = amplitude * np.cos(phi)
        # update phase at boundary for continuity
        phase = (phase + two_pi * f * (sym_len / sample_rate)) % (2 * np.pi)

    return RealSamples(sig, sample_rate_in_hz=sample_rate)
