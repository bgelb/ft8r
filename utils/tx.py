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
    amplitude: float = 1.0,
    ramp_fraction: float = 0.5,
    ramp_samples: int | None = None,
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

    # Output buffer
    sig = np.zeros(int(total_duration_sec * sample_rate), dtype=float)
    start_idx = int(round(start_offset_sec * sample_rate))
    n_sym_total = FT8_SYMBOLS_PER_MESSAGE
    two_pi = 2.0 * np.pi

    # Continuous-phase FSK with symmetric raised-cosine frequency transitions
    active_len = n_sym_total * sym_len
    # Ramp half-width in samples
    if ramp_samples is None:
        L = max(0, min(sym_len // 2, int(round(ramp_fraction * sym_len))))
    else:
        L = max(0, min(sym_len // 2, int(ramp_samples)))
    # Build instantaneous frequency array
    tone_freqs = [base_freq_hz + t * TONE_SPACING_IN_HZ for t in tones]
    f_inst = np.empty(active_len, dtype=float)
    # Start with step frequencies
    for i in range(n_sym_total):
        i0 = i * sym_len
        f_inst[i0 : i0 + sym_len] = tone_freqs[i]
    # Symmetric transition around each boundary
    if L > 0:
        for i in range(1, n_sym_total):
            f0 = tone_freqs[i - 1]
            f1 = tone_freqs[i]
            b = i * sym_len
            for j in range(-L, L):
                t = b + j
                if 0 <= t < active_len:
                    u = (j + L + 0.5) / (2 * L)
                    s = 0.5 * (1 - np.cos(np.pi * u))
                    f_inst[t] = f0 + (f1 - f0) * s
    # Integrate to phase and synthesize constant-envelope signal
    dphi = two_pi * f_inst / sample_rate
    phi = np.cumsum(dphi)
    tone = np.cos(phi)
    sig[start_idx : start_idx + active_len] = amplitude * tone

    return RealSamples(sig, sample_rate_in_hz=sample_rate)
