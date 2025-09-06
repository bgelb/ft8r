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

    # Exact WSJT-X ft8sim shaping: Gaussian-filtered frequency pulses (BT=2.0)
    active_len = n_sym_total * sym_len
    bt = 2.0
    # Build Gaussian-filtered frequency pulse over 3 symbols (1-based indices in reference code)
    t_idx = np.arange(1, 3 * sym_len + 1, dtype=float)
    tt = (t_idx - 1.5 * sym_len) / float(sym_len)
    c = np.pi * np.sqrt(2.0 / np.log(2.0))
    # gfsk_pulse(b,t) = 0.5 * (erf(c*b*(t+0.5)) - erf(c*b*(t-0.5)))
    from math import erf
    pulse = 0.5 * (np.array([erf(c * bt * (u + 0.5)) - erf(c * bt * (u - 0.5)) for u in tt]))
    # dphi array length (nsym+2)*nsps to allow edge extension
    dphi = np.zeros((n_sym_total + 2) * sym_len, dtype=float)
    dphi_peak = two_pi * 1.0 / float(sym_len)  # hmod=1.0
    tones_arr = np.asarray(tones, dtype=float)
    for j in range(n_sym_total):
        ib = j * sym_len
        dphi[ib : ib + 3 * sym_len] += dphi_peak * pulse * tones_arr[j]
    # Dummy symbols at beginning and end
    dphi[0 : 2 * sym_len] += dphi_peak * tones_arr[0] * pulse[sym_len : 3 * sym_len]
    dphi[n_sym_total * sym_len : (n_sym_total + 2) * sym_len] += dphi_peak * tones_arr[-1] * pulse[0 : 2 * sym_len]
    # Add carrier contribution
    dphi += two_pi * base_freq_hz / sample_rate
    # Generate nwave samples from indices [nsps, nsps + active_len)
    nwave = active_len
    start = sym_len
    end = start + nwave
    seg_dphi = dphi[start:end]
    # Phase before increment at each sample equals cumulative sum excluding current
    phi_before = np.cumsum(seg_dphi) - seg_dphi
    # Reduce modulo 2π to limit growth (stability); sin is invariant to 2π wraps
    phi_before = np.remainder(phi_before, two_pi)
    wave = np.sin(phi_before)
    # Apply Hann ramps to first and last 1/8 symbol
    nramp = int(round(sym_len / 8.0))
    if nramp > 0:
        n = np.arange(nramp)
        wave[:nramp] *= 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (2.0 * nramp)))
        wave[-nramp:] *= 0.5 * (1.0 + np.cos(2.0 * np.pi * n / (2.0 * nramp)))
    # Place into 15 s buffer then circularly shift by -nint((0.5 + xdt)/dt)
    full_len = int(total_duration_sec * sample_rate)
    buf = np.zeros(full_len, dtype=float)
    buf[:nwave] = amplitude * wave
    shift = -int(round((start_offset_sec) * sample_rate))
    sig = np.roll(buf, shift)

    return RealSamples(sig, sample_rate_in_hz=sample_rate)
