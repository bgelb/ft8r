import numpy as np
from math import erf, pi, sqrt, log

# Inverse of demod.GRAY_MAP (tone -> Gray code). We need the inverse to map
# Gray-coded 3-bit payloads back to tone indices.
_GRAY_MAP = [0b000, 0b001, 0b011, 0b010, 0b110, 0b100, 0b101, 0b111]
_INV_GRAY = {code: tone for tone, code in enumerate(_GRAY_MAP)}


def tones_from_bits(bits174: str) -> list[int]:
    """Return 79 tone indices (0..7) for a 174-bit FT8 codeword.

    Positions 0–6, 36–42, 72–78 carry the Costas sync tones. The remaining 58
    positions are the payload symbols, each derived from a 3‑bit Gray‑coded
    chunk of the input bitstring.
    """
    if len(bits174) != 174:
        raise ValueError("bits174 must have length 174")
    # Local import to avoid circular dependency during utils package init
    from . import COSTAS_SEQUENCE, FT8_SYMBOLS_PER_MESSAGE
    costas_positions = list(range(7)) + list(range(36, 43)) + list(range(72, 79))
    tones: list[int] = [0] * FT8_SYMBOLS_PER_MESSAGE

    # Insert Costas sync tones
    for i, pos in enumerate(costas_positions):
        tones[pos] = COSTAS_SEQUENCE[i % 7]

    # Fill payload symbols from 3‑bit Gray code chunks
    payload_positions = [i for i in range(FT8_SYMBOLS_PER_MESSAGE) if i not in costas_positions]
    for k, pos in enumerate(payload_positions):
        payload_bits = bits174[3 * k : 3 * k + 3]
        tones[pos] = _INV_GRAY[int(payload_bits, 2)]
    return tones


def generate_ft8_waveform(
    bits174: str,
    sample_rate: int = 12000,
    base_freq_hz: float = 1500.0,
    *,
    start_offset_sec: float = None,
    total_duration_sec: float = 15.0,
    amplitude: float = 1.0,
    per_symbol_phase: list[float] | None = None,
):
    """Generate a WSJT‑X‑compliant FT8 waveform for one 15 s period.

    The implementation matches ft8sim (WSJT‑X) shaping exactly:
    Gaussian‑filtered frequency pulses (BT=2.0), dummy symbols at the edges for
    transition smoothing, and a contiguous 12.64 s transmission placed at
    +0.5 s into a 15 s frame.
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
    samples_per_symbol = int(round(sample_rate * FT8_SYMBOL_LENGTH_IN_SEC))
    tone_indices = tones_from_bits(bits174)

    # Build instantaneous phase increments over (NSYM+2)*NSPS samples
    num_symbols = FT8_SYMBOLS_PER_MESSAGE
    active_samples = num_symbols * samples_per_symbol
    # Gaussian frequency pulse (BT=2.0)
    time_idx = np.arange(1, 3 * samples_per_symbol + 1, dtype=float)
    t_norm = (time_idx - 1.5 * samples_per_symbol) / float(samples_per_symbol)
    const_c = pi * sqrt(2.0 / log(2.0))
    pulse = 0.5 * (
        np.array([erf(const_c * 2.0 * (u + 0.5)) - erf(const_c * 2.0 * (u - 0.5)) for u in t_norm])
    )
    dphi = np.zeros((num_symbols + 2) * samples_per_symbol, dtype=float)
    dphi_peak = (2.0 * pi) / float(samples_per_symbol)  # hmod=1.0
    tone_vals = np.asarray(tone_indices, dtype=float)
    for s in range(num_symbols):
        start = s * samples_per_symbol
        dphi[start : start + 3 * samples_per_symbol] += dphi_peak * pulse * tone_vals[s]
    # Dummy symbol smoothing at edges
    dphi[0 : 2 * samples_per_symbol] += dphi_peak * tone_vals[0] * pulse[samples_per_symbol : 3 * samples_per_symbol]
    tail = num_symbols * samples_per_symbol
    dphi[tail : tail + 2 * samples_per_symbol] += dphi_peak * tone_vals[-1] * pulse[0 : 2 * samples_per_symbol]
    # Carrier increment
    dphi += (2.0 * pi * base_freq_hz) / float(sample_rate)

    # Vectorized phase synthesis over the active span (exclude initial dummy)
    start = samples_per_symbol
    seg_dphi = dphi[start : start + active_samples]
    phase_before = np.cumsum(seg_dphi) - seg_dphi
    phase_before = np.remainder(phase_before, 2.0 * pi)
    # Apply optional per-symbol constant phase offsets after phase integration
    if per_symbol_phase is not None:
        if len(per_symbol_phase) != FT8_SYMBOLS_PER_MESSAGE:
            raise ValueError("per_symbol_phase must have length 79 (symbols)")
        phase_before = phase_before.copy()
        for i in range(FT8_SYMBOLS_PER_MESSAGE):
            if per_symbol_phase[i] == 0.0:
                continue
            i0 = i * samples_per_symbol
            i1 = i0 + samples_per_symbol
            phase_before[i0:i1] += float(per_symbol_phase[i])
    wave = np.sin(phase_before)

    # Apply gentle Hann ramps over first/last 1/8 symbol
    ramp = int(round(samples_per_symbol / 8.0))
    if ramp > 0:
        ramp_idx = np.arange(ramp)
        wave[:ramp] *= 0.5 * (1.0 - np.cos(2.0 * pi * ramp_idx / (2.0 * ramp)))
        wave[-ramp:] *= 0.5 * (1.0 + np.cos(2.0 * pi * ramp_idx / (2.0 * ramp)))

    # Place waveform at +0.5 s in a 15 s frame (no wrap)
    frame_len = int(total_duration_sec * sample_rate)
    sig = np.zeros(frame_len, dtype=float)
    start_idx = int(round(start_offset_sec * sample_rate))
    sig[start_idx : start_idx + active_samples] = amplitude * wave

    return RealSamples(sig, sample_rate_in_hz=sample_rate)
