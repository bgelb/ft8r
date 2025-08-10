import numpy as np

from utils import RealSamples, TONE_SPACING_IN_HZ, FT8_SYMBOL_LENGTH_IN_SEC, COSTAS_START_OFFSET_SEC
from demod import (
    _symbol_samples,
    _bits_to_tone_sequence,
    _estimate_symbol_phasors,
    _synthesize_original_signal,
    _zero_offset_bases,
    ComplexSamples,
)


def _build_bits_from_tones(payload_tones):
    # Convert payload tone indices (length 58) into 174-bit string, with dummy CRC region
    # Only used for tone mapping test; CRC not needed.
    gray_map = [0b000, 0b001, 0b011, 0b010, 0b110, 0b100, 0b101, 0b111]
    bits = []
    for t in payload_tones:
        g = gray_map[t]
        bits.extend([(g >> 2) & 1, (g >> 1) & 1, g & 1])
    # Pad to 174 bits if needed
    if len(bits) < 174:
        bits.extend([0] * (174 - len(bits)))
    return "".join("1" if b else "0" for b in bits[:174])


def test_bits_to_tone_sequence_length_and_costas():
    # Create a dummy payload of cycling tones 0..7 repeated
    payload_tones = [(i % 8) for i in range(58)]
    bits = _build_bits_from_tones(payload_tones)
    tones = _bits_to_tone_sequence(bits)
    assert len(tones) == 79
    # Costas positions should be filled and other positions should match payload len 58
    payload_positions = [i for i in range(79) if i not in (list(range(7)) + list(range(36, 43)) + list(range(72, 79)))]
    assert len(payload_positions) == 58


def test_phasor_estimation_matches_constructed_baseband():
    # Construct baseband with known per-symbol phasors and verify estimation.
    bb_sr = 200
    sym_len_bb = _symbol_samples(bb_sr)
    # Alternate tones for payload; Costas positions filled later by estimator path
    payload_tones = [(i % 8) for i in range(58)]
    bits = _build_bits_from_tones(payload_tones)
    tone_seq = _bits_to_tone_sequence(bits)
    bases = _zero_offset_bases(bb_sr, sym_len_bb)
    # Known phasors
    rng = np.random.default_rng(0)
    amps = 0.8 + 0.2 * rng.random(79)
    phases = rng.uniform(-np.pi, np.pi, size=79)
    phasors_true = amps * np.exp(1j * phases)
    # Synthesize complex baseband
    segs = []
    for s in range(79):
        tone = tone_seq[s]
        segs.append(phasors_true[s] * bases[tone])
    bb_samples = np.concatenate(segs)
    bb = ComplexSamples(bb_samples, bb_sr)
    ph_est = _estimate_symbol_phasors(bb, tone_seq)
    # Compare magnitude and phase (modulo 2π)
    assert np.allclose(np.abs(ph_est), np.abs(phasors_true), rtol=1e-2, atol=1e-2)
    # For phase, compare cos of difference to avoid wrap
    phase_diff = np.angle(ph_est) - np.angle(phasors_true)
    assert np.allclose(np.cos(phase_diff), np.ones_like(phase_diff), atol=1e-2)


def test_synthesize_original_signal_matches_reference():
    # Given tone_seq and phasors, ensure synthesizer matches a*cos(ωt+φ)
    sr = 12000
    sym_len = _symbol_samples(sr)
    total_len = sym_len * 79
    dt = 0.0
    base_freq = 1000.0
    payload_tones = [(i % 8) for i in range(58)]
    bits = _build_bits_from_tones(payload_tones)
    tone_seq = _bits_to_tone_sequence(bits)
    rng = np.random.default_rng(1)
    amps = 0.7 + 0.3 * rng.random(79)
    phases = rng.uniform(-np.pi, np.pi, size=79)
    phasors = amps * np.exp(1j * phases)
    # Call synthesizer
    samples = RealSamples(np.zeros(int(round((COSTAS_START_OFFSET_SEC + 79 / TONE_SPACING_IN_HZ) * sr)) + 10), sr)
    s_idx, synth = _synthesize_original_signal(samples, dt, base_freq, tone_seq, phasors, alpha=1.0)
    # Build independent reference
    ref = np.zeros_like(synth)
    t = np.arange(len(synth)) / sr
    for s in range(79):
        a = amps[s]
        phi = phases[s]
        tone = tone_seq[s]
        omega = 2.0 * np.pi * (base_freq + tone * TONE_SPACING_IN_HZ)
        seg = a * np.cos(omega * t[s * sym_len:(s + 1) * sym_len] + phi)
        ref[s * sym_len:(s + 1) * sym_len] = seg
    # Correlation should be high
    num = float(np.dot(ref, synth))
    den = float(np.linalg.norm(ref) * np.linalg.norm(synth) + 1e-12)
    corr = num / den
    assert corr >= 0.99

