import numpy as np
import pytest

from utils import RealSamples
from utils.tx import generate_ft8_waveform
from demod import decode_full_period, decode_full_period_multipass
from tests.utils import ft8code_bits, resolve_wsjt_binary


def _mix(sig_a: RealSamples, sig_b: RealSamples, noise_std: float = 0.0) -> RealSamples:
    assert sig_a.sample_rate_in_hz == sig_b.sample_rate_in_hz
    sr = sig_a.sample_rate_in_hz
    n = min(len(sig_a.samples), len(sig_b.samples))
    out = sig_a.samples[:n] + sig_b.samples[:n]
    if noise_std > 0:
        rng = np.random.default_rng(0)
        out = out + rng.normal(scale=noise_std, size=n)
    return RealSamples(out, sr)


@pytest.mark.skipif(resolve_wsjt_binary('ft8code') is None, reason='requires ft8code')
def test_sic_enables_hidden_signal():
    # Two messages at different frequencies; one much weaker
    msg_strong = "CQ K1ABC FN31"
    msg_weak = "K1ABC W9XYZ EN37"
    bits_s = ft8code_bits(msg_strong)
    bits_w = ft8code_bits(msg_weak)
    sr = 12000
    # Place both at default start (dt=0) with different carriers
    wav_s = generate_ft8_waveform(bits_s, sample_rate=sr, base_freq_hz=1400.0, amplitude=1.0)
    wav_w = generate_ft8_waveform(bits_w, sample_rate=sr, base_freq_hz=1850.0, amplitude=0.22)
    mix = _mix(wav_s, wav_w, noise_std=0.01)

    # Single-pass likely decodes only the strong signal in this mixture
    r1 = decode_full_period(mix, include_bits=False)
    texts1 = {r['message'] for r in r1}
    assert msg_strong in texts1
    # Either weak is missing or, if present, ensure the test remains meaningful
    if msg_weak in texts1:
        pytest.skip("weak signal already decodable without SIC; adjust amplitudes if this flakes")

    # Multi-pass with 2 passes should reveal the weak signal after subtraction
    out = decode_full_period_multipass(
        mix, passes=2, sic_scale=0.7, return_pass_records=True, return_residuals=True
    )
    per_pass = out['per_pass']
    residuals = out.get('residuals') or []
    texts2 = {r['message'] for r in per_pass[0]}
    texts3 = texts2 | {r['message'] for r in per_pass[1]}
    assert msg_strong in texts2
    assert msg_weak in texts3, "SIC pass 2 failed to recover the weak signal"
    # Residual after pass1 should drop the strong while exposing the weak
    assert residuals, "No residual snapshot captured"
    red = decode_full_period(residuals[0], include_bits=False)
    red_texts = {r['message'] for r in red}
    assert msg_strong not in red_texts
    assert msg_weak in red_texts


@pytest.mark.skipif(resolve_wsjt_binary('ft8code') is None, reason='requires ft8code')
def test_sic_removes_strong_signal_from_residual():
    # Single strong signal: after subtraction, it should not re-decode
    msg = "CQ K1ABC FN31"
    bits = ft8code_bits(msg)
    sr = 12000
    base_freq = 1500.0
    wav = generate_ft8_waveform(bits, sample_rate=sr, base_freq_hz=base_freq, amplitude=1.0)
    # Add light noise
    rng = np.random.default_rng(1)
    noisy = RealSamples(wav.samples + rng.normal(scale=0.01, size=len(wav.samples)), sr)

    # Sanity: original decodes
    base = decode_full_period(noisy, include_bits=False)
    assert any(r['message'] == msg for r in base)

    # Subtract after pass1 and re-decode residual
    out = decode_full_period_multipass(
        noisy, passes=2, sic_scale=0.7, sic_use_phase=True,
        return_pass_records=True, return_residuals=True
    )
    residuals = out.get('residuals') or []
    assert residuals, "No residual snapshot captured after pass1 subtraction"
    red = decode_full_period(residuals[0], include_bits=False)
    assert all(r['message'] != msg for r in red), "Strong signal still present after subtraction"
