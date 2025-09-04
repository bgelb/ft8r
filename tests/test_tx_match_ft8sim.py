import numpy as np
import pytest

from utils import read_wav, generate_ft8_waveform
from tests.utils import generate_ft8_wav, ft8code_bits, resolve_wsjt_binary


@pytest.mark.skipif(resolve_wsjt_binary('ft8sim') is None or resolve_wsjt_binary('ft8code') is None, reason='requires ft8sim and ft8code')
@pytest.mark.xfail(reason='Pulse shaping not yet matched to WSJT-X; placeholder until tuned', strict=False)
def test_tx_matches_ft8sim(tmp_path):
    msg = "CQ K1ABC FN31"
    bits = ft8code_bits(msg)
    # Generate ft8sim reference WAV at 1500 Hz, dt=0, no Doppler, high SNR (effectively no noise)
    wav_path = generate_ft8_wav(msg, tmp_path, freq=1500, snr=60, dt=0.0, fdop=0.0)
    ref = read_wav(str(wav_path))

    # Our synthesized waveform (same sample rate and base freq)
    ours = generate_ft8_waveform(bits, sample_rate=ref.sample_rate_in_hz, base_freq_hz=1500.0, amplitude=1.0)

    # Align lengths (wsjtx typically outputs exact 15 s)
    n = min(len(ref.samples), len(ours.samples))
    a = ref.samples[:n].astype(float)
    b = ours.samples[:n].astype(float)

    # Fit gain to account for amplitude scaling differences between implementations
    # gain = argmin ||a - g*b|| = (a·b) / (b·b)
    denom = float(np.dot(b, b)) + 1e-20
    g = float(np.dot(a, b) / denom)
    resid = a - g * b
    max_err = float(np.max(np.abs(resid)))

    # Expect near-identical shaping modulo a scalar gain
    assert max_err < 1e-3, f"max sample error {max_err} too large"
