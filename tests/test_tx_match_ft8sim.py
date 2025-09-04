import numpy as np
import pytest

from utils import read_wav, generate_ft8_waveform
from tests.utils import generate_ft8_wav, ft8code_bits, resolve_wsjt_binary


@pytest.mark.skipif(resolve_wsjt_binary('ft8sim') is None or resolve_wsjt_binary('ft8code') is None, reason='requires ft8sim and ft8code')
@pytest.mark.xfail(reason='Pulse shaping not yet exactly matched to WSJT-X', strict=False)
def test_tx_matches_ft8sim(tmp_path):
    msg = "CQ K1ABC FN31"
    bits = ft8code_bits(msg)
    wav_path = generate_ft8_wav(msg, tmp_path, freq=1500, snr=60, dt=0.0, fdop=0.0)
    ref = read_wav(str(wav_path))

    ours = generate_ft8_waveform(bits, sample_rate=ref.sample_rate_in_hz, base_freq_hz=1500.0)
    n = min(len(ref.samples), len(ours.samples))
    a = ref.samples[:n].astype(float)
    b = ours.samples[:n].astype(float)

    denom = float(np.dot(b, b)) + 1e-20
    g = float(np.dot(a, b) / denom)
    resid = a - g * b
    max_err = float(np.max(np.abs(resid)))
    assert max_err < 1e-3

