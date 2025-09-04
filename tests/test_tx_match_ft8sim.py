import numpy as np
import pytest

from utils import read_wav, generate_ft8_waveform
from tests.utils import generate_ft8_wav, ft8code_bits, resolve_wsjt_binary


@pytest.mark.skipif(resolve_wsjt_binary('ft8sim') is None or resolve_wsjt_binary('ft8code') is None, reason='requires ft8sim and ft8code')
def test_tx_matches_ft8sim(tmp_path):
    msg = "CQ K1ABC FN31"
    bits = ft8code_bits(msg)
    # Use high SNR so ft8sim emits a pure, noise-free waveform
    wav_path = generate_ft8_wav(msg, tmp_path, freq=1500, snr=90, dt=0.0, fdop=0.0)
    ref = read_wav(str(wav_path))

    ours = generate_ft8_waveform(bits, sample_rate=ref.sample_rate_in_hz, base_freq_hz=1500.0)
    n = min(len(ref.samples), len(ours.samples))
    a = ref.samples[:n].astype(float)
    b = ours.samples[:n].astype(float)

    # Align by best circular shift (FFT cross-correlation)
    size = 1 << ((2 * n - 1).bit_length())
    A = np.fft.rfft(a, size)
    B = np.fft.rfft(b, size)
    xc = np.fft.irfft(A * np.conj(B), size)
    xc = np.concatenate([xc[-(n - 1):], xc[:n]])
    lag = int(np.argmax(xc)) - (n - 1)
    if lag >= 0:
        a_al = a[lag:]
        b_al = b[:len(a_al)]
    else:
        b_al = b[-lag:]
        a_al = a[:len(b_al)]

    denom = float(np.dot(b_al, b_al)) + 1e-20
    g = float(np.dot(a_al, b_al) / denom)
    resid = a_al - g * b_al
    max_err = float(np.max(np.abs(resid)))
    # Our generator should match ft8sim sample-for-sample after gain fit
    assert max_err < 5e-3
