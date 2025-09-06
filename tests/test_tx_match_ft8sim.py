import numpy as np
import pytest

from utils import read_wav, generate_ft8_waveform, generate_ft8_waveform
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

@pytest.mark.skipif(resolve_wsjt_binary('ft8sim') is None or resolve_wsjt_binary('ft8code') is None, reason='requires ft8sim and ft8code')
def test_tx_bt_mismatch_fails(tmp_path):
    # A slight change in BT should cause a measurable mismatch vs ft8sim
    msg = "CQ K1ABC FN31"
    bits = ft8code_bits(msg)
    wav_path = generate_ft8_wav(msg, tmp_path, freq=1500, snr=90, dt=0.0, fdop=0.0)
    ref = read_wav(str(wav_path))

    sr = ref.sample_rate_in_hz
    a = ref.samples.astype(float)

    # Generate our compliant waveform
    ours = generate_ft8_waveform(bits, sample_rate=sr, base_freq_hz=1500.0)
    b = ours.samples.astype(float)

    # Generate a waveform with slightly perturbed BT (2.01 instead of 2.0)
    from utils.tx import tones_from_bits
    def gen_with_bt(bt: float):
        from utils import FT8_SYMBOL_LENGTH_IN_SEC, COSTAS_START_OFFSET_SEC
        import numpy as np
        from math import erf
        sym_len = int(round(sr * FT8_SYMBOL_LENGTH_IN_SEC))
        tones = tones_from_bits(bits)
        two_pi = 2.0 * np.pi
        n_sym_total = len(tones)
        active_len = n_sym_total * sym_len
        t_idx = np.arange(1, 3 * sym_len + 1, dtype=float)
        tt = (t_idx - 1.5 * sym_len) / float(sym_len)
        c = np.pi * np.sqrt(2.0 / np.log(2.0))
        pulse = 0.5 * (np.array([erf(c * bt * (u + 0.5)) - erf(c * bt * (u - 0.5)) for u in tt]))
        dphi = np.zeros((n_sym_total + 2) * sym_len, dtype=float)
        dphi_peak = two_pi * 1.0 / float(sym_len)
        tones_arr = np.asarray(tones, dtype=float)
        for j in range(n_sym_total):
            ib = j * sym_len
            dphi[ib : ib + 3 * sym_len] += dphi_peak * pulse * tones_arr[j]
        dphi[0 : 2 * sym_len] += dphi_peak * tones_arr[0] * pulse[sym_len : 3 * sym_len]
        dphi[n_sym_total * sym_len : (n_sym_total + 2) * sym_len] += dphi_peak * tones_arr[-1] * pulse[0 : 2 * sym_len]
        dphi += two_pi * 1500.0 / sr
        seg = dphi[sym_len : sym_len + active_len]
        phi_before = np.cumsum(seg) - seg
        phi_before = np.remainder(phi_before, two_pi)
        w = np.sin(phi_before)
        nramp = int(round(sym_len / 8.0))
        if nramp > 0:
            n = np.arange(nramp)
            w[:nramp] *= 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (2.0 * nramp)))
            w[-nramp:] *= 0.5 * (1.0 + np.cos(2.0 * np.pi * n / (2.0 * nramp)))
        full = np.zeros_like(a)
        full[:active_len] = w
        shift = -int(round(COSTAS_START_OFFSET_SEC * sr))
        return np.roll(full, shift)

    b_bad = gen_with_bt(2.01)

    # Compare helper: align and fit gain, return max abs error
    def max_err(x, y):
        import numpy as np
        n = min(len(x), len(y))
        x = x[:n].astype(float)
        y = y[:n].astype(float)
        size = 1 << ((2 * n - 1).bit_length())
        X = np.fft.rfft(x, size)
        Y = np.fft.rfft(y, size)
        xc = np.fft.irfft(X * np.conj(Y), size)
        xc = np.concatenate([xc[-(n - 1):], xc[:n]])
        lag = int(np.argmax(xc)) - (n - 1)
        if lag >= 0:
            xa = x[lag:]
            ya = y[:len(xa)]
        else:
            ya = y[-lag:]
            xa = x[:len(ya)]
        denom = float(np.dot(ya, ya)) + 1e-20
        g = float(np.dot(xa, ya) / denom)
        return float(np.max(np.abs(xa - g * ya)))

    # Compliant path should pass with tight threshold
    assert max_err(a, b) < 5e-3
    # Perturbed BT should fail by a noticeable margin
    assert max_err(a, b_bad) > 5e-3
