import numpy as np

from utils import RealSamples
from utils.waveform import bits_to_tones, synth_waveform
from search import candidate_score_map
from tests.utils import default_search_params


def test_score_reduction_after_subtraction():
    sample_rate = 12000
    total_len = int(15 * sample_rate)
    bits = "0" * 174
    tones = bits_to_tones(bits)
    wave = synth_waveform(tones, sample_rate, 1500, 0.0, amplitude=1.0, total_len=total_len)

    audio = RealSamples(wave.copy(), sample_rate)
    max_freq_bin, max_dt_symbols = default_search_params(sample_rate)
    scores, dts, freqs = candidate_score_map(audio, max_freq_bin, max_dt_symbols)
    dt_idx = int(np.argmin(np.abs(dts - 0.0)))
    freq_idx = int(np.argmin(np.abs(freqs - 1500)))
    before = scores[dt_idx, freq_idx]

    residual = RealSamples(audio.samples - wave, sample_rate)
    scores_after, _, _ = candidate_score_map(residual, max_freq_bin, max_dt_symbols)
    after = scores_after[dt_idx, freq_idx]
    assert after < before * 0.1
