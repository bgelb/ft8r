from search import find_candidates
from utils import (
    read_wav,
    RealSamples,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
    FT8_SYMBOL_LENGTH_IN_SEC,
    FT8_SYMBOLS_PER_MESSAGE,
)
import random

from tests.utils import generate_ft8_wav, default_search_params


def test_candidate_search(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    candidates = find_candidates(
        audio,
        max_freq_bin,
        max_dt_symbols,
        threshold=0.002,
    )
    assert candidates, "no candidates found"
    score, dt, freq = candidates[0]
    assert abs(freq - 1500) < 1.0
    assert abs(dt - 0.0) < 0.2
    assert score > 0.002
    for _, _, f in candidates:
        assert abs(f - 1500) <= 32


def test_candidate_search_noise():
    sample_rate_in_hz = 12000
    max_freq_bin, max_dt_symbols = default_search_params(sample_rate_in_hz)
    random.seed(123)
    total_len_sec = COSTAS_START_OFFSET_SEC + FT8_SYMBOLS_PER_MESSAGE * FT8_SYMBOL_LENGTH_IN_SEC
    num_samples = int(total_len_sec * sample_rate_in_hz)
    noise = [random.uniform(-1e-2, 1e-2) for _ in range(num_samples)]
    noise_audio = RealSamples(noise, sample_rate_in_hz)
    cands = find_candidates(noise_audio, max_freq_bin, max_dt_symbols, threshold=0.002)
    assert not cands
