from demod import decode_full_period
from utils import read_wav, RealSamples
from tests.utils import (
    generate_ft8_wav,
    DEFAULT_SEARCH_THRESHOLD,
    DEFAULT_FREQ_EPS,
    DEFAULT_DT_EPS,
)


def test_full_decode_single(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path, snr=0)
    audio = read_wav(str(wav))
    results = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD)
    assert any(r["message"] == msg for r in results)
    rec = next(r for r in results if r["message"] == msg)
    assert abs(rec["freq"] - 1500) < DEFAULT_FREQ_EPS
    assert abs(rec["dt"] - 0.0) < DEFAULT_DT_EPS


def test_full_decode_multiple(tmp_path):
    msg1 = "K1ABC W9XYZ EN37"
    msg2 = "K9XYZ W1ABC EN37"

    # Generate two independent FT8 signals using the same working directory.
    wav1 = generate_ft8_wav(msg1, tmp_path, freq=1500, snr=10)
    a1 = read_wav(str(wav1))

    # ``ft8sim`` always writes the same output filename so the second call
    # overwrites the first file.  That's fine because we already loaded ``a1``.
    wav2 = generate_ft8_wav(msg2, tmp_path, freq=1600, snr=10)
    a2 = read_wav(str(wav2))

    mixed = RealSamples(a1.samples + a2.samples, a1.sample_rate_in_hz)
    results = decode_full_period(mixed, threshold=0.5)

    assert isinstance(results, list)

    rec1 = next(r for r in results if r["message"] == msg1)
    rec2 = next(r for r in results if r["message"] == msg2)

    assert abs(rec1["freq"] - 1500) < DEFAULT_FREQ_EPS
    assert abs(rec2["freq"] - 1600) < DEFAULT_FREQ_EPS

    assert abs(rec1["dt"] - 0.0) < DEFAULT_DT_EPS
    assert abs(rec2["dt"] - 0.0) < DEFAULT_DT_EPS

