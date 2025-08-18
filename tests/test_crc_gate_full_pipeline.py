import pytest

from demod import decode_full_period
from utils import read_wav
from tests.utils import (
    generate_ft8_wav,
    default_search_params,
    DEFAULT_SEARCH_THRESHOLD,
    resolve_wsjt_binary,
)


def test_decode_full_period_crc_gated(tmp_path):
    # Requires ft8sim to generate a known-good WAV
    if resolve_wsjt_binary("ft8sim") is None:
        pytest.skip("ft8sim not available; skipping full-pipeline CRC gate test")

    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path, snr=-10)
    audio = read_wav(str(wav))
    results = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD)
    # Should decode at least one message and it must match exactly
    assert any(r["message"] == msg for r in results)
    # And no result should slip through with CRC failure (implicit in gating)

