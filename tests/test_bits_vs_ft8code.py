import pytest

from demod import decode_full_period
from utils import read_wav
from tests.utils import (
    generate_ft8_wav,
    DEFAULT_SEARCH_THRESHOLD,
    resolve_wsjt_binary,
    ft8code_bits,
)


@pytest.mark.parametrize(
    "message",
    [
        "K1ABC W9XYZ EN37",
        "EA8TH F8DBF R-04",
        "4X5MZ RA6FSD 73",
        "CQ OE3UKW JN88",
    ],
)
def test_decoded_payload_crc_matches_ft8code_91_bits(tmp_path, message):
    # Requires ft8sim/ft8code; skip if not present
    if resolve_wsjt_binary("ft8sim") is None or resolve_wsjt_binary("ft8code") is None:
        pytest.skip("WSJT-X tools not available")

    # Generate a clean WAV for the message and decode it
    wav = generate_ft8_wav(message, tmp_path, snr=0)
    audio = read_wav(str(wav))
    recs = decode_full_period(audio, threshold=DEFAULT_SEARCH_THRESHOLD, include_bits=True)
    # Find a matching text decode
    match = next((r for r in recs if r["message"] == message and "bits" in r), None)
    assert match is not None, f"Message '{message}' not decoded from synthetic WAV"

    got = match["bits"][:91]
    want = ft8code_bits(message)[:91]
    assert got == want

