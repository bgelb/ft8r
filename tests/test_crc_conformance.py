import pytest

from utils import check_crc
from tests.utils import ft8code_bits, resolve_wsjt_binary


@pytest.mark.parametrize(
    "message",
    [
        "K1ABC W9XYZ EN37",
        "CQ DM1YS JO30",
        "CQ OE3UKW JN88",
        "CQ CU2DX HM77",
        "EA8TH F8DBF R-04",
        "4X5MZ RA6FSD 73",
        "IK2YCW OE3UKW JN88",
        "CQ UY5AX KO70",
    ],
)
def test_crc_conformance_against_ft8code(message):
    if resolve_wsjt_binary("ft8code") is None:
        pytest.skip("ft8code not available; skipping CRC conformance test")
    bits = ft8code_bits(message)
    assert check_crc(bits)

