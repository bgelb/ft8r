import numpy as np
import pytest

from utils import LDPC_174_91_H
from utils.ldpc_encode import compute_parity_bits
from tests.utils import resolve_wsjt_binary, ft8code_bits


@pytest.mark.parametrize(
    "message",
    [
        "K1ABC W9XYZ EN37",
        "EA8TH F8DBF R-04",
        "4X5MZ RA6FSD 73",
        "CQ OE3UKW JN88",
    ],
)
def test_compute_parity_bits_matches_wsjtx(message):
    if resolve_wsjt_binary("ft8code") is None:
        pytest.skip("ft8code not available; skipping parity conformance test")
    bits = ft8code_bits(message)
    info = bits[:91]
    want_par = bits[91:]
    got_par = "".join("1" if b else "0" for b in compute_parity_bits(info))
    assert got_par == want_par


def test_compute_parity_bits_yields_zero_syndrome():
    # Random info+CRC vector of length 91; check that computed parity zeros the syndrome
    rng = np.random.default_rng(0)
    info_vec = rng.integers(0, 2, size=91, dtype=np.uint8)
    par = compute_parity_bits(info_vec)
    codeword = np.concatenate([info_vec, par]).astype(np.uint8)
    syn = (LDPC_174_91_H @ codeword) % 2
    assert int(syn.sum()) == 0

