import numpy as np
import pytest

from utils import LDPC_174_91_H
from tests.utils import resolve_wsjt_binary, ft8code_bits


def test_parity_on_ft8code_bits():
    # Require wsjtx ft8code for ground-truth bits
    if resolve_wsjt_binary("ft8code") is None:
        pytest.skip("ft8code not available; skipping order/parity test")

    msg = "K1ABC W9XYZ EN37"
    bits = ft8code_bits(msg)
    assert len(bits) == 174
    vec = np.array([1 if b == "1" else 0 for b in bits], dtype=np.uint8)
    syndrome = (LDPC_174_91_H @ vec) % 2
    assert int(syndrome.sum()) == 0, "H columns align with transmitted bit order"

