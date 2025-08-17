import numpy as np

from demod import ldpc_decode, naive_hard_decode
from utils import check_crc, LDPC_174_91_H
from tests.utils import ft8code_bits


def _llrs_from_bits(bits: str, mag: float = 5.0) -> np.ndarray:
    assert len(bits) == 174
    return np.array([mag if b == "1" else -mag for b in bits], dtype=float)


def test_ldpc_crc_on_known_codeword():
    msg = "K1ABC W9XYZ EN37"
    expected = ft8code_bits(msg)
    # Construct strong LLRs consistent with expected codeword
    llrs = _llrs_from_bits(expected, mag=5.0)
    # Step 2: CRC on the codeword produced by ft8code must be valid
    assert check_crc(expected)

    # Step 3: Push perfect LLRs through LDPC and verify CRC/order
    decoded = ldpc_decode(llrs)
    # Decoder returns the same 174-bit codeword
    assert decoded == expected
    # CRC over the first 91 bits must be valid
    assert check_crc(decoded)
    # Parity checks must be satisfied
    hard = np.array([1 if b == "1" else 0 for b in decoded], dtype=np.uint8)
    syndrome = (LDPC_174_91_H @ hard) % 2
    assert int(syndrome.sum()) == 0


def test_ldpc_corrects_few_llr_errors():
    msg = "K1ABC W9XYZ EN37"
    expected = ft8code_bits(msg)

    # Start with strong LLRs for the true codeword
    llrs = _llrs_from_bits(expected, mag=3.0)

    # Introduce a few sign errors in info bits (indices < 91)
    flip_idx = [5, 40, 85]
    for i in flip_idx:
        llrs[i] *= -1.0

    # Hard decision should now fail CRC
    hard = naive_hard_decode(llrs)
    assert not check_crc(hard)

    # LDPC should correct and CRC must pass post-correction
    decoded = ldpc_decode(llrs)
    assert decoded == expected
    assert check_crc(decoded)
    # Parity satisfied
    hard_vec = np.array([1 if b == "1" else 0 for b in decoded], dtype=np.uint8)
    syndrome = (LDPC_174_91_H @ hard_vec) % 2
    assert int(syndrome.sum()) == 0
