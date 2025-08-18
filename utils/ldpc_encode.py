import numpy as np
from .ldpc_matrix import LDPC_174_91_H


def _gf2_inv(mat: np.ndarray) -> np.ndarray:
    """Return inverse of a square matrix over GF(2)."""
    n = mat.shape[0]
    a = (mat.copy() % 2).astype(np.uint8)
    inv = np.eye(n, dtype=np.uint8)
    r = 0
    for c in range(n):
        # find pivot
        piv = None
        for i in range(r, n):
            if a[i, c]:
                piv = i
                break
        if piv is None:
            raise ValueError("matrix is singular over GF(2)")
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
            inv[[r, piv]] = inv[[piv, r]]
        # eliminate other rows
        for i in range(n):
            if i != r and a[i, c]:
                a[i, :] ^= a[r, :]
                inv[i, :] ^= inv[r, :]
        r += 1
        if r == n:
            break
    return inv


# Partition H into [H_info | H_par]
_H = (LDPC_174_91_H % 2).astype(np.uint8)
_H_info = _H[:, :91]
_H_par = _H[:, 91:]
if _H_par.shape[0] != _H_par.shape[1]:
    # Expect square parity submatrix (83x83)
    raise RuntimeError("Unexpected LDPC parity submatrix shape")
_H_par_inv = _gf2_inv(_H_par)


def compute_parity_bits(info_crc_bits_91: np.ndarray | str) -> np.ndarray:
    """Compute the 83 parity bits given the first 91 bits (77 msg + 14 CRC).

    Returns a uint8 numpy array of length 83 in transmitted bit order.
    """
    if isinstance(info_crc_bits_91, str):
        if len(info_crc_bits_91) != 91:
            raise ValueError("info_crc_bits_91 must be 91 bits long")
        info_vec = np.fromiter((1 if c == '1' else 0 for c in info_crc_bits_91), dtype=np.uint8)
    else:
        info_vec = np.asarray(info_crc_bits_91, dtype=np.uint8)
        if info_vec.shape[0] != 91:
            raise ValueError("info_crc_bits_91 must be length 91")
    # Solve H_par * p = H_info * i  (mod 2)
    rhs = (_H_info @ info_vec) % 2
    p = (_H_par_inv @ rhs) % 2
    return p.astype(np.uint8)

