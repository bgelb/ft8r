"""WSJT-X FT8 LDPC (174,91) column order mapping.

This array maps indices in the decoder/parity-matrix column order to the
transmitted bit order. It is copied from WSJT-X 2.7.0
lib/ft8/ldpc_174_91_c_colorder.f90.

Given a 174-bit vector ``v_tx`` in transmitted order, the corresponding vector
in the decoder (H) column order is ``v_h[i] = v_tx[COLORDER_174[i]]``.
"""

COLORDER_174 = [
  0,  1,  2,  3, 28,  4,  5,  6,  7,  8,  9, 10, 11, 34, 12, 32, 13, 14, 15, 16,
 17, 18, 36, 29, 43, 19, 20, 42, 21, 40, 30, 37, 22, 47, 61, 45, 44, 23, 41, 39,
 49, 24, 46, 50, 48, 26, 31, 33, 51, 38, 52, 59, 55, 66, 57, 27, 60, 35, 54, 58,
 25, 56, 62, 64, 67, 69, 63, 68, 70, 72, 65, 73, 75, 74, 71, 77, 78, 76, 79, 80,
 53, 81, 83, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,
120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,
140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
160,161,162,163,164,165,166,167,168,169,170,171,172,173,
]

INV_COLORDER_174 = [0]*174
for i, p in enumerate(COLORDER_174):
    INV_COLORDER_174[p] = i

