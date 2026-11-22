"""Depth-four temporal interleaver for four existing 2560-bit codewords.

The permutation has zero transmitted-bit overhead. It sends bit 0 of each
codeword, then bit 1 of each, and so on. The receiver applies the inverse
permutation to Manchester soft scores before decoding each codeword.
"""

from __future__ import annotations

import numpy as np

from .fec import CODEWORD_BITS

BLOCKS = 4
STREAM_BITS = BLOCKS * CODEWORD_BITS
COLUMN_ORDER = np.random.default_rng(20261121).permutation(CODEWORD_BITS)
INVERSE_COLUMN_ORDER = np.argsort(COLUMN_ORDER)


def interleave(values: np.ndarray, *, scramble: bool = False) -> np.ndarray:
    """Map four concatenated codewords into time order."""
    array = np.asarray(values)
    if array.shape != (STREAM_BITS,):
        raise ValueError(f"expected {STREAM_BITS} codeword values")
    codewords = array.reshape(BLOCKS, CODEWORD_BITS)
    if scramble:
        codewords = codewords[:, COLUMN_ORDER]
    return codewords.T.reshape(STREAM_BITS).copy()


def deinterleave(values: np.ndarray, *, scramble: bool = False) -> np.ndarray:
    """Map time-ordered soft scores back to four concatenated codewords."""
    array = np.asarray(values)
    if array.shape != (STREAM_BITS,):
        raise ValueError(f"expected {STREAM_BITS} received values")
    codewords = array.reshape(CODEWORD_BITS, BLOCKS).T
    if scramble:
        codewords = codewords[:, INVERSE_COLUMN_ORDER]
    return codewords.reshape(STREAM_BITS).copy()


if __name__ == "__main__":
    original = np.arange(STREAM_BITS)
    assert np.array_equal(deinterleave(interleave(original)), original)
    assert np.array_equal(deinterleave(interleave(original, scramble=True), scramble=True), original)
    print("interleaver inverse verified", STREAM_BITS)
