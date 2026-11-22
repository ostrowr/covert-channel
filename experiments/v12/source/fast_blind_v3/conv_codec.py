"""Rate-1/2, K=7 convolutional FEC with terminated Viterbi decoding.

Bit convention: for each input bit, form ``reg = ((state << 1) | bit) & 0x7f``.
Transmit parity(reg & 0o171), then parity(reg & 0o133); retain the low six
bits of reg as the next state. The encoder adds six zero input bits, so both
the initial and final trellis states are zero.

``decode`` takes one LLR per transmitted output bit, in that same order.
Positive LLRs favor zero. The ML branch metric assumes independent output-bit
observations; scaling all LLRs by a positive constant does not change the
decision. This fixed code has no puncturing, interleaving, or error detection.
Decoding uses O(64 * (message_bits + 6)) time and traceback memory.
"""

from __future__ import annotations

from itertools import chain, repeat
from time import perf_counter

import numpy as np


CONSTRAINT_LENGTH = 7
TAIL_BITS = CONSTRAINT_LENGTH - 1
GENERATORS = (0o171, 0o133)
_N_STATES = 1 << TAIL_BITS


def _parity(value: int) -> int:
    return value.bit_count() & 1


# For each destination state q, the input bit is q & 1, and its two possible
# predecessor states differ only in the oldest state bit.
_DESTINATIONS = np.arange(_N_STATES, dtype=np.intp)
_PRED0 = _DESTINATIONS >> 1
_PRED1 = _PRED0 + (_N_STATES >> 1)


def _output_signs(predecessors: np.ndarray) -> np.ndarray:
    signs = np.empty((_N_STATES, 2), dtype=np.int8)
    for q, predecessor in enumerate(predecessors):
        reg = ((int(predecessor) << 1) | (q & 1)) & 0x7f
        signs[q, 0] = 1 - 2 * _parity(reg & GENERATORS[0])
        signs[q, 1] = 1 - 2 * _parity(reg & GENERATORS[1])
    return signs


_SIGN0 = _output_signs(_PRED0)
_SIGN1 = _output_signs(_PRED1)


def encode(bits: np.ndarray | list[int]) -> np.ndarray:
    """Encode binary input bits and append the six zero termination bits.

    Returns a one-dimensional ``uint8`` array of length ``2 * (len(bits)+6)``.
    An empty message therefore produces twelve zero output bits.
    """
    data = np.asarray(bits)
    if data.ndim != 1 or data.dtype.kind not in "biuf" or np.any((data != 0) & (data != 1)):
        raise ValueError("bits must be a one-dimensional array of zeros and ones")

    output = np.empty(2 * (len(data) + TAIL_BITS), dtype=np.uint8)
    state = 0
    for i, bit in enumerate(chain(data, repeat(0, TAIL_BITS))):
        reg = ((state << 1) | int(bit)) & 0x7f
        output[2 * i] = _parity(reg & GENERATORS[0])
        output[2 * i + 1] = _parity(reg & GENERATORS[1])
        state = reg & 0x3f
    return output


def decode(llrs: np.ndarray | list[float]) -> np.ndarray:
    """Decode terminated codeword LLRs and omit the six tail input bits.

    ``llrs`` must contain at least twelve finite values and have even length.
    Each value is log(P(observation | output=0) / P(observation | output=1));
    positive values therefore favor output bit zero. The initial and final
    trellis states are both fixed at zero.
    """
    observations = np.asarray(llrs, dtype=np.float64)
    if (
        observations.ndim != 1
        or len(observations) < 2 * TAIL_BITS
        or len(observations) & 1
        or not np.all(np.isfinite(observations))
    ):
        raise ValueError("llrs must be a one-dimensional, finite, even-length array of at least 12 values")

    n_steps = len(observations) // 2
    survivors = np.empty((n_steps, _N_STATES), dtype=np.bool_)
    scores = np.full(_N_STATES, -np.inf, dtype=np.float64)
    scores[0] = 0.0

    for t in range(n_steps):
        llr0 = observations[2 * t]
        llr1 = observations[2 * t + 1]
        candidate0 = scores[_PRED0] + _SIGN0[:, 0] * llr0 + _SIGN0[:, 1] * llr1
        candidate1 = scores[_PRED1] + _SIGN1[:, 0] * llr0 + _SIGN1[:, 1] * llr1
        choose1 = candidate1 > candidate0
        survivors[t] = choose1
        scores = np.maximum(candidate0, candidate1)
        scores -= scores.max()  # Prevent score growth on long codewords.

    decoded = np.empty(n_steps, dtype=np.uint8)
    state = 0  # Known terminal state from the six zero input bits.
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = state & 1
        state = int(_PRED1[state] if survivors[t, state] else _PRED0[state])
    return decoded[:-TAIL_BITS]


def selftest() -> dict[str, float | int]:
    """Check round trips and a seeded noisy channel, then time both paths."""
    rng = np.random.default_rng(0xFEC171133)
    for length in (0, 1, 7, 64, 257):
        message = rng.integers(0, 2, size=length, dtype=np.uint8)
        codeword = encode(message)
        recovered = decode((1 - 2 * codeword.astype(np.int16)) * 8.0)
        assert np.array_equal(recovered, message), f"noiseless round trip failed at {length} bits"

    channel_flips = 0
    residual_errors = 0
    for _ in range(10):
        message = rng.integers(0, 2, size=256, dtype=np.uint8)
        codeword = encode(message)
        flips = rng.random(len(codeword)) < 0.04
        channel_flips += int(flips.sum())
        hard_observations = codeword ^ flips.astype(np.uint8)
        llrs = (1 - 2 * hard_observations.astype(np.int16)) * 2.5
        recovered = decode(llrs)
        residual_errors += int(np.count_nonzero(recovered != message))
    assert residual_errors == 0, f"seeded noisy-channel test had {residual_errors} residual bit errors"

    message = rng.integers(0, 2, size=4096, dtype=np.uint8)
    codeword = encode(message)
    llrs = (1 - 2 * codeword.astype(np.int16)) * 4.0
    trials = 3
    start = perf_counter()
    for _ in range(trials):
        encode(message)
    encode_s = (perf_counter() - start) / trials
    start = perf_counter()
    for _ in range(trials):
        decode(llrs)
    decode_s = (perf_counter() - start) / trials
    return {
        "noisy_channel_flips": channel_flips,
        "noisy_residual_bit_errors": residual_errors,
        "encode_input_bits_per_second": len(message) / encode_s,
        "decode_input_bits_per_second": len(message) / decode_s,
    }


if __name__ == "__main__":
    result = selftest()
    print("Noiseless and seeded 4% output-bit-flip tests passed.")
    print(f"Channel flips: {result['noisy_channel_flips']}; residual message-bit errors: {result['noisy_residual_bit_errors']}")
    print(f"Encode: {result['encode_input_bits_per_second']:,.0f} input bits/s")
    print(f"Decode: {result['decode_input_bits_per_second']:,.0f} input bits/s")
    print("Limits: fixed terminated code; independent-bit LLR metric; no puncturing, interleaving, or error detection.")
