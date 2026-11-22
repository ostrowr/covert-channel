"""Systematic rate-4/5 soft-decision code for the SDR near-field link.

One block contains 2048 payload bits, a CRC-16/CCITT, and 496 parity bits:
2560 transmitted bits total. At 5000 Manchester symbols/s this carries
2000 payload bits/s. The check matrix is a fixed quasi-cyclic irregular
repeat-accumulate (IRA) LDPC construction. It has 2064 systematic bits and
496 accumulator parity bits; each systematic bit participates in three
checks. No generated matrix or external LDPC package is required.

LLRs use the convention log(P(bit=0)/P(bit=1)): positive favors zero.
``decode`` returns the payload only when *both* parity checks and CRC pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PAYLOAD_BITS = 2048
CRC_BITS = 16
SYSTEMATIC_BITS = PAYLOAD_BITS + CRC_BITS
PARITY_BITS = 496
CODEWORD_BITS = SYSTEMATIC_BITS + PARITY_BITS
CODE_RATE = SYSTEMATIC_BITS / CODEWORD_BITS
USEFUL_RATE = PAYLOAD_BITS / CODEWORD_BITS

# Each tuple defines three circulant shifts for one 496-bit input group.
# Pairwise differences are distinct, so the systematic subgraph has no
# length-four cycles. These constants are part of the wire format.
_SHIFTS = ((89, 269, 480), (143, 178, 404), (42, 172, 311),
           (82, 92, 185), (84, 87, 179))


def _bits(value: Iterable[int], count: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8)
    if result.shape != (count,) or np.any(result > 1):
        raise ValueError(f"{name} must contain exactly {count} zero/one bits")
    return result


def crc16(bits: Iterable[int]) -> int:
    """CRC-16/CCITT-FALSE over a bit stream, MSB first (init FFFF)."""
    reg = 0xFFFF
    for bit in bits:
        top = (reg >> 15) & 1
        reg = (reg << 1) & 0xFFFF
        if top ^ int(bit):
            reg ^= 0x1021
    return reg


def _crc_bits(payload: np.ndarray) -> np.ndarray:
    value = crc16(payload)
    return np.fromiter(((value >> shift) & 1 for shift in range(15, -1, -1)),
                       dtype=np.uint8, count=CRC_BITS)


def _crc_influence() -> tuple[np.ndarray, dict[int, tuple[int, ...]]]:
    """CRC change caused by flipping one systematic bit, including CRC bits."""
    changes = np.empty(SYSTEMATIC_BITS, dtype=np.uint16)
    value = 0x1021
    for index in range(PAYLOAD_BITS - 1, -1, -1):
        changes[index] = value
        value = ((value << 1) & 0xFFFF) ^ (0x1021 if value & 0x8000 else 0)
    for index in range(CRC_BITS):
        changes[PAYLOAD_BITS + index] = 1 << (15 - index)
    by_change: dict[int, list[int]] = {}
    for index, change in enumerate(changes):
        by_change.setdefault(int(change), []).append(index)
    return changes, {change: tuple(indices) for change, indices in by_change.items()}


_CRC_INFLUENCE, _CRC_CHANGE_INDEX = _crc_influence()


def _build_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[int]] = [[] for _ in range(PARITY_BITS)]
    for group, shifts in enumerate(_SHIFTS):
        for j in range(PARITY_BITS):
            index = group * PARITY_BITS + j
            if index >= SYSTEMATIC_BITS:
                break
            for shift in shifts:
                rows[(j + shift) % PARITY_BITS].append(index)
    for row in range(PARITY_BITS):
        rows[row].append(SYSTEMATIC_BITS + row)
        if row:
            rows[row].append(SYSTEMATIC_BITS + row - 1)

    width = max(map(len, rows))
    nodes = np.zeros((PARITY_BITS, width), dtype=np.int32)
    mask = np.zeros((PARITY_BITS, width), dtype=bool)
    edges = np.full((PARITY_BITS, width), -1, dtype=np.int32)
    edge_vars: list[int] = []
    for row, variables in enumerate(rows):
        n = len(variables)
        nodes[row, :n] = variables
        mask[row, :n] = True
        edges[row, :n] = np.arange(len(edge_vars), len(edge_vars) + n)
        edge_vars.extend(variables)
    return nodes, mask, edges, np.asarray(edge_vars, dtype=np.int32)


_NODES, _MASK, _EDGES, _EDGE_VARS = _build_graph()


def syndrome(word: Iterable[int]) -> np.ndarray:
    """Return 496 parity-check bits; all zero means a valid codeword."""
    bits = _bits(word, CODEWORD_BITS, "codeword")
    return (np.sum(bits[_NODES] * _MASK, axis=1) & 1).astype(np.uint8)


def encode(data: Iterable[int]) -> np.ndarray:
    """Encode 2048 payload bits; return 2560 systematic codeword bits."""
    payload = _bits(data, PAYLOAD_BITS, "data")
    systematic = np.concatenate((payload, _crc_bits(payload)))
    return _encode_systematic(systematic)


def _encode_systematic(systematic: np.ndarray) -> np.ndarray:
    """Compute parity for any 2064 systematic bits (CRC need not match)."""
    # Each check is XOR(systematic neighbors, current parity, prior parity)=0.
    parity = np.empty(PARITY_BITS, dtype=np.uint8)
    previous = 0
    for row in range(PARITY_BITS):
        acc = previous
        for group, shifts in enumerate(_SHIFTS):
            for shift in shifts:
                index = group * PARITY_BITS + (row - shift) % PARITY_BITS
                if index < SYSTEMATIC_BITS:
                    acc ^= int(systematic[index])
        parity[row] = acc
        previous = acc
    return np.concatenate((systematic, parity))


@dataclass(frozen=True)
class DecodeResult:
    data: np.ndarray | None
    syndrome_ok: bool
    crc_ok: bool
    iterations: int
    hard_errors_corrected: int
    list_repaired: bool = False

    @property
    def ok(self) -> bool:
        return self.data is not None


def _crc_list_repair(hard: np.ndarray, llr: np.ndarray) -> np.ndarray | None:
    """Search CRC-valid neighbors of a parity-valid tentative word.

    The syndrome can be zero for an incorrect codeword. The 16-bit CRC
    residual identifies all one- and two-bit changes to the systematic part
    that would repair its CRC. Score their *full re-encoded codewords* against
    the received soft evidence. Permit a candidate at most one median
    nonzero |LLR| below the tentative word only when it also beats the next
    CRC-valid candidate by six such units. This uses no known payload bits.
    CRC is used to select the candidate, so its pass is not an independent
    integrity check for list repairs; inspect ``DecodeResult.list_repaired``.
    """
    systematic = hard[:SYSTEMATIC_BITS].astype(np.uint8)
    received_crc = 0
    for bit in systematic[PAYLOAD_BITS:]:
        received_crc = (received_crc << 1) | int(bit)
    residual = crc16(systematic[:PAYLOAD_BITS]) ^ received_crc
    if residual == 0:
        return None

    candidate_indices: list[tuple[int, ...]] = [
        (index,) for index in _CRC_CHANGE_INDEX.get(residual, ())
    ]
    for first, change in enumerate(_CRC_INFLUENCE):
        for second in _CRC_CHANGE_INDEX.get(int(change) ^ residual, ()):
            if second > first:
                candidate_indices.append((first, second))

    weights = llr.astype(np.float64)
    tentative_score = float(np.dot(weights, 1 - 2 * hard.astype(np.int16)))
    best_score = -np.inf
    second_score = -np.inf
    best_word: np.ndarray | None = None
    for indices in candidate_indices:
        candidate_systematic = systematic.copy()
        candidate_systematic[list(indices)] ^= 1
        candidate = _encode_systematic(candidate_systematic)
        score = float(np.dot(weights, 1 - 2 * candidate.astype(np.int16)))
        if score > best_score:
            second_score = best_score
            best_score = score
            best_word = candidate
        elif score > second_score:
            second_score = score
    nonzero = np.abs(weights[weights != 0])
    median = float(np.median(nonzero)) if nonzero.size else np.inf
    if best_score < tentative_score - median or best_score - second_score < 6.0 * median:
        return None
    return best_word


def decode(soft_llrs: Iterable[float], *, max_iterations: int = 60) -> DecodeResult:
    """Decode one block using sum-product belief propagation and CRC gating.

    ``soft_llrs`` must contain 2560 finite real values. Positive means the
    receiver favors zero, and its magnitude expresses confidence. A failed
    block has ``data=None``; do not use its tentative bits as a payload.
    """
    llr = np.asarray(soft_llrs, dtype=np.float64)
    if llr.shape != (CODEWORD_BITS,) or not np.all(np.isfinite(llr)):
        raise ValueError(f"soft_llrs must contain {CODEWORD_BITS} finite values")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    llr = np.clip(llr, -30.0, 30.0)
    initial = llr < 0
    c_to_v = np.zeros(len(_EDGE_VARS), dtype=np.float64)
    hard = initial.copy()

    for iteration in range(max_iterations + 1):
        if iteration:
            belief = llr + np.bincount(_EDGE_VARS, weights=c_to_v,
                                      minlength=CODEWORD_BITS)
            hard = belief < 0
        checks = (np.sum(hard[_NODES] * _MASK, axis=1) & 1)
        parity_ok = bool(np.all(checks == 0))
        payload = hard[:PAYLOAD_BITS].astype(np.uint8)
        crc_ok = bool(np.array_equal(hard[PAYLOAD_BITS:SYSTEMATIC_BITS],
                                     _crc_bits(payload)))
        if parity_ok and crc_ok:
            return DecodeResult(payload, True, True, iteration,
                                int(np.count_nonzero(hard != initial)))
        if iteration == max_iterations:
            break

        # Variable-to-check messages are posterior LLRs with the destination
        # check's previous contribution removed.
        v_to_c = (llr + np.bincount(_EDGE_VARS, weights=c_to_v,
                                   minlength=CODEWORD_BITS))[_EDGE_VARS] - c_to_v
        padded = np.zeros_like(_EDGES, dtype=np.float64)
        padded[_MASK] = v_to_c[_EDGES[_MASK]]
        tanh_values = np.tanh(np.clip(padded, -30.0, 30.0) / 2.0)
        # Padded entries are neutral elements of the check product.
        tanh_values[~_MASK] = 1.0
        signs = np.where(tanh_values < 0.0, -1.0, 1.0)
        magnitudes = np.maximum(np.abs(tanh_values), 1e-12)
        log_magnitudes = np.log(magnitudes)
        sign_product = np.prod(signs, axis=1)
        log_product = np.sum(log_magnitudes, axis=1)
        excluding_self = (sign_product[:, None] * signs *
                          np.exp(log_product[:, None] - log_magnitudes))
        messages = 2.0 * np.arctanh(np.clip(excluding_self, -1 + 1e-12,
                                            1 - 1e-12))
        # Mild damping improves convergence on the accumulator's short loops.
        c_to_v = 0.75 * messages[_MASK] + 0.25 * c_to_v

    if parity_ok and not crc_ok:
        repaired = _crc_list_repair(hard, llr)
        if repaired is not None:
            payload = repaired[:PAYLOAD_BITS].copy()
            return DecodeResult(payload, True, True, max_iterations,
                                int(np.count_nonzero(repaired != initial)), True)

    return DecodeResult(None, parity_ok, crc_ok, max_iterations,
                        int(np.count_nonzero(hard != initial)))
