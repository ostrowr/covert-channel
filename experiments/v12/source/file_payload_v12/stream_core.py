"""Received-header and K7 checks for the binary-file Manchester stream.

This module does not load private truth. It accepts only decoded RF bytes and
soft Manchester margins. The sender's final header syntax can extend the
field parser without changing the CRC or body-decode trust boundary.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "fast_blind_v3"))
from conv_codec import decode, encode  # noqa: E402


@dataclass(frozen=True)
class HeaderParameters:
    coded_bits: tuple[int, ...]
    interleaver_strides: tuple[int, ...]
    payload_bytes: tuple[int, ...]
    block_crc32: tuple[int | None, ...]
    message_sha256: str | None
    source_length_bytes: int
    body_after_header_passes: int
    body_preamble_pairs: int
    body_sync_hex: str
    body_guard_chips: int

    @property
    def body_sync_chips(self) -> str:
        bits = f"{int(self.body_sync_hex, 16):064b}"
        return "".join("01" if bit == "1" else "10" for bit in bits)


@dataclass(frozen=True)
class BlockDecode:
    payload: bytes
    crc32_valid: bool
    header_crc32_matches: bool | None


def parse_received_header_line(data: bytes) -> tuple[int, int, bytes] | None:
    """Validate one RF-decoded numbered ASCII line using its own CRC32."""
    match = re.fullmatch(rb"(NXT1:([0-9]{2})/([0-9]{2}):([^\n]*)\*CRC32=([0-9A-F]{8}))\n", data)
    if match is None:
        return None
    body = data[:data.index(b"*CRC32=")]
    if (zlib.crc32(body) & 0xFFFFFFFF) != int(match.group(5), 16):
        return None
    index = int(match.group(2))
    total = int(match.group(3))
    if not 1 <= index <= total <= 32:
        return None
    return index, total, match.group(4)


def assemble_received_header(lines: list[bytes]) -> HeaderParameters | None:
    """Use only independently CRC-valid RF lines to build body parameters."""
    if not lines:
        return None
    parsed = [parse_received_header_line(line) for line in lines]
    if any(item is None for item in parsed):
        return None
    assert all(item is not None for item in parsed)
    total = len(lines)
    if any(index != position + 1 or count != total
           for position, (index, count, _) in enumerate(parsed)):
        return None
    fields = [field for _, _, field in parsed]
    if b"CHIP=6500;0=10;1=01;MSB8;HBIT3;BBIT2" not in fields:
        return None
    required = (
        b"HCRC=IEEE32;BCRC=IEEE32BE;APPEND_TO_RAW_BEFORE_K7",
        b"FEC=K7,R1/2,OCT171,133,S0,T6;NEWEST=BIT",
        b"TAPS_NEW2OLD=1001111,1101101;OUT=171,133",
        b"TEST_INPUT=1+000000;OUTPUT=11010011111011",
        b"INT TX[t]=CODED[(t*P)%L]",
    )
    if any(value not in fields for value in required):
        return None
    lengths = next((re.fullmatch(rb"BLOCKS=([0-9]+);LEN_EACH=([0-9]+);TOTAL=([0-9]+);BIN;PAD=0", f)
                    for f in fields if f.startswith(b"BLOCKS=")), None)
    coded = next((re.fullmatch(rb"P=([0-9]+);L=([0-9]+);SAME_EACH_BLOCK", f)
                  for f in fields if f.startswith(b"P=")), None)
    sync = next((re.fullmatch(rb"BPRE=10x([0-9]+);SYNC64=([0-9A-F]{16});G=([0-9]+)", f)
                 for f in fields if f.startswith(b"BPRE=")), None)
    crcs = b"BCRC=APPENDED_4_BE_EACH;SCOPE=DATA" in fields
    if lengths is None or coded is None or sync is None or not crcs:
        return None
    preamble_pairs = int(sync.group(1))
    sync_hex = sync.group(2).decode("ascii")
    guard_chips = int(sync.group(3))
    if not 1 <= preamble_pairs <= 2_048 or not 0 <= guard_chips <= 50_000:
        return None
    if int(sync_hex, 16).bit_count() != 32:
        return None
    block_count = int(lengths.group(1))
    bytes_each = int(lengths.group(2))
    total_bytes = int(lengths.group(3))
    if (not 1 <= block_count <= 32 or not 1 <= bytes_each <= 100_000 or
            not (block_count - 1) * bytes_each < total_bytes <= block_count * bytes_each):
        return None
    n_values = (bytes_each,) * block_count
    p_values = (int(coded.group(1)),) * block_count
    l_values = (int(coded.group(2)),) * block_count
    crc_values = (None,) * block_count
    sha_a = next((re.fullmatch(rb"SHA256_RAW_CONCAT_A=([0-9a-f]{32})", f)
                  for f in fields if f.startswith(b"SHA256_RAW_CONCAT_A=")), None)
    sha_b = next((re.fullmatch(rb"SHA256_RAW_CONCAT_B=([0-9a-f]{32})", f)
                  for f in fields if f.startswith(b"SHA256_RAW_CONCAT_B=")), None)
    handoff = next((re.fullmatch(rb"CRC32TEST123456789=CBF43926;BODY_AFTER=H([0-9]+)P([0-9]+)", f)
                    for f in fields if f.startswith(b"CRC32TEST")), None)
    if sha_a is None or sha_b is None or handoff is None:
        return None
    if int(handoff.group(1)) != total:
        return None
    for coded_bits, stride, payload_bytes in zip(l_values, p_values, n_values):
        if not 1 <= payload_bytes <= 100_000:
            return None
        if coded_bits != 2 * (8 * (payload_bytes + 4) + 6):
            return None
        if not 1 <= stride < coded_bits or math.gcd(stride, coded_bits) != 1:
            return None
    return HeaderParameters(l_values, p_values, n_values, crc_values,
                            (sha_a.group(1) + sha_b.group(1)).decode("ascii"),
                            total_bytes,
                            int(handoff.group(2)), preamble_pairs,
                            sync_hex, guard_chips)


def validate_k7_convention() -> bool:
    """Check the in-band K7 test vector against the actual receiver codec."""
    encoded = encode(np.array([1], dtype=np.uint8))
    return "".join(map(str, encoded)) == "11010011111011"


def decode_interleaved_block(rx_margins: np.ndarray, length: int, stride: int,
                             payload_bytes: int,
                             header_crc32: int | None = None) -> BlockDecode:
    """RF margin >0 favors transmitted 1; conv_codec LLR >0 favors 0."""
    margins = np.asarray(rx_margins, dtype=np.float64)
    if len(margins) != length or not np.all(np.isfinite(margins)):
        raise ValueError("invalid received coded-bit margins")
    if length != 2 * (8 * (payload_bytes + 4) + 6):
        raise ValueError("coded length disagrees with received payload length")
    if not 1 <= stride < length or math.gcd(stride, length) != 1:
        raise ValueError("invalid received interleaver stride")
    permutation = (np.arange(length, dtype=np.int64) * stride) % length
    coded_llrs = np.empty(length, dtype=np.float64)
    coded_llrs[permutation] = -margins
    decoded_bits = decode(coded_llrs)
    raw = np.packbits(decoded_bits, bitorder="big").tobytes()
    payload, wire_crc = raw[:-4], int.from_bytes(raw[-4:], "big")
    measured_crc = zlib.crc32(payload) & 0xFFFFFFFF
    return BlockDecode(payload, measured_crc == wire_crc,
                       None if header_crc32 is None else measured_crc == header_crc32)


def overall_sha256_valid(parts: list[bytes], header: HeaderParameters) -> bool | None:
    if header.message_sha256 is None:
        return None
    raw = b"".join(parts)
    if len(raw) != sum(header.payload_bytes):
        return False
    if any(raw[header.source_length_bytes:]):
        return False
    return hashlib.sha256(raw[:header.source_length_bytes]).hexdigest() == header.message_sha256


def recover_file_bytes(parts: list[bytes], header: HeaderParameters) -> bytes:
    """Return original bytes only after the RF-carried hash and padding verify."""
    if not overall_sha256_valid(parts, header):
        raise ValueError("block bytes, zero padding, or RF-carried SHA-256 mismatch")
    return b"".join(parts)[:header.source_length_bytes]
