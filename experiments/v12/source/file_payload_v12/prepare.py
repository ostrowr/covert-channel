#!/usr/bin/env python3
"""Build a V12 RF stream for an exact user-supplied binary file, off air.

The generated private packet.bits is one ASCII 0/1 per 6,500-Hz physical
memory-workload chip. The slow Morse cue, fast known-text anchor, and repeated
CRC-checked ASCII header bootstrap a single-copy variable-length body. Each block
has a 64-bit balanced Manchester sync word, appended CRC32, K7 coding,
interleaving, and two-chip Manchester. Only the last block is zero-padded. The
RF-carried SHA-256 covers the exact original file bytes in transmission order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_PRIVATE_DIR = ROOT / "private_candidate"
V3 = ROOT.parent / "fast_blind_v3"
sys.path.insert(0, str(V3))
from conv_codec import decode as viterbi_decode, encode as k7_encode  # noqa: E402


CHIP_RATE = 6_500
BLOCK_BYTES = 2_022
BLOCK_COUNT = 18
BLOCK_CAPS = (BLOCK_BYTES,) * BLOCK_COUNT
MAX_MESSAGE_BYTES = sum(BLOCK_CAPS)
MORSE_UNIT_CHIPS = 390  # 60-ms dot
MORSE_TEXT = "SOS NEXT 6500 CHIPS PER SEC MANCHESTER NXT1 REPEATS ASCII BIT3 ZERO10 ONE01 MSB8"
BEACON_BITS = "1111100110101"  # Barker-13
BEACON_SYMBOL_CHIPS = 780  # 120 ms
BEACON_GUARD_CHIPS = 1_300  # 200 ms quiet
MORSE_SYNC_ON_CHIPS = MORSE_UNIT_CHIPS * 6
MORSE_SYNC_OFF_CHIPS = MORSE_UNIT_CHIPS * 8
HANDOFF_QUIET_CHIPS = 3_900  # 600 ms
HANDOFF_ON_CHIPS = 3_900
HANDOFF_OFF_CHIPS = 3_900
FAST_CLOCK_PAIRS = 3_250  # one second at the physical chip clock
FAST_GUARD_CHIPS = 1_300
HEADER_BIT_REPETITIONS = 3  # 0=101010, 1=010101
ANCHOR_WORD = b"NXT1"
ANCHOR_REPEATS = 32
ANCHOR_GUARD_CHIPS = 1_300  # 200 ms quiet after the known-text anchor
HEADER_PREAMBLE_PAIRS = 32
HEADER_COMMA = "1" * 16 + "0" * 16  # deliberate header Manchester violation
HEADER_TRAINING = b"\x55" * 4  # known alternating byte pattern before every line
HEADER_LINE_GAP_CHIPS = 32
DEFAULT_HEADER_PASSES = 4
BODY_HANDOFF_QUIET_CHIPS = 1_300
BODY_HANDOFF_ON_CHIPS = 2_600
BODY_HANDOFF_OFF_CHIPS = 2_600
BODY_PREAMBLE_PAIRS = 128
BODY_SYNC_HEX = "EA30AD19307964BF"
BODY_SYNC_BITS = f"{int(BODY_SYNC_HEX, 16):064b}"
BODY_SYNC_CHIPS = "".join("01" if bit == "1" else "10" for bit in BODY_SYNC_BITS)
BODY_INTERBLOCK_GAP_CHIPS = 1_300
END_GUARD_CHIPS = 1_300
END_QUIET_CHIPS = 650
HEADER_LINE_COUNT = 13

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".",
    "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
    "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---",
    "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--",
    "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....",
    "7": "--...", "8": "---..", "9": "----.",
}


def morse_chips(text: str) -> str:
    parts: list[str] = []
    words = text.split(" ")
    for word_index, word in enumerate(words):
        for letter_index, letter in enumerate(word):
            symbols = MORSE[letter]
            for symbol_index, symbol in enumerate(symbols):
                parts.append("1" * MORSE_UNIT_CHIPS * (3 if symbol == "-" else 1))
                if symbol_index + 1 < len(symbols):
                    parts.append("0" * MORSE_UNIT_CHIPS)
            if letter_index + 1 < len(word):
                parts.append("0" * (3 * MORSE_UNIT_CHIPS))
        if word_index + 1 < len(words):
            parts.append("0" * (7 * MORSE_UNIT_CHIPS))
    return "".join(parts)


def validate_message(message: bytes) -> bytes:
    if not 1 <= len(message) <= MAX_MESSAGE_BYTES:
        raise ValueError(f"source file must contain 1..{MAX_MESSAGE_BYTES} bytes")
    return message


def split_blocks(message: bytes) -> list[bytes]:
    """Split the exact bytes into full transport blocks with a zero-padded tail."""
    validate_message(message)
    blocks: list[bytes] = []
    offset = 0
    for cap in BLOCK_CAPS:
        if offset == len(message):
            break
        part = message[offset : offset + cap]
        blocks.append(part.ljust(cap, b"\0"))
        offset += len(part)
    if offset != len(message):
        raise AssertionError("message exceeds the body block capacity")
    return blocks


def closest_coprime_stride(length: int) -> int:
    target = max(1, length // 32)
    for distance in range(length):
        candidates = (target,) if distance == 0 else (target - distance, target + distance)
        for candidate in candidates:
            if 1 <= candidate < length and math.gcd(candidate, length) == 1:
                return candidate
    raise AssertionError("no coprime interleaver stride")


def encode_body(block: bytes) -> dict[str, object]:
    crc = zlib.crc32(block) & 0xFFFFFFFF
    source = block + crc.to_bytes(4, "big")
    bits = np.unpackbits(np.frombuffer(source, dtype=np.uint8))
    coded = k7_encode(bits)
    length = len(coded)
    stride = closest_coprime_stride(length)
    indices = (np.arange(length, dtype=np.int64) * stride) % length
    interleaved = coded[indices]
    physical = "".join("01" if int(bit) else "10" for bit in interleaved)
    return {
        "source": source,
        "crc": crc,
        "coded_bits": length,
        "stride": stride,
        "interleaved": interleaved,
        "physical": physical,
    }


def header_line_payloads(
    blocks: list[bytes], coded_lengths: list[int], strides: list[int],
    block_crcs: list[int], message_sha256: str, passes: int,
    source_length: int,
) -> list[bytes]:
    if (not 1 <= len(blocks) <= BLOCK_COUNT or
            any(len(block) != BLOCK_BYTES for block in blocks) or
            len(set(coded_lengths)) != 1 or len(set(strides)) != 1 or
            len(block_crcs) != len(blocks)):
        raise AssertionError("V12 header requires equal-length coded blocks")
    fields = [
        "CHIP=6500;0=10;1=01;MSB8;HBIT3;BBIT2",
        "HCRC=IEEE32;BCRC=IEEE32BE;APPEND_TO_RAW_BEFORE_K7",
        f"BLOCKS={len(blocks)};LEN_EACH={BLOCK_BYTES};TOTAL={source_length};BIN;PAD=0",
        "FEC=K7,R1/2,OCT171,133,S0,T6;NEWEST=BIT",
        "TAPS_NEW2OLD=1001111,1101101;OUT=171,133",
        "TEST_INPUT=1+000000;OUTPUT=11010011111011",
        "INT TX[t]=CODED[(t*P)%L]",
        f"P={strides[0]};L={coded_lengths[0]};SAME_EACH_BLOCK",
        f"BPRE=10x128;SYNC64={BODY_SYNC_HEX};G=1300",
        "BCRC=APPENDED_4_BE_EACH;SCOPE=DATA",
        f"CRC32TEST123456789=CBF43926;BODY_AFTER=H13P{passes}",
        f"SHA256_RAW_CONCAT_A={message_sha256[:32]}",
        f"SHA256_RAW_CONCAT_B={message_sha256[32:]}",
    ]
    if len(fields) != HEADER_LINE_COUNT:
        raise AssertionError("header line count changed")
    bodies = [
        f"NXT1:{index:02d}/{HEADER_LINE_COUNT:02d}:{field}".encode("ascii")
        for index, field in enumerate(fields, start=1)
    ]
    lines = [
        body + f"*CRC32={zlib.crc32(body) & 0xFFFFFFFF:08X}\n".encode("ascii")
        for body in bodies
    ]
    if max(map(len, lines)) > 80:
        raise AssertionError("header line exceeds the 80-byte clarity bound")
    return lines


def check_header_line(line: bytes) -> bytes:
    body, marker, suffix = line.partition(b"*CRC32=")
    if marker != b"*CRC32=" or len(suffix) != 9 or suffix[-1:] != b"\n":
        raise AssertionError("header line lacks an ASCII CRC32 field")
    received_crc = int(suffix[:8], 16)
    if (zlib.crc32(body) & 0xFFFFFFFF) != received_crc:
        raise AssertionError("header line CRC32 mismatch")
    if not body.startswith(b"NXT1:"):
        raise AssertionError("header line lacks NXT1 sync")
    return body


def header_chips(line: bytes) -> str:
    pieces = []
    for byte in line:
        for bit_index in range(7, -1, -1):
            pair = "01" if (byte >> bit_index) & 1 else "10"
            pieces.append(pair * HEADER_BIT_REPETITIONS)
    return "".join(pieces)


def decode_header_chips(chips: str) -> bytes:
    chips_per_byte = 8 * 2 * HEADER_BIT_REPETITIONS
    if len(chips) % chips_per_byte:
        raise AssertionError("header chip stream has a partial byte")
    bits: list[int] = []
    for offset in range(0, len(chips), 2 * HEADER_BIT_REPETITIONS):
        symbol = chips[offset : offset + 2 * HEADER_BIT_REPETITIONS]
        pairs = [symbol[i : i + 2] for i in range(0, len(symbol), 2)]
        if any(pair not in ("10", "01") for pair in pairs):
            raise AssertionError("invalid header Manchester pair")
        bits.append(int(sum(pair == "01" for pair in pairs) > len(pairs) // 2))
    return bytes(
        sum(bits[offset + bit_index] << (7 - bit_index) for bit_index in range(8))
        for offset in range(0, len(bits), 8)
    )


def decode_body_chips(chips: str, coded_bits: int, stride: int) -> bytes:
    if len(chips) != 2 * coded_bits:
        raise AssertionError("body coded length mismatch")
    pairs = np.frombuffer(chips.encode("ascii"), dtype=np.uint8).reshape(-1, 2) - 48
    if not np.all(pairs.sum(axis=1) == 1):
        raise AssertionError("body has a non-Manchester chip pair")
    interleaved_llrs = (1 - 2 * pairs[:, 1].astype(np.int16)).astype(np.float64) * 8.0
    coded_llrs = np.empty(coded_bits, dtype=np.float64)
    indices = (np.arange(coded_bits, dtype=np.int64) * stride) % coded_bits
    coded_llrs[indices] = interleaved_llrs
    return np.packbits(viterbi_decode(coded_llrs)).tobytes()


def build(message: bytes, chip_file: Path, passes: int) -> tuple[str, dict[str, object]]:
    blocks = split_blocks(validate_message(message))
    encoded = [encode_body(block) for block in blocks]
    lengths = [len(block) for block in blocks]
    coded_lengths = [int(item["coded_bits"]) for item in encoded]
    strides = [int(item["stride"]) for item in encoded]
    block_crcs = [int(item["crc"]) for item in encoded]
    message_sha256 = hashlib.sha256(message).hexdigest()
    lines = header_line_payloads(
        blocks, coded_lengths, strides, block_crcs, message_sha256, passes,
        len(message),
    )
    beacon = "".join(bit * BEACON_SYMBOL_CHIPS for bit in BEACON_BITS)
    morse = morse_chips(MORSE_TEXT)
    morse_sync = "1" * MORSE_SYNC_ON_CHIPS + "0" * MORSE_SYNC_OFF_CHIPS
    first_handoff = (
        "0" * HANDOFF_QUIET_CHIPS + "1" * HANDOFF_ON_CHIPS
        + "0" * HANDOFF_OFF_CHIPS + "10" * FAST_CLOCK_PAIRS
        + "0" * FAST_GUARD_CHIPS
    )
    second_handoff = (
        "0" * BODY_HANDOFF_QUIET_CHIPS + "1" * BODY_HANDOFF_ON_CHIPS
        + "0" * BODY_HANDOFF_OFF_CHIPS
    )
    header_preamble = "10" * HEADER_PREAMBLE_PAIRS
    header_training = header_chips(HEADER_TRAINING)
    fast_anchor = header_chips(ANCHOR_WORD * ANCHOR_REPEATS)
    body_preamble = "10" * BODY_PREAMBLE_PAIRS
    parts: list[str] = []
    offsets: dict[str, int] = {}
    running = 0

    def append(name: str, chips: str) -> None:
        nonlocal running
        offsets[name] = running
        parts.append(chips)
        running += len(chips)

    append("beacon_start", beacon)
    append("beacon_guard", "0" * BEACON_GUARD_CHIPS)
    append("morse_sync", morse_sync)
    append("morse", morse)
    append("first_handoff", first_handoff)
    offsets["fast_clock"] = offsets["first_handoff"] + (
        HANDOFF_QUIET_CHIPS + HANDOFF_ON_CHIPS + HANDOFF_OFF_CHIPS
    )
    append("fast_anchor", fast_anchor)
    append("fast_anchor_guard", "0" * ANCHOR_GUARD_CHIPS)
    header_start = running
    for pass_index in range(passes):
        for line_index, line in enumerate(lines):
            encoded_line = header_chips(line)
            copy = (
                header_preamble + HEADER_COMMA + header_training + encoded_line
                + "0" * HEADER_LINE_GAP_CHIPS
            )
            name = f"header_{pass_index}_{line_index}"
            append(name, copy)
            offsets[f"header_data_{pass_index}_{line_index}"] = (
                offsets[name] + len(header_preamble) + len(HEADER_COMMA) + len(header_training)
            )
    header_end = running
    append("second_handoff", second_handoff)
    body_start = running
    for index, item in enumerate(encoded):
        physical = str(item["physical"])
        name = f"body_{index}"
        append(name, body_preamble + BODY_SYNC_CHIPS + physical)
        offsets[f"body_sync_{index}"] = offsets[name] + len(body_preamble)
        offsets[f"body_data_{index}"] = offsets[f"body_sync_{index}"] + len(BODY_SYNC_CHIPS)
        if index + 1 < len(encoded):
            append(f"body_gap_{index}", "0" * BODY_INTERBLOCK_GAP_CHIPS)
    body_end = running
    append("end_guard", "0" * END_GUARD_CHIPS)
    append("beacon_end", beacon)
    append("end_quiet", "0" * END_QUIET_CHIPS)
    physical = "".join(parts)
    if len(physical) > CHIP_RATE * 300:
        raise AssertionError("stream exceeds five minutes")
    unique_bits = len(message) * 8
    body_seconds = (body_end - body_start) / CHIP_RATE
    truth: dict[str, object] = {
        "source_chip_file": str(chip_file.resolve()),
        "chip_format": "one ASCII 0/1 per physical chip; no newline",
        "chip_rate": CHIP_RATE,
        "morse_text": MORSE_TEXT,
        "morse_unit_chips": MORSE_UNIT_CHIPS,
        "morse_chips": len(morse),
        "beacon_bits": BEACON_BITS,
        "beacon_symbol_chips": BEACON_SYMBOL_CHIPS,
        "header_bit_repetitions": HEADER_BIT_REPETITIONS,
        "fast_anchor_ascii": ANCHOR_WORD.decode("ascii"),
        "fast_anchor_repeats": ANCHOR_REPEATS,
        "fast_anchor_chips": len(fast_anchor),
        "fast_anchor_guard_chips": ANCHOR_GUARD_CHIPS,
        "header_training_hex": HEADER_TRAINING.hex(),
        "header_training_chips": len(header_training),
        "header_sync_ascii": "NXT1",
        "header_zero_chips": "10" * HEADER_BIT_REPETITIONS,
        "header_one_chips": "01" * HEADER_BIT_REPETITIONS,
        "header_passes": passes,
        "header_line_count": len(lines),
        "header_lines_ascii": [line.decode("ascii") for line in lines],
        "header_line_lengths_bytes": [len(line) for line in lines],
        "header_start_chips": header_start,
        "header_end_chips": header_end,
        "body_start_chips": body_start,
        "body_end_chips": body_end,
        "body_zero_chips": "10",
        "body_one_chips": "01",
        "body_preamble_pairs": BODY_PREAMBLE_PAIRS,
        "body_sync_hex": BODY_SYNC_HEX,
        "body_sync_bits": BODY_SYNC_BITS,
        "body_sync_chips": BODY_SYNC_CHIPS,
        "body_interblock_gap_chips": BODY_INTERBLOCK_GAP_CHIPS,
        "body_block_lengths_bytes": lengths,
        "body_coded_bits": coded_lengths,
        "body_interleaver_strides": strides,
        "body_block_crc32": [f"{crc:08X}" for crc in block_crcs],
        "source_file": str((chip_file.parent / "source.bin").resolve()),
        "message_length_bytes": len(message),
        "transport_padding_bytes": len(blocks) * BLOCK_BYTES - len(message),
        "message_sha256": message_sha256,
        "stage_offsets_chips": offsets,
        "total_chips": len(physical),
        "air_seconds": len(physical) / CHIP_RATE,
        "body_seconds": body_seconds,
        "unique_payload_bps_body": unique_bits / body_seconds,
        "unique_payload_bps_end_to_end": unique_bits * CHIP_RATE / len(physical),
    }
    return physical, truth


def verify_noiseless(physical: str, truth: dict[str, object], message: bytes) -> None:
    if (zlib.crc32(b"123456789") & 0xFFFFFFFF) != 0xCBF43926:
        raise AssertionError("CRC32 reference check failed")
    test = k7_encode(np.array([1], dtype=np.uint8))
    if "".join(map(str, test)) != "11010011111011":
        raise AssertionError("K7 reference vector failed")
    tap_a = "".join(str((0o171 >> index) & 1) for index in range(7))
    tap_b = "".join(str((0o133 >> index) & 1) for index in range(7))
    if (tap_a, tap_b) != ("1001111", "1101101"):
        raise AssertionError("newest-to-oldest K7 tap masks are wrong")
    lines = [line.encode("ascii") for line in truth["header_lines_ascii"]]
    offsets = truth["stage_offsets_chips"]
    anchor_start = int(offsets["fast_anchor"])
    anchor_end = int(offsets["fast_anchor_guard"])
    if anchor_end - anchor_start != ANCHOR_REPEATS * len(ANCHOR_WORD) * 8 * 2 * HEADER_BIT_REPETITIONS:
        raise AssertionError("fast anchor length mismatch")
    if decode_header_chips(physical[anchor_start:anchor_end]) != ANCHOR_WORD * ANCHOR_REPEATS:
        raise AssertionError("fast anchor ASCII round trip failed")
    if physical[anchor_end:anchor_end + ANCHOR_GUARD_CHIPS] != "0" * ANCHOR_GUARD_CHIPS:
        raise AssertionError("fast anchor guard mismatch")
    header_preamble = "10" * HEADER_PREAMBLE_PAIRS
    header_training = header_chips(HEADER_TRAINING)
    for pass_index in range(int(truth["header_passes"])):
        for line_index, expected in enumerate(lines):
            copy_start = int(offsets[f"header_{pass_index}_{line_index}"])
            data_start = int(offsets[f"header_data_{pass_index}_{line_index}"])
            if physical[copy_start : copy_start + len(header_preamble)] != header_preamble:
                raise AssertionError("header preamble mismatch")
            comma_start = copy_start + len(header_preamble)
            if physical[comma_start : comma_start + len(HEADER_COMMA)] != HEADER_COMMA:
                raise AssertionError("header comma mismatch")
            training_start = comma_start + len(HEADER_COMMA)
            if decode_header_chips(
                physical[training_start : data_start]
            ) != HEADER_TRAINING:
                raise AssertionError("header 0x55 training mismatch")
            chip_count = len(expected) * 8 * 2 * HEADER_BIT_REPETITIONS
            decoded = decode_header_chips(physical[data_start : data_start + chip_count])
            if decoded != expected:
                raise AssertionError("header ASCII round trip failed")
            check_header_line(decoded)
    blocks = split_blocks(message)
    recovered: list[bytes] = []
    for index, block in enumerate(blocks):
        preamble_start = int(offsets[f"body_{index}"])
        sync_start = int(offsets[f"body_sync_{index}"])
        if physical[preamble_start:sync_start] != "10" * BODY_PREAMBLE_PAIRS:
            raise AssertionError(f"body block {index} preamble mismatch")
        if physical[sync_start:sync_start + len(BODY_SYNC_CHIPS)] != BODY_SYNC_CHIPS:
            raise AssertionError(f"body block {index} sync mismatch")
        start = int(offsets[f"body_data_{index}"])
        coded_bits = int(truth["body_coded_bits"][index])
        stride = int(truth["body_interleaver_strides"][index])
        decoded = decode_body_chips(physical[start : start + 2 * coded_bits], coded_bits, stride)
        data, crc_bytes = decoded[:-4], decoded[-4:]
        if data != block or (zlib.crc32(data) & 0xFFFFFFFF) != int.from_bytes(crc_bytes, "big"):
            raise AssertionError(f"body block {index} FEC or CRC round trip failed")
        recovered.append(data)
    padded = b"".join(recovered)
    if padded[:len(message)] != message or any(padded[len(message):]):
        raise AssertionError("source or zero padding differs after body decoding")
    if hashlib.sha256(padded[:len(message)]).hexdigest() != truth["message_sha256"]:
        raise AssertionError("full-message hash differs")


def atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        os.fchmod(temporary.fileno(), 0o600)
        temporary.write(data)
        temporary_name = temporary.name
    os.replace(temporary_name, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--source-file", type=Path, required=True,
                        help=f"exact binary file (1..{MAX_MESSAGE_BYTES} bytes)")
    parser.add_argument("--header-passes", type=int, default=DEFAULT_HEADER_PASSES)
    args = parser.parse_args()
    if args.header_passes != DEFAULT_HEADER_PASSES:
        parser.error("the V12 validated profile requires four header passes")
    private = args.output_dir.resolve()
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    private.chmod(0o700)
    chip_file = private / "packet.bits"
    truth_file = private / "truth.json"
    message_file = private / "source.bin"
    existing = any(path.exists() for path in (chip_file, truth_file, message_file))
    if existing and not all(path.exists() for path in (chip_file, truth_file, message_file)):
        raise ValueError("incomplete private packet; choose a new output directory")
    message = validate_message(args.source_file.read_bytes())
    physical, truth = build(message, chip_file, args.header_passes)
    truth["source_basename_private"] = args.source_file.name
    verify_noiseless(physical, truth, message)
    if existing:
        old = json.loads(truth_file.read_text(encoding="ascii"))
        if (old["message_sha256"] != truth["message_sha256"] or
                old["header_passes"] != args.header_passes or
                old["message_length_bytes"] != len(message)):
            raise ValueError("existing private packet differs; choose a new output directory")
        if message_file.read_bytes() != message:
            raise ValueError("existing private source differs; choose a new output directory")
        if chip_file.read_text(encoding="ascii") != physical:
            raise ValueError("existing private chips differ; choose a new output directory")
    else:
        atomic_write(message_file, message)
        atomic_write(truth_file, (json.dumps(truth, indent=2) + "\n").encode("ascii"))
        atomic_write(chip_file, physical.encode("ascii"))
    print(json.dumps({
        "message_bytes": len(message),
        "header_lines": len(truth["header_lines_ascii"]),
        "header_passes": args.header_passes,
        "header_line_max_bytes": max(truth["header_line_lengths_bytes"]),
        "body_blocks": len(truth["body_block_lengths_bytes"]),
        "transport_padding_bytes": truth["transport_padding_bytes"],
        "body_seconds": round(truth["body_seconds"], 3),
        "air_seconds": round(truth["air_seconds"], 3),
        "unique_payload_bps_body": round(truth["unique_payload_bps_body"], 1),
        "unique_payload_bps_end_to_end": round(truth["unique_payload_bps_end_to_end"], 1),
        "noiseless_header_body_crc_hash": True,
    }))


if __name__ == "__main__":
    main()
