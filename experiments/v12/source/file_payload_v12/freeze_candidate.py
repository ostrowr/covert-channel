#!/usr/bin/env python3
"""Freeze a V12 source file and its exact off-air packet before hardware use."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from prepare import (
    BLOCK_BYTES, CHIP_RATE, HEADER_LINE_COUNT, MAX_MESSAGE_BYTES,
    BODY_SYNC_CHIPS, verify_noiseless,
)
from stream_core import assemble_received_header


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_dir", type=Path)
    args = parser.parse_args()
    private = args.packet_dir.resolve()
    packet = (private / "packet.bits").read_bytes()
    source = (private / "source.bin").read_bytes()
    truth_bytes = (private / "truth.json").read_bytes()
    truth = json.loads(truth_bytes)
    assert 1 <= len(source) <= MAX_MESSAGE_BYTES
    assert len(packet) == truth["total_chips"]
    assert set(packet) <= {ord("0"), ord("1")}
    assert 1 <= len(packet) / CHIP_RATE <= 300
    assert len(source) == truth["message_length_bytes"]
    assert digest(source) == truth["message_sha256"]
    assert truth["header_line_count"] == HEADER_LINE_COUNT == 13
    assert truth["header_passes"] == 4
    blocks = math.ceil(len(source) / BLOCK_BYTES)
    assert len(truth["body_block_lengths_bytes"]) == blocks
    assert set(truth["body_block_lengths_bytes"]) == {BLOCK_BYTES}
    assert truth["transport_padding_bytes"] == blocks * BLOCK_BYTES - len(source)
    assert truth["body_sync_chips"] == BODY_SYNC_CHIPS
    assert len(BODY_SYNC_CHIPS) == 128
    lines = [line.encode("ascii") for line in truth["header_lines_ascii"]]
    header = assemble_received_header(lines)
    assert header is not None
    assert len(header.coded_bits) == blocks
    assert header.source_length_bytes == len(source)
    assert header.message_sha256 == digest(source)
    for index in range(blocks):
        part = source[index * BLOCK_BYTES:(index + 1) * BLOCK_BYTES].ljust(BLOCK_BYTES, b"\0")
        import zlib
        assert zlib.crc32(part) & 0xFFFFFFFF == int(truth["body_block_crc32"][index], 16)
        start = truth["stage_offsets_chips"][f"body_sync_{index}"]
        end = truth["stage_offsets_chips"][f"body_data_{index}"]
        assert packet[start:end] == BODY_SYNC_CHIPS.encode("ascii")
    verify_noiseless(packet.decode("ascii"), truth, source)
    result = {
        "candidate_kind": "off_air_binary_file_packet",
        "packet_file": "packet.bits",
        "packet_chips": len(packet),
        "packet_sha256": digest(packet),
        "source_file": "source.bin",
        "source_bytes": len(source),
        "source_sha256": digest(source),
        "source_basename_private": truth.get("source_basename_private"),
        "truth_file": "truth.json",
        "truth_sha256": digest(truth_bytes),
        "body_blocks": blocks,
        "body_bytes_per_block": BLOCK_BYTES,
        "transport_padding_bytes": truth["transport_padding_bytes"],
        "body_sync_hex": truth["body_sync_hex"],
        "header_line_count": len(lines),
        "header_passes": truth["header_passes"],
        "air_seconds": truth["air_seconds"],
        "body_seconds": truth["body_seconds"],
        "planned_source_bps_end_to_end": truth["unique_payload_bps_end_to_end"],
        "planned_source_bps_body": truth["unique_payload_bps_body"],
        "noiseless_header_crc_and_block_crc_and_sha": True,
    }
    serialized = (json.dumps(result, indent=2) + "\n").encode("utf-8")
    manifest = private / "frozen_manifest.json"
    if manifest.exists():
        if manifest.read_bytes() != serialized:
            raise ValueError("existing frozen manifest disagrees; use a new packet directory")
    else:
        with manifest.open("xb") as output:
            output.write(serialized)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
