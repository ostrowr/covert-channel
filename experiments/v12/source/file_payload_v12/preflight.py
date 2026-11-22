#!/usr/bin/env python3
"""Off-air receiver-side round trip for a frozen V12 file packet.

The RF protocol parser sees only transmitted header bytes and chips. The
original file is read only for the final equality assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare import BODY_SYNC_CHIPS, HEADER_BIT_REPETITIONS, decode_header_chips
from stream_core import (
    assemble_received_header,
    decode_interleaved_block,
    parse_received_header_line,
    recover_file_bytes,
    validate_k7_convention,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("--expected-file", type=Path, required=True)
    args = parser.parse_args()
    source = args.expected_file.read_bytes()
    private = args.packet_dir
    truth = json.loads((private / "truth.json").read_text(encoding="ascii"))
    physical = (private / "packet.bits").read_text(encoding="ascii")
    offsets = truth["stage_offsets_chips"]
    if len(physical) != truth["total_chips"] or set(physical) - {"0", "1"}:
        raise AssertionError("invalid physical chip file")
    if not validate_k7_convention():
        raise AssertionError("K7 convention check failed")
    lines = []
    for line_index, expected_length in enumerate(truth["header_line_lengths_bytes"]):
        chip_start = offsets[f"header_data_0_{line_index}"]
        chip_end = chip_start + expected_length * 8 * 2 * HEADER_BIT_REPETITIONS
        received = decode_header_chips(physical[chip_start:chip_end])
        if parse_received_header_line(received) is None:
            raise AssertionError(f"RF header line {line_index + 1} CRC failed")
        lines.append(received)
    header = assemble_received_header(lines)
    if header is None or header.source_length_bytes != len(source):
        raise AssertionError("RF header parameters or source length failed")
    if len(header.payload_bytes) != len(truth["body_block_lengths_bytes"]):
        raise AssertionError("RF block count failed")
    blocks = []
    for index, (coded_bits, stride, byte_count) in enumerate(zip(
        header.coded_bits, header.interleaver_strides, header.payload_bytes
    )):
        sync_start = offsets[f"body_sync_{index}"]
        if physical[sync_start:sync_start + len(BODY_SYNC_CHIPS)] != header.body_sync_chips:
            raise AssertionError(f"RF sync failed for block {index + 1}")
        chip_start = offsets[f"body_data_{index}"]
        pairs = physical[chip_start:chip_start + 2 * coded_bits]
        if len(pairs) != 2 * coded_bits or any(
            pairs[j:j + 2] not in ("01", "10") for j in range(0, len(pairs), 2)
        ):
            raise AssertionError(f"RF Manchester pairs failed for block {index + 1}")
        margins = np.fromiter(
            (8.0 if pairs[j:j + 2] == "01" else -8.0 for j in range(0, len(pairs), 2)),
            dtype=np.float64, count=coded_bits,
        )
        decoded = decode_interleaved_block(margins, coded_bits, stride, byte_count)
        if not decoded.crc32_valid:
            raise AssertionError(f"RF body CRC failed for block {index + 1}")
        blocks.append(decoded.payload)
    recovered = recover_file_bytes(blocks, header)
    if recovered != source:
        raise AssertionError("recovered file bytes differ from supplied source")
    print(json.dumps({
        "source_bytes": len(source),
        "transport_blocks": len(blocks),
        "padding_bytes": len(blocks) * header.payload_bytes[0] - len(source),
        "sha256": hashlib.sha256(recovered).hexdigest(),
        "header_crc_lines": len(lines),
        "body_crc_blocks": len(blocks),
        "source_equals_receiver_output": True,
        "air_seconds": truth["air_seconds"],
        "source_bps_end_to_end": truth["unique_payload_bps_end_to_end"],
    }, indent=2))


if __name__ == "__main__":
    main()
