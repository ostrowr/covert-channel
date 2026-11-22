#!/usr/bin/env python3
"""Audit a V12 binary-file stream from measured SDR IQ.

The sender truth supplies physical stage timing and private error counts. Body
parameters come only from an independently received CRC-valid ASCII header.
The report omits message text and all digest/CRC literals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "robust_sync_v11"))
import rf_core as rf
from stream_core import (assemble_received_header, decode_interleaved_block,
                         overall_sha256_valid, parse_received_header_line,
                         validate_k7_convention)


ROOT = Path(__file__).resolve().parent
DEFAULT_PRIVATE = ROOT / "private_candidate"


def private_input(private: Path) -> tuple[dict, np.ndarray, list[bytes], bytes]:
    truth = json.loads((private / "truth.json").read_text())
    chip_bytes = (private / "packet.bits").read_bytes().strip()
    if not chip_bytes or set(chip_bytes) - {ord("0"), ord("1")}:
        raise ValueError("private physical chip file is not binary ASCII")
    physical = np.frombuffer(chip_bytes, dtype=np.uint8) - ord("0")
    if len(physical) != int(truth["total_chips"]):
        raise ValueError("private physical length disagrees with truth")
    header_lines = [line.encode("ascii") for line in truth["header_lines_ascii"]]
    if assemble_received_header(header_lines) is None:
        raise ValueError("private header-line self-check failed")
    message = (private / "source.bin").read_bytes()
    if len(message) != int(truth["message_length_bytes"]):
        raise ValueError("private message length disagrees with truth")
    if hashlib.sha256(message).hexdigest() != truth["message_sha256"]:
        raise ValueError("private message SHA disagrees with truth")
    return truth, physical, header_lines, message


def known_prefix(physical: np.ndarray, stage_offset: int, length: int) -> np.ndarray:
    prefix = physical[stage_offset:stage_offset + length]
    if len(prefix) != length:
        raise ValueError("private timing prefix is incomplete")
    return prefix


def header_copy(amp: np.ndarray, predicted: float, rate: float,
                prefix: np.ndarray, expected_line: bytes,
                pair_repetitions: int, expected_index: int,
                expected_total: int, training_chips: int) -> tuple[dict, np.ndarray, bytes]:
    start, quality = rf.align_stage(amp, predicted, rate, prefix)
    if training_chips != 4 * 8 * 2 * pair_repetitions or training_chips > len(prefix):
        raise ValueError("header training geometry is inconsistent")
    selected_phase = 0.0
    selected_rate = rate
    selected_margins = None
    selected_data = None
    valid_line = False
    for phase in (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4):
        trial_margins = rf.repeated_manchester_margins(
            amp, start, rate, len(prefix), len(expected_line) * 8,
            pair_repetitions, phase_chips=phase)
        trial_data = np.packbits((trial_margins > 0).astype(np.uint8),
                                 bitorder="big").tobytes()
        parsed = parse_received_header_line(trial_data)
        trial_valid = (parsed is not None and parsed[0] == expected_index and
                       parsed[1] == expected_total)
        if selected_margins is None or trial_valid:
            selected_margins = trial_margins
            selected_data = trial_data
            selected_phase = phase
            valid_line = trial_valid
        if trial_valid:
            break
    if not valid_line:
        # A short header line can need a slightly different local chip clock.
        # Search candidates solely by the CRC carried inside the received
        # numbered line. No private header bytes choose this correction.
        for local_rate in np.arange(rate - 1.0, rate + 1.0001, 0.1):
            if abs(local_rate - rate) < 1e-6:
                continue
            for phase in np.arange(-0.5, 0.5001, 0.05):
                trial_margins = rf.repeated_manchester_margins(
                    amp, start, float(local_rate), len(prefix),
                    len(expected_line) * 8, pair_repetitions,
                    phase_chips=float(phase))
                trial_data = np.packbits((trial_margins > 0).astype(np.uint8),
                                         bitorder="big").tobytes()
                parsed = parse_received_header_line(trial_data)
                if parsed is not None and parsed[0] == expected_index and parsed[1] == expected_total:
                    selected_margins = trial_margins
                    selected_data = trial_data
                    selected_phase = round(float(phase), 3)
                    selected_rate = round(float(local_rate), 3)
                    valid_line = True
                    break
            if valid_line:
                break
    assert selected_margins is not None and selected_data is not None
    training_margin = rf.repeated_manchester_margins(
        amp, start, selected_rate, len(prefix) - training_chips, 4 * 8,
        pair_repetitions, phase_chips=selected_phase)
    training_bytes = np.packbits((training_margin > 0).astype(np.uint8), bitorder="big").tobytes()
    margins = selected_margins
    hard = (margins > 0).astype(np.uint8)
    data = selected_data
    expected_bits = np.unpackbits(np.frombuffer(expected_line, dtype=np.uint8), bitorder="big")
    result = {
        "start_seconds": round(start, 6),
        "preamble_quality": round(quality, 4),
        "selected_phase_chips": selected_phase,
        "selected_chip_rate": selected_rate,
        "training_55x4_valid": training_bytes == b"\x55" * 4,
        "training_bit_errors": int(np.count_nonzero(
            (training_margin > 0) != np.unpackbits(
                np.frombuffer(b"\x55" * 4, dtype=np.uint8), bitorder="big"))),
        "received_crc32_valid": valid_line,
        "raw_header_bit_errors_vs_private": int(np.count_nonzero(hard != expected_bits)),
        "exact_private_line": data == expected_line,
    }
    return result, margins, data


def body_one_mode(amp: np.ndarray, start: float, rate: float, prefix_len: int,
                  coded_bits: int, stride: int, payload_bytes: int,
                  header_crc32: int | None,
                  expected_tx_bits: np.ndarray, private_block: bytes,
                  mode: str) -> tuple[dict, bytes]:
    if mode == "fixed":
        margins = rf.manchester_margins(amp, start, rate, prefix_len, coded_bits)
        phase = (0.0, 0.0)
    elif mode == "tracked":
        margins, phase = rf.tracked_manchester_margins(amp, start, rate,
                                                       prefix_len, coded_bits)
    else:
        raise ValueError("unknown body timing mode")
    decoded = decode_interleaved_block(margins, coded_bits, stride, payload_bytes,
                                       header_crc32)
    same_private_length = len(decoded.payload) == len(private_block)
    decoded_errors = (int(np.count_nonzero(
        np.unpackbits(np.frombuffer(decoded.payload, dtype=np.uint8), bitorder="big") !=
        np.unpackbits(np.frombuffer(private_block, dtype=np.uint8), bitorder="big")))
        if same_private_length else None)
    report = {
        "mode": mode,
        "raw_interleaved_bit_errors_vs_private": int(np.count_nonzero(
            (margins > 0) != expected_tx_bits)),
        "crc32_valid": decoded.crc32_valid,
        "crc32_matches_received_header": decoded.header_crc32_matches,
        "length_matches_received_header": len(decoded.payload) == payload_bytes,
        "phase_range_chips": [round(value, 4) for value in phase],
        "decoded_bit_errors_vs_private": decoded_errors,
        "exact_private_block": bool(decoded.crc32_valid and decoded.header_crc32_matches is not False and
                                     same_private_length and decoded.payload == private_block),
    }
    return report, decoded.payload


def run(public: Path, filename: str, private: Path) -> dict:
    truth, physical, expected_lines, private_message = private_input(private)
    iq_path, fs, center, sample_count, sha_ok, capture_sha = rf.load_capture(public, filename)
    if not sha_ok:
        raise ValueError("physical IQ SHA-256 disagrees with capture metadata")
    raw = np.memmap(iq_path, dtype="<i2", mode="r")
    if raw.size != sample_count * 2:
        raise ValueError("physical IQ sample count disagrees with metadata")
    rf_hz, rf_contrast = rf.spectral_peak(raw, fs, center)
    amp = rf.envelope(raw, fs, center, rf_hz)
    rate = float(truth["chip_rate"])
    t0, start_quality = rf.find_packet_start(amp, physical, int(rate))
    offsets = truth["stage_offsets_chips"]
    header_passes = int(truth["header_passes"])
    header_line_count = int(truth["header_line_count"])
    if len(expected_lines) != header_line_count:
        raise ValueError("private header line count disagrees with truth")
    header_pair_repetitions = int(truth["header_bit_repetitions"])
    header_reports: list[list[dict]] = [[] for _ in range(header_line_count)]
    header_margins: list[list[np.ndarray]] = [[] for _ in range(header_line_count)]
    header_candidates: list[list[tuple[int, bytes]]] = [[] for _ in range(header_line_count)]
    for pass_index in range(header_passes):
        for line_index, expected_line in enumerate(expected_lines):
            name = f"header_{pass_index}_{line_index}"
            chip_offset = int(offsets[name])
            prefix_len = int(offsets[f"header_data_{pass_index}_{line_index}"]) - chip_offset
            prefix = known_prefix(physical, chip_offset, prefix_len)
            predicted = t0 + chip_offset / rate
            report, margins, data = header_copy(
                amp, predicted, rate, prefix, expected_line,
                header_pair_repetitions, line_index + 1, header_line_count,
                int(truth["header_training_chips"]))
            report["pass"] = pass_index
            report["line_index"] = line_index + 1
            header_reports[line_index].append(report)
            header_margins[line_index].append(margins)
            if report["received_crc32_valid"]:
                header_candidates[line_index].append((pass_index, data))
    selected_lines: list[bytes] = []
    selected_line_sources: list[str | None] = []
    for line_index in range(header_line_count):
        if header_candidates[line_index]:
            pass_index, data = header_candidates[line_index][0]
            selected_lines.append(data)
            selected_line_sources.append(f"pass_{pass_index}")
            continue
        margins = np.sum(header_margins[line_index], axis=0)
        data = np.packbits((margins > 0).astype(np.uint8), bitorder="big").tobytes()
        parsed = parse_received_header_line(data)
        if parsed is not None and parsed[0] == line_index + 1 and parsed[1] == header_line_count:
            selected_lines.append(data)
            selected_line_sources.append("soft_sum")
        else:
            selected_line_sources.append(None)
    params = (assemble_received_header(selected_lines)
              if len(selected_lines) == header_line_count else None)
    if params is not None and params.body_after_header_passes != header_passes:
        params = None
    body_sync_header_matches_private = bool(
        params is not None and
        params.body_preamble_pairs == int(truth["body_preamble_pairs"]) and
        params.body_sync_hex == truth["body_sync_hex"] and
        params.body_sync_chips == truth["body_sync_chips"] and
        params.body_guard_chips == int(truth["body_interblock_gap_chips"])
    )
    all_lines_individual = all(bool(candidates) for candidates in header_candidates)
    complete_passes = sum(
        all(header_reports[line_index][pass_index]["received_crc32_valid"]
            for line_index in range(header_line_count))
        for pass_index in range(header_passes)
    )
    base_report = {
        "capture_file": filename,
        "capture_iq_sha256": capture_sha,
        "capture_sha256_matches_metadata": sha_ok,
        "capture_seconds": round(sample_count / fs, 6),
        "sample_rate_hz": fs,
        "center_frequency_hz": center,
        "measured_rf_peak_hz": round(rf_hz, 2),
        "rf_peak_contrast_db": round(rf_contrast, 2),
        "packet_start_seconds": round(t0, 6),
        "packet_start_correlation": round(start_quality, 4),
        "header_passes": header_passes,
        "header_line_count": header_line_count,
        "header_line_copies": header_reports,
        "individual_received_header_crc_valid_counts_by_line": [
            len(candidates) for candidates in header_candidates],
        "complete_individual_header_passes_crc_valid": complete_passes,
        "selected_header_line_sources": selected_line_sources,
        "all_header_lines_individually_crc_valid": all_lines_individual,
        "header_parameter_source": "received_crc_valid_numbered_lines" if params is not None else None,
        "body_sync_declared_in_received_header": params is not None,
        "body_sync_header_matches_frozen_packet": body_sync_header_matches_private,
        "body_blocks": [],
        "overall_sha256_matches_received_header": None,
        "overall_sha256_matches_private": False,
        "exact_private_message": False,
        "data_segment_useful_bps": None,
        "end_to_end_useful_bps": None,
        "full_payload_valid": False,
    }
    if params is None or not body_sync_header_matches_private or not validate_k7_convention():
        return base_report
    block_count = len(params.payload_bytes)
    if block_count != len(truth["body_block_lengths_bytes"]):
        raise ValueError("received header block count disagrees with frozen packet")
    padded_private = private_message.ljust(sum(params.payload_bytes), b"\0")
    private_parts = [padded_private[sum(params.payload_bytes[:i]):sum(params.payload_bytes[:i + 1])]
                     for i in range(block_count)]
    recovered_parts: list[bytes] = []
    body_starts: list[float] = []
    body_ends: list[float] = []
    for index, (coded_bits, stride, payload_bytes) in enumerate(zip(
        params.coded_bits, params.interleaver_strides, params.payload_bytes
    )):
        chip_offset = int(offsets[f"body_{index}"])
        prefix_len = int(offsets[f"body_data_{index}"]) - chip_offset
        prefix = known_prefix(physical, chip_offset, prefix_len)
        predicted = t0 + chip_offset / rate
        start, quality = rf.align_stage(amp, predicted, rate, prefix)
        local_rate, rate_score = rf.fit_body_rate(amp, start, rate, prefix_len, coded_bits)
        expected_chips = physical[chip_offset + prefix_len:chip_offset + prefix_len + 2*coded_bits]
        if len(expected_chips) != 2*coded_bits:
            raise ValueError("private body timing slot is incomplete")
        expected_tx = expected_chips.reshape(-1, 2)[:, 1]
        attempts: list[tuple[dict, bytes, float]] = []
        ambiguity_candidates = (0.0, -2.0, 2.0, -4.0, 4.0,
                                -6.0, 6.0, -8.0, 8.0,
                                -10.0, 10.0, -12.0, 12.0,
                                -14.0, 14.0, -16.0, 16.0)
        for mode in ("fixed", "tracked"):
            for ambiguity_chips in ambiguity_candidates:
                trial_start = start + ambiguity_chips / local_rate
                attempt_report, attempt_payload = body_one_mode(
                    amp, trial_start, local_rate, prefix_len, coded_bits,
                    stride, payload_bytes, params.block_crc32[index],
                    expected_tx, private_parts[index], mode)
                attempts.append((attempt_report, attempt_payload, ambiguity_chips))
                if (attempt_report["crc32_valid"] and
                        attempt_report["crc32_matches_received_header"] is not False):
                    break
            if (attempts[-1][0]["crc32_valid"] and
                    attempts[-1][0]["crc32_matches_received_header"] is not False):
                break
        chosen_report, chosen_payload, selected_shift = next(
            (item for item in attempts if item[0]["crc32_valid"] and
             item[0]["crc32_matches_received_header"] is not False), attempts[-1]
        )
        selected_start = start + selected_shift / local_rate
        chosen_report.update({
            "block_index": index,
            "start_seconds": round(selected_start, 6),
            "preamble_quality_initial": round(quality, 4),
            "preamble_quality_selected": round(
                rf.known_prefix_quality(amp, selected_start, local_rate, prefix), 4),
            "selected_integer_shift_chips": selected_shift,
            "demod_chip_rate": round(local_rate, 5),
            "rate_fit_score": round(rate_score, 5),
            "coded_bits": coded_bits,
            "message_bytes": payload_bytes,
        })
        base_report["body_blocks"].append(chosen_report)
        recovered_parts.append(chosen_payload)
        body_starts.append(selected_start)
        body_ends.append(selected_start + (prefix_len + 2*coded_bits) / local_rate)
    recovered_padded = b"".join(recovered_parts)
    recovered = recovered_padded[:params.source_length_bytes]
    overall_rf_sha = overall_sha256_valid(recovered_parts, params)
    base_report["overall_sha256_matches_received_header"] = overall_rf_sha
    all_block_crc = all(item["crc32_valid"] and item["crc32_matches_received_header"] is not False
                        for item in base_report["body_blocks"])
    if all_block_crc and overall_rf_sha is True:
        base_report["overall_sha256_matches_private"] = (
            hashlib.sha256(recovered).digest() == hashlib.sha256(private_message).digest())
        base_report["exact_private_message"] = recovered == private_message
    message_bits = params.source_length_bytes * 8
    body_span = body_ends[-1] - body_starts[0]
    beacon_offset = int(offsets["beacon_end"])
    beacon_length = len(truth["beacon_bits"]) * int(truth["beacon_symbol_chips"])
    end_prefix = known_prefix(physical, beacon_offset, beacon_length)
    end_start, end_quality = rf.align_stage(
        amp, t0 + beacon_offset / rate, rate, end_prefix, search_chips=150.0)
    effective_rate = beacon_offset / (end_start - t0)
    packet_span = len(physical) / effective_rate
    base_report["end_beacon_start_seconds"] = round(end_start, 6)
    base_report["end_beacon_alignment_quality"] = round(end_quality, 4)
    base_report["effective_chip_rate_start_to_end"] = round(effective_rate, 5)
    base_report["data_segment_useful_bps"] = round(message_bits / body_span, 2)
    base_report["end_to_end_useful_bps"] = round(message_bits / packet_span, 2)
    base_report["full_payload_valid"] = bool(
        sha_ok and params is not None and body_sync_header_matches_private and
        all_lines_individual and all_block_crc and
        overall_rf_sha is True and base_report["overall_sha256_matches_private"] and
        base_report["exact_private_message"]
    )
    return base_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path,
                        default=ROOT / "private_candidate" / "capture_public")
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--file", default="range_01.iq")
    parser.add_argument("--report-file", type=Path,
                        default=ROOT / "private_candidate" / "oracle_report.json",
                        help="sanitized report destination; use a distinct path for derived IQ")
    args = parser.parse_args()
    try:
        report = run(args.public_dir, args.file, args.private_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(json.dumps({"ready": False, "reason": str(exc)}))
        sys.exit(2)
    report_path = args.report_file
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path),
                      "individual_header_crc_valid_counts_by_line":
                          report["individual_received_header_crc_valid_counts_by_line"],
                      "body_crc_valid": [block["crc32_valid"] for block in report["body_blocks"]],
                      "full_payload_valid": report["full_payload_valid"]}))


if __name__ == "__main__":
    main()
