"""Capture and verify a coded MacBook-to-SDRplay near-field RF packet."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import websockets

from .fast_receiver import EnergyReceiver, decode_manchester, energies_from_file
from .fec import CODEWORD_BITS, PAYLOAD_BITS, decode, encode
from .harmonic_receiver import combine_preamble_weighted
from .interleave import deinterleave, interleave


BASE = Path(__file__).resolve().parent
SAMPLE_RATE = 2_000_000
LOGICAL_PREAMBLE_BITS = 112
PHYSICAL_PREAMBLE_BITS = 2 * LOGICAL_PREAMBLE_BITS
MESSAGE_BLOCKS = 4
MESSAGE_CAPACITY_BYTES = MESSAGE_BLOCKS * PAYLOAD_BITS // 8 - 4


def frame_message(message: bytes) -> list[np.ndarray]:
    """Put a UTF-8 message and its length into four 256-byte RF payloads."""
    if len(message) > MESSAGE_CAPACITY_BYTES:
        raise ValueError(f"message exceeds {MESSAGE_CAPACITY_BYTES} UTF-8 bytes")
    message.decode("utf-8", errors="strict")
    frame = len(message).to_bytes(4, "big") + message
    frame = frame.ljust(MESSAGE_BLOCKS * PAYLOAD_BITS // 8, b"\0")
    return [np.unpackbits(np.frombuffer(frame[i:i + PAYLOAD_BITS // 8], dtype=np.uint8))
            for i in range(0, len(frame), PAYLOAD_BITS // 8)]


def unframe_message(decoded: list[np.ndarray]) -> str:
    """Read only the receiver's decoded payload bits, never the source text."""
    frame = b"".join(np.packbits(block).tobytes() for block in decoded)
    size = int.from_bytes(frame[:4], "big")
    if size > MESSAGE_CAPACITY_BYTES:
        raise ValueError("decoded message length is out of range")
    return frame[4:4 + size].decode("utf-8", errors="strict")


async def send(ws, event: str, prop: str = "", value: object = "") -> None:
    await ws.send(json.dumps({"event_type": event, "property": prop, "value": str(value)}))


async def discard_for(ws, seconds: float) -> None:
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        try:
            await asyncio.wait_for(ws.recv(), max(0.001, until - time.monotonic()))
        except asyncio.TimeoutError:
            break


def read_log(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


async def capture(args, directory: Path, codeword_bits: str) -> tuple[Path, dict[str, str], int]:
    iq_path = directory / "samples.iq"
    bit_path = directory / "payload.bits"
    sender_log = directory / "sender.log"
    sender_stdout = directory / "sender.stdout"
    bit_path.write_text(codeword_bits + "\n")
    air_seconds = 2 * (LOGICAL_PREAMBLE_BITS + len(codeword_bits)) / args.rate

    async with websockets.connect("ws://127.0.0.1:9876", max_size=None, max_queue=None) as ws:
        await send(ws, "device_stream_enable", value="true")
        await discard_for(ws, 0.8)
        await send(ws, "set_property", "device_sample_rate", SAMPLE_RATE)
        await send(ws, "set_property", "device_center_frequency", args.center)
        await send(ws, "set_property", "lna_state", 0)
        await discard_for(ws, 0.7)
        await send(ws, "iq_stream_enable", value="true")
        await discard_for(ws, 0.5)
        with sender_stdout.open("w") as stdout, iq_path.open("wb") as iq_file:
            command = [
                str(BASE / "fast_sender"),
                "--rate", str(args.rate),
                "--payload-file", str(bit_path),
                "--workers", str(args.workers),
                "--buffer-mib", str(args.buffer_mib),
                "--lead-ms", "1000",
                "--mode", args.sender_mode,
                "--clock", "busy",
                "--worker-qos", args.worker_qos,
                "--clock-qos", args.clock_qos,
                "--encoding", "manchester",
                "--log", str(sender_log),
            ]
            process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT)
            frames = 0
            done_at = None
            started_at = time.monotonic()
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), 0.1)
                        if isinstance(message, bytes) and int.from_bytes(message[:2], "little") == 2:
                            iq_file.write(message[2:])
                            frames += 1
                    except asyncio.TimeoutError:
                        pass
                    if process.poll() is not None and done_at is None:
                        done_at = time.monotonic()
                    if done_at is not None and time.monotonic() - done_at >= 0.5:
                        break
                    if time.monotonic() - started_at > air_seconds + 8:
                        raise TimeoutError("sender exceeded expected packet duration")
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)
        await send(ws, "iq_stream_enable", value="false")
        await send(ws, "device_stream_enable", value="false")
        await discard_for(ws, 0.2)

    if process.returncode:
        raise RuntimeError(f"sender failed: {sender_stdout.read_text()}")
    return iq_path, read_log(sender_log), frames


def run(args) -> dict:
    if args.blocks < 1 or args.workers < 1 or args.rate < 1 or args.rate > 8_000:
        raise ValueError("blocks/workers/rate must be positive; rate must be at most 8000")
    if args.interleave and args.blocks != 4:
        raise ValueError("the current interleaver requires exactly four blocks")
    message_file = getattr(args, "message_file", None)
    message_bytes = None
    if message_file is not None:
        if args.blocks != MESSAGE_BLOCKS:
            raise ValueError("message mode requires exactly four blocks")
        message_bytes = message_file.read_bytes()
        payloads = frame_message(message_bytes)
    else:
        rng = np.random.default_rng(args.seed)
        payloads = [rng.integers(0, 2, size=PAYLOAD_BITS, dtype=np.uint8) for _ in range(args.blocks)]
    coded = np.concatenate([encode(payload) for payload in payloads])
    transmitted = interleave(coded, scramble=True) if args.interleave else coded
    codeword_bits = "".join(map(str, transmitted))

    with tempfile.TemporaryDirectory(prefix="rf-channel-fast-") as temp:
        iq_path, sender, frames = asyncio.run(capture(args, Path(temp), codeword_bits))
        expected = sender["logical_bits"][LOGICAL_PREAMBLE_BITS:]
        if expected != codeword_bits:
            raise RuntimeError("sender log differs from the requested coded payload")
        receiver_half_bandwidth_hz = 30_000 if args.rate >= 6_400 else 15_000
        receiver = EnergyReceiver(
            sample_rate=SAMPLE_RATE,
            center_hz=args.center,
            carrier_hz=args.carrier,
            window_samples=250,
            hop_samples=50,
            fft_size=1024,
            half_bandwidth_hz=receiver_half_bandwidth_hz,
            energy_mode="signal",
        )
        energy = energies_from_file(iq_path, receiver)
        harmonic_details = None
        if getattr(args, "harmonic_combine", None) == "preamble":
            energy, harmonic_details = combine_preamble_weighted(
                iq_path,
                energy,
                sample_rate=SAMPLE_RATE,
                center_hz=args.center,
                window_samples=250,
                hop_samples=50,
                fft_size=1024,
                physical_rate=args.rate,
                preamble_bits=sender["bits"][:PHYSICAL_PREAMBLE_BITS],
                payload_logical_bits_count=len(coded),
            )
        result = decode_manchester(
            energy,
            sample_rate=SAMPLE_RATE,
            window_samples=250,
            hop_samples=50,
            physical_rate=args.rate,
            preamble_bits=sender["bits"][:PHYSICAL_PREAMBLE_BITS],
            payload_logical_bits_count=len(coded),
            expected_logical_bits=expected,
            timing_track=getattr(args, "timing_track", False),
        )
        if args.iq_out is not None:
            args.iq_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(iq_path, args.iq_out)
            shutil.copyfile(Path(temp) / "sender.log", args.iq_out.with_name(args.iq_out.name + ".sender.log"))

    soft_scores = deinterleave(result.soft_scores_db, scramble=True) if args.interleave else result.soft_scores_db
    block_results = []
    payload_errors = 0
    recovered_blocks = []
    for index, payload in enumerate(payloads):
        start = index * CODEWORD_BITS
        stop = start + CODEWORD_BITS
        scores = soft_scores[start:stop]
        raw_expected = coded[start:stop]
        raw_decisions = (scores < 0).astype(np.uint8)
        decoded = decode(scores)
        if decoded.data is not None:
            recovered_blocks.append(decoded.data)
        data_errors = None if decoded.data is None else int(np.count_nonzero(decoded.data != payload))
        if data_errors is not None:
            payload_errors += data_errors
        block_results.append({
            "index": index,
            "raw_errors": int(np.count_nonzero(raw_decisions != raw_expected)),
            "coded_bits": CODEWORD_BITS,
            "parity_ok": decoded.syndrome_ok,
            "crc_ok": decoded.crc_ok,
            "list_repaired": decoded.list_repaired,
            "data_errors": data_errors,
            "iterations": decoded.iterations,
        })

    air_seconds = 2 * (LOGICAL_PREAMBLE_BITS + args.blocks * CODEWORD_BITS) / args.rate
    all_verified = all(
        item["parity_ok"] and item["crc_ok"] and item["data_errors"] == 0
        for item in block_results
    )
    summary = {
        "all_blocks_verified": all_verified,
        "physical_symbol_rate": args.rate,
        "sender_mode": args.sender_mode,
        "worker_qos_requested": args.worker_qos,
        "clock_qos_requested": args.clock_qos,
        "interleaved": args.interleave,
        "payload_bits": args.blocks * PAYLOAD_BITS,
        "coded_bits": args.blocks * CODEWORD_BITS,
        "preamble_logical_bits": LOGICAL_PREAMBLE_BITS,
        "air_seconds": air_seconds,
        "on_air_payload_bps": args.blocks * PAYLOAD_BITS / air_seconds,
        "raw_errors": sum(item["raw_errors"] for item in block_results),
        "payload_errors": payload_errors if all_verified else None,
        "preamble_errors": result.preamble_bit_errors,
        "preamble_margin_db": result.preamble_margin_db,
        "frames": frames,
        "center_hz": args.center,
        "carrier_hz": args.carrier,
        "sample_rate": SAMPLE_RATE,
        "receiver_half_bandwidth_hz": receiver_half_bandwidth_hz,
        "seed": args.seed,
        "blocks": block_results,
    }
    summary["sender_timing"] = {
        "mean_abs_symbol_late_ns": int(sender["timing_mean_abs_late_ns"]),
        "max_symbol_late_ns": int(sender["timing_max_late_ns"]),
        "worker_mean_transition_delay_ns": [
            int(sender[f"worker_{index}_mean_delay_ns"]) for index in range(args.workers)
        ],
        "worker_max_transition_delay_ns": [
            int(sender[f"worker_{index}_max_delay_ns"]) for index in range(args.workers)
        ],
        "worker_missed_transitions": [
            int(sender[f"worker_{index}_missed"]) for index in range(args.workers)
        ],
        "worker_bytes_touched": int(sender["worker_bytes_touched"]),
        "clock_qos_set_rc": int(sender["clock_qos_set_rc"]),
        "clock_qos_get_rc": int(sender["clock_qos_get_rc"]),
        "clock_qos_actual": sender["clock_qos_actual"],
        "clock_qos_actual_relative": int(sender["clock_qos_actual_relative"]),
        "worker_qos_set_rc": [
            int(sender[f"worker_{index}_qos_set_rc"]) for index in range(args.workers)
        ],
        "worker_qos_get_rc": [
            int(sender[f"worker_{index}_qos_get_rc"]) for index in range(args.workers)
        ],
        "worker_qos_actual": [
            sender[f"worker_{index}_qos_actual"] for index in range(args.workers)
        ],
    }
    if harmonic_details is not None:
        summary["harmonic_combine"] = harmonic_details
    if getattr(args, "timing_track", False):
        summary["timing_track"] = {
            "chunk_logical_bits": 256,
            "phase_offsets_windows": result.timing_phase_windows.tolist()
                if result.timing_phase_windows is not None else [],
        }
    if message_bytes is not None:
        decoded_message = None
        decode_error = None
        if len(recovered_blocks) == MESSAGE_BLOCKS:
            try:
                decoded_message = unframe_message(recovered_blocks)
            except (ValueError, UnicodeDecodeError) as exc:
                decode_error = str(exc)
        summary.update({
            "message_bytes": len(message_bytes),
            "message_capacity_bytes": MESSAGE_CAPACITY_BYTES,
            "decoded_message": decoded_message,
            "message_matches": decoded_message is not None and decoded_message.encode("utf-8") == message_bytes,
            "message_decode_error": decode_error,
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=int, default=6250, help="physical symbols/s (default: 6250)")
    parser.add_argument("--blocks", type=int, default=4, help="2048-bit payload blocks (default: 4)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--buffer-mib", type=int, default=128)
    parser.add_argument("--sender-mode", choices=("park", "spin"), default="park",
                        help="worker off-symbol behavior (default: park)")
    parser.add_argument("--worker-qos", choices=("default", "initiated"), default="default",
                        help="opt-in worker thread QoS (default: inherit existing QoS)")
    parser.add_argument("--clock-qos", choices=("default", "interactive"), default="default",
                        help="opt-in timing thread QoS (default: inherit existing QoS)")
    parser.add_argument("--center", type=int, default=1_400_000)
    parser.add_argument("--carrier", type=int, default=1_505_000)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--result", type=Path, default=BASE / "last_result.json")
    parser.add_argument("--message-file", type=Path,
                        help="UTF-8 text to frame, transmit, and recover through the SDR")
    parser.add_argument("--iq-out", type=Path, help="optionally keep raw IQ and sender log for analysis")
    parser.add_argument("--harmonic-combine", choices=("preamble",),
                        help="experimentally combine 1.505 MHz and 754 kHz envelopes")
    parser.add_argument("--timing-track", action="store_true",
                        help="experimentally track payload phase from unsigned Manchester contrast")
    parser.add_argument("--interleave", action=argparse.BooleanOptionalAction, default=True,
                        help="spread four FEC blocks across the packet (default: enabled)")
    args = parser.parse_args()
    result = run(args)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "verified": result["all_blocks_verified"],
        "payload_bits": result["payload_bits"],
        "interleaved": result["interleaved"],
        "raw_errors": result["raw_errors"],
        "on_air_payload_bps": round(result["on_air_payload_bps"], 2),
        "blocks": result["blocks"],
        "result": str(args.result),
    }, indent=2))
    return 0 if result["all_blocks_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
