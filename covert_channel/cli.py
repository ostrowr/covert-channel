"""Mac memory-workload RF sender and SDR IQ decoder."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import blind_decode, fast_link
from .fec import encode
from .interleave import interleave

ROOT = Path(__file__).resolve().parent
SENDER = ROOT / "fast_sender"


def ensure_sender() -> Path:
    source = ROOT / "fast_sender.c"
    if not source.exists():
        raise FileNotFoundError(f"Missing native sender source: {source}")
    if not SENDER.exists() or source.stat().st_mtime > SENDER.stat().st_mtime:
        subprocess.run(["cc", "-O3", "-std=c11", "-pthread", "-o", str(SENDER), str(source)], check=True)
    return SENDER


def read_message(args: argparse.Namespace) -> bytes:
    if args.text is not None:
        raw = args.text.encode("utf-8")
    else:
        raw = args.file.read_bytes()
        raw.decode("utf-8", errors="strict")
    if not 1 <= len(raw) <= fast_link.MESSAGE_CAPACITY_BYTES:
        raise ValueError(f"Message must contain 1–{fast_link.MESSAGE_CAPACITY_BYTES} UTF-8 bytes")
    return raw


def emit(args: argparse.Namespace) -> int:
    message = read_message(args)
    blocks = fast_link.frame_message(message)
    coded = np.concatenate([encode(block) for block in blocks])
    wire = interleave(coded, scramble=True)
    with tempfile.TemporaryDirectory(prefix="covert-channel-") as directory:
        bit_file = Path(directory) / "payload.bits"
        bit_file.write_text("".join(map(str, wire)) + "\n")
        command = [str(ensure_sender()), "--rate", str(args.rate), "--payload-file", str(bit_file),
                   "--workers", "4", "--buffer-mib", "128", "--lead-ms", "1000",
                   "--mode", "park", "--clock", "busy", "--encoding", "manchester"]
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    air_seconds = 2 * (fast_link.LOGICAL_PREAMBLE_BITS + wire.size) / args.rate
    print(json.dumps({"sent_utf8_bytes": len(message), "rate_symbols_per_second": args.rate,
                      "air_seconds": round(air_seconds, 5),
                      "file_bits_per_second": round(len(message) * 8 / air_seconds, 2)}, indent=2))
    return 0


def decode(args: argparse.Namespace) -> int:
    result = blind_decode.blind_decode(args.iq, args.profile, include_text=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["message_valid"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="covert-channel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    send = commands.add_parser("emit", help="emit a high-rate memory-workload packet from this Mac")
    source = send.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="UTF-8 message")
    source.add_argument("--file", type=Path, help="UTF-8 text file")
    send.add_argument("--rate", type=int, choices=(6250, 6500), default=6500)
    send.set_defaults(action=emit)
    receive = commands.add_parser("decode", help="decode a recorded raw 2 MS/s int16 IQ file")
    receive.add_argument("iq", type=Path)
    receive.add_argument("--profile", choices=tuple(blind_decode.PROFILES), default="fast6500")
    receive.set_defaults(action=decode)
    args = parser.parse_args()
    try:
        return args.action(args)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"covert-channel: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
