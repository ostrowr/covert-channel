"""Small user-facing entrypoint for the bundled IQ decoder."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from covert_channel.blind_decode import blind_decode


def main() -> int:
    parser = argparse.ArgumentParser(prog="covert-channel decode")
    parser.add_argument("iq", type=Path, help="raw 2 MS/s interleaved int16 IQ capture")
    parser.add_argument("--rate", type=int, choices=(6250, 6500), default=6500)
    args = parser.parse_args()
    profile = "fast6500" if args.rate == 6500 else "baseline6250"
    try:
        result = blind_decode(args.iq, profile, include_text=True)
    except (OSError, ValueError) as error:
        print(f"decode failed: {error}", file=sys.stderr)
        return 1
    if not result["message_valid"]:
        print(f"decode failed: {result['fec_blocks_passed']}/4 blocks valid", file=sys.stderr)
        return 1
    print(result["decoded_message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
