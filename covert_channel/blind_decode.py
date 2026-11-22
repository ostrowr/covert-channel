"""Decode a fixed RF text packet from SDR IQ without sender text or logs.

Only the public wire format and the IQ file are inputs. The 112-bit logical
preamble is fixed by fast_sender.c; payload bits never select timing, harmonic
weight, or FEC parameters. A decoded message is returned only after all four
LDPC blocks pass parity and CRC and the length-prefixed UTF-8 frame is valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .fast_receiver import EnergyReceiver, decode_manchester, energies_from_file
from .fec import CODEWORD_BITS, PAYLOAD_BITS, decode
from .harmonic_receiver import combine_preamble_weighted
from .interleave import deinterleave


SAMPLE_RATE = 2_000_000
CENTER_HZ = 1_400_000
HIGH_CARRIER_HZ = 1_505_000
WINDOW_SAMPLES = 250
HOP_SAMPLES = 50
FFT_SIZE = 1024
BLOCKS = 4
TEXT_CAPACITY = BLOCKS * PAYLOAD_BITS // 8 - 4
PROFILES = {"baseline6250": 6250, "fast6500": 6500}


def known_preamble_chips() -> str:
    logical = ([0] * 16 + [1] * 16 +
               [1 if index % 2 == 0 else 0 for index in range(64)] +
               [(0xD391 >> shift) & 1 for shift in range(15, -1, -1)])
    assert len(logical) == 112
    return "".join(f"{bit}{1 - bit}" for bit in logical)


def unframe(blocks: list[np.ndarray]) -> tuple[bytes, str]:
    frame = b"".join(np.packbits(block).tobytes() for block in blocks)
    count = int.from_bytes(frame[:4], "big")
    if count > TEXT_CAPACITY:
        raise ValueError(f"decoded length {count} exceeds {TEXT_CAPACITY}")
    raw = frame[4:4 + count]
    return raw, raw.decode("utf-8", errors="strict")


def blind_decode(iq_path: Path, profile: str, *, include_text: bool = False) -> dict:
    if profile not in PROFILES:
        raise ValueError("Unknown RF profile")
    rate = PROFILES[profile]
    preamble = known_preamble_chips()
    payload_bits = BLOCKS * CODEWORD_BITS
    # The 1.505 MHz family can have a noisy upper sideband at a different loop
    # placement. Keep the established receiver first, then try a fixed lower
    # passband. Selection requires four independent parity+CRC-valid blocks;
    # the sender text and any capture-side truth are never inputs here.
    candidates = [(HIGH_CARRIER_HZ, 30_000 if rate >= 6_400 else 15_000,
                   profile == "fast6500")]
    if profile == "fast6500":
        candidates.append((1_495_000, 30_000, False))

    best: dict | None = None
    for carrier_hz, half_bandwidth_hz, use_harmonic in candidates:
        receiver = EnergyReceiver(
            sample_rate=SAMPLE_RATE,
            center_hz=CENTER_HZ,
            carrier_hz=carrier_hz,
            window_samples=WINDOW_SAMPLES,
            hop_samples=HOP_SAMPLES,
            fft_size=FFT_SIZE,
            half_bandwidth_hz=half_bandwidth_hz,
            energy_mode="signal",
        )
        energy = energies_from_file(iq_path, receiver)
        harmonic = None
        if use_harmonic:
            energy, harmonic = combine_preamble_weighted(
                iq_path,
                energy,
                sample_rate=SAMPLE_RATE,
                center_hz=CENTER_HZ,
                window_samples=WINDOW_SAMPLES,
                hop_samples=HOP_SAMPLES,
                fft_size=FFT_SIZE,
                physical_rate=rate,
                preamble_bits=preamble,
                payload_logical_bits_count=payload_bits,
            )
        demod = decode_manchester(
            energy,
            sample_rate=SAMPLE_RATE,
            window_samples=WINDOW_SAMPLES,
            hop_samples=HOP_SAMPLES,
            physical_rate=rate,
            preamble_bits=preamble,
            payload_logical_bits_count=payload_bits,
        )
        scores = deinterleave(demod.soft_scores_db, scramble=True)
        recovered: list[np.ndarray] = []
        block_status = []
        for index in range(BLOCKS):
            block = decode(scores[index * CODEWORD_BITS:(index + 1) * CODEWORD_BITS])
            valid = bool(block.data is not None and block.syndrome_ok and block.crc_ok)
            block_status.append({"block": index, "parity_ok": bool(block.syndrome_ok),
                                 "crc_ok": bool(block.crc_ok), "list_repaired": bool(block.list_repaired)})
            if valid:
                recovered.append(block.data)
        result = {
            "profile": profile,
            "physical_symbols_per_second": rate,
            "receiver_carrier_hz": carrier_hz,
            "receiver_half_bandwidth_hz": half_bandwidth_hz,
            "preamble_start_seconds": demod.start_window * HOP_SAMPLES / SAMPLE_RATE,
            "preamble_errors": demod.preamble_bit_errors,
            "preamble_margin_db": demod.preamble_margin_db,
            "harmonic_low_weight": harmonic["low_weight"] if harmonic else None,
            "fec_blocks_passed": len(recovered),
            "fec_blocks_total": BLOCKS,
            "blocks": block_status,
            "message_valid": False,
        }
        if len(recovered) == BLOCKS:
            try:
                raw, message = unframe(recovered)
            except (ValueError, UnicodeDecodeError) as exc:
                result["frame_error"] = str(exc)
            else:
                result.update({
                    "message_valid": True,
                    "utf8_bytes": len(raw),
                    "message_sha256": hashlib.sha256(raw).hexdigest(),
                    "text_bit_rate_on_air": len(raw) * 8 /
                        (2 * (len(preamble) // 2 + payload_bits) / rate),
                })
                if include_text:
                    result["decoded_message"] = message
                return result
        if best is None or result["fec_blocks_passed"] > best["fec_blocks_passed"]:
            best = result
    assert best is not None
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iq_path", type=Path, help="raw interleaved i16 I/Q samples")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="fast6500")
    parser.add_argument("--include-text", action="store_true", help="include decoded UTF-8 text in JSON")
    args = parser.parse_args()
    result = blind_decode(args.iq_path, args.profile, include_text=args.include_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["message_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
