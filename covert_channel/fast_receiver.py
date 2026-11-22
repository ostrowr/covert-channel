"""Sample-clocked IQ energy receiver for short OOK RF experiments.

Input files contain interleaved little-endian signed 16-bit I,Q samples with
no WebSocket payload headers. ``EnergyReceiver.feed_ws_frame`` accepts the
SDRconnect binary WebSocket message (2-byte payload type + IQ samples).
The small carry buffer keeps the 0.25 ms analysis grid continuous across
SDRconnect's ~50 ms WebSocket messages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DecodeResult:
    start_window: int
    threshold_db: float
    preamble_gap_db: float
    preamble_score_db: float
    payload_bits: str
    bit_errors: int | None
    symbol_scores_db: np.ndarray


@dataclass(frozen=True)
class ManchesterDecodeResult:
    """Logical Manchester decisions; positive soft score means logical zero."""

    start_window: float
    windows_per_chip: float
    chip_average_windows: int
    preamble_score_db: float
    preamble_margin_db: float
    preamble_bit_errors: int
    logical_bits: str
    bit_errors: int | None
    soft_scores_db: np.ndarray
    chip_scores_db: np.ndarray
    timing_phase_windows: np.ndarray | None = None


class EnergyReceiver:
    """Convert a continuous IQ stream to narrowband energy per time window."""

    def __init__(
        self,
        *,
        sample_rate: int,
        center_hz: float,
        carrier_hz: float,
        window_samples: int = 500,
        fft_size: int = 1024,
        half_bandwidth_hz: float = 15_000,
        reference_offset_hz: float = 65_000,
        hop_samples: int | None = None,
        energy_mode: str = "contrast",
    ) -> None:
        if fft_size < window_samples or window_samples <= 0:
            raise ValueError("fft_size must be >= positive window_samples")
        if hop_samples is None:
            hop_samples = window_samples
        if not 1 <= hop_samples <= window_samples:
            raise ValueError("hop_samples must be between 1 and window_samples")
        if energy_mode not in ("contrast", "signal"):
            raise ValueError("energy_mode must be contrast or signal")
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.energy_mode = energy_mode
        self.fft_size = fft_size
        self.window = np.hanning(window_samples).astype(np.float32)
        self.leftover = np.empty(0, dtype=np.complex64)
        self.windows_processed = 0
        offset = carrier_hz - center_hz
        frequencies = np.fft.fftfreq(fft_size, 1 / sample_rate)
        self.signal_bins = np.abs(frequencies - offset) <= half_bandwidth_hz
        self.reference_bins = (
            (np.abs(frequencies - (offset + reference_offset_hz)) <= half_bandwidth_hz)
            | (np.abs(frequencies - (offset - reference_offset_hz)) <= half_bandwidth_hz)
        )
        if not self.signal_bins.any() or not self.reference_bins.any():
            raise ValueError("carrier or reference band outside FFT passband")

    def feed_iq_i16(self, iq_interleaved: bytes | memoryview) -> np.ndarray:
        scalars = np.frombuffer(iq_interleaved, dtype="<i2")
        if scalars.size % 2:
            raise ValueError("IQ data must contain an even number of int16 scalars")
        iq = scalars.reshape(-1, 2)
        z = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
        if self.leftover.size:
            z = np.concatenate((self.leftover, z))
        n = max(0, 1 + (z.size - self.window_samples) // self.hop_samples)
        if n == 0:
            self.leftover = z
            return np.empty(0, dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(z, self.window_samples)[:: self.hop_samples][:n]
        self.leftover = z[n * self.hop_samples :]
        spectrum = np.fft.fft(windows * self.window, n=self.fft_size, axis=1)
        power = np.abs(spectrum) ** 2
        signal = power[:, self.signal_bins].mean(axis=1)
        reference = power[:, self.reference_bins].mean(axis=1)
        self.windows_processed += n
        if self.energy_mode == "signal":
            return (10 * np.log10(signal + 1e-12)).astype(np.float32)
        return (10 * np.log10((signal + 1e-12) / (reference + 1e-12))).astype(np.float32)

    def feed_ws_frame(self, frame: bytes) -> np.ndarray:
        if len(frame) < 2 or int.from_bytes(frame[:2], "little") != 2:
            return np.empty(0, dtype=np.float32)
        return self.feed_iq_i16(memoryview(frame)[2:])


def decode_packet(
    energy_db: np.ndarray,
    *,
    sample_rate: int,
    window_samples: int,
    bitrate: int,
    preamble_bits: str,
    payload_bits_count: int,
    expected_bits: str | None = None,
) -> DecodeResult:
    if set(preamble_bits) - {"0", "1"}:
        raise ValueError("preamble must contain only 0 and 1")
    windows_per_symbol = sample_rate / (window_samples * bitrate)
    if not windows_per_symbol.is_integer():
        raise ValueError("bitrate must give an integer number of analysis windows per symbol")
    width = int(windows_per_symbol)
    if width < 1:
        raise ValueError("analysis windows are too long for bitrate")
    p = len(preamble_bits)
    total_symbols = p + payload_bits_count
    max_start = len(energy_db) - total_symbols * width
    if max_start < 0:
        raise ValueError("capture is too short for packet")

    # Search timing against only the known preamble. Payload bits are hidden
    # from synchronization and threshold selection to keep BER meaningful.
    template = np.fromiter((1 if b == "1" else -1 for b in preamble_bits), dtype=np.float32)
    best_start = 0
    best_score = -np.inf
    for start in range(max_start + 1):
        symbols = energy_db[start : start + p * width].reshape(p, width).mean(axis=1)
        score = float(np.dot(symbols, template) / p)
        if score > best_score:
            best_score = score
            best_start = start

    symbols = energy_db[best_start : best_start + total_symbols * width].reshape(total_symbols, width).mean(axis=1)
    pre = symbols[:p]
    on = pre[template > 0]
    off = pre[template < 0]
    # Off-symbol RF energy has a long positive tail: a few quiet bits after a
    # busy memory interval can look much brighter than the median off bit.
    # Set the boundary using the upper off tail and the stable lower quartile
    # of on symbols. Both estimates come only from the known preamble.
    upper_off = float(np.quantile(off, 0.99))
    lower_on = float(np.quantile(on, 0.25))
    if upper_off < lower_on:
        threshold = (upper_off + lower_on) / 2
    else:
        # At rates where the preamble classes overlap heavily, the tail is no
        # longer a useful boundary. Retain the conventional median boundary.
        threshold = float((np.median(on) + np.median(off)) / 2)
    gap = float(np.min(on) - np.max(off))
    bits = "".join("1" if e > threshold else "0" for e in symbols[p:])
    errors = None
    if expected_bits is not None:
        if len(expected_bits) != payload_bits_count:
            raise ValueError("expected_bits length does not match payload_bits_count")
        errors = sum(a != b for a, b in zip(bits, expected_bits))
    return DecodeResult(best_start, threshold, gap, best_score, bits, errors, symbols)


def _tracked_payload_chips(
    averaged: np.ndarray,
    *,
    start_window: float,
    windows_per_chip: float,
    preamble_chip_count: int,
    payload_logical_bits_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Track Manchester phase using unsigned payload contrast, never bit values.

    Each 256-bit chunk scores phases within two analysis hops of the public
    preamble timing. A small continuity penalty prevents a noisy chunk from
    choosing an isolated phase jump. This is experimental and adds no pilots.
    """
    chunk_bits = 256
    phase_options = np.arange(-8, 9, dtype=np.float64) / 4
    phase_penalty_db = 0.05
    chunks = [(first, min(first + chunk_bits, payload_logical_bits_count))
              for first in range(0, payload_logical_bits_count, chunk_bits)]
    if not chunks:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    grid = np.arange(len(averaged), dtype=np.float64)
    contrast = np.full((len(chunks), len(phase_options)), -np.inf, dtype=np.float64)
    chip_positions = []
    for row, (first, stop) in enumerate(chunks):
        logical = np.arange(first, stop, dtype=np.float64)
        chips = preamble_chip_count + 2 * logical[:, None] + np.arange(2)[None, :]
        positions = start_window + chips * windows_per_chip
        chip_positions.append(positions)
        alternatives = positions[None, :, :] + phase_options[:, None, None]
        valid = (alternatives[:, 0, 0] >= 0) & (alternatives[:, -1, -1] <= len(averaged) - 1)
        if not np.any(valid):
            raise ValueError("capture has no room for tracked Manchester sampling")
        sampled = np.interp(alternatives[valid], grid, averaged)
        contrast[row, valid] = np.mean(np.abs(sampled[:, :, 1] - sampled[:, :, 0]), axis=1)

    metric = contrast[0] - phase_penalty_db * np.abs(phase_options)
    previous = np.zeros(contrast.shape, dtype=np.int16)
    for row in range(1, len(chunks)):
        transition = metric[:, None] - phase_penalty_db * np.abs(
            phase_options[:, None] - phase_options[None, :])
        previous[row] = np.argmax(transition, axis=0)
        metric = contrast[row] + transition[previous[row], np.arange(len(phase_options))]
    selected = np.empty(len(chunks), dtype=np.int16)
    selected[-1] = int(np.argmax(metric))
    for row in range(len(chunks) - 1, 0, -1):
        selected[row - 1] = previous[row, selected[row]]
    phases = phase_options[selected]
    chip_scores = np.concatenate([
        np.interp(positions + phase, grid, averaged).ravel()
        for positions, phase in zip(chip_positions, phases)
    ])
    return chip_scores, phases


def decode_manchester(
    energy_db: np.ndarray,
    *,
    sample_rate: int,
    window_samples: int,
    hop_samples: int,
    physical_rate: int,
    preamble_bits: str,
    payload_logical_bits_count: int,
    expected_logical_bits: str | None = None,
    chip_average_windows: int | None = None,
    timing_track: bool = False,
) -> ManchesterDecodeResult:
    """Synchronize and decode a Manchester packet from overlapped energy windows.

    ``preamble_bits`` holds *physical* chips, encoded 0 -> 01 and 1 -> 10.
    Synchronization uses only that known preamble. Optional timing tracking
    selects a smooth phase from unsigned Manchester contrast in payload chunks,
    without using the payload values or ``expected_logical_bits``. Each logical
    payload bit is decided by comparing energy in its two chips, without an
    absolute-power threshold. A positive soft score in dB denotes logical zero.
    """
    if not preamble_bits or len(preamble_bits) % 2 or set(preamble_bits) - {"0", "1"}:
        raise ValueError("preamble_bits must be a nonempty even-length bit string")
    if any(a == b for a, b in zip(preamble_bits[::2], preamble_bits[1::2])):
        raise ValueError("Manchester preamble chips must be complementary pairs")
    if payload_logical_bits_count < 0:
        raise ValueError("payload_logical_bits_count must be nonnegative")
    if expected_logical_bits is not None and len(expected_logical_bits) != payload_logical_bits_count:
        raise ValueError("expected_logical_bits length does not match payload count")
    if sample_rate <= 0 or physical_rate <= 0 or hop_samples <= 0 or window_samples <= 0:
        raise ValueError("sample rate, physical rate, and window sizes must be positive")
    chip_samples = sample_rate / physical_rate
    width = chip_samples / hop_samples
    if window_samples > chip_samples:
        raise ValueError("analysis FFT window must not exceed one physical chip")
    if chip_average_windows is None:
        # Number of FFT windows that can fit completely inside one chip. The
        # preamble search determines their common phase on the analysis grid.
        chip_average_windows = min(int(width), 1 + round((chip_samples - window_samples) / hop_samples))
    if not 1 <= chip_average_windows <= int(width):
        raise ValueError("chip_average_windows must be between 1 and windows_per_chip")

    energy = np.asarray(energy_db, dtype=np.float64)
    mean_width = chip_average_windows
    cumulative = np.cumsum(np.r_[0.0, energy])
    averaged = (cumulative[mean_width:] - cumulative[:-mean_width]) / mean_width
    physical_count = len(preamble_bits) + 2 * payload_logical_bits_count
    template = np.fromiter((1 if b == "1" else -1 for b in preamble_bits), dtype=np.float64)
    best_score = -np.inf
    best_start: float = 0
    if width.is_integer():
        integer_width = int(width)
        max_start = len(averaged) - 1 - (physical_count - 1) * integer_width
        if max_start < 0:
            raise ValueError("capture is too short for Manchester packet")
        for phase in range(integer_width):
            number_of_starts = (max_start - phase) // integer_width + 1
            if number_of_starts <= 0:
                continue
            # Searching one chip-spaced substream per phase avoids constructing
            # a huge dense time-by-symbol matrix for long IQ captures.
            scores = np.correlate(averaged[phase::integer_width], template, mode="valid")
            candidate = int(np.argmax(scores[:number_of_starts]))
            score = float(scores[candidate])
            if score > best_score:
                best_score = score
                best_start = phase + candidate * integer_width
        chip_scores = averaged[int(best_start) + np.arange(physical_count) * integer_width].astype(np.float32)
    else:
        # At rates such as 5120 chips/s the chip lasts 390.625 IQ samples, so
        # its position advances by a fractional number of FFT hops. Interpolate
        # only the known preamble while searching, in bounded-size batches.
        max_start = int(np.floor(len(averaged) - 2 - (physical_count - 1) * width))
        if max_start < 0:
            raise ValueError("capture is too short for Manchester packet")
        offsets = np.arange(len(template), dtype=np.float64) * width

        def correlation(starts: np.ndarray) -> np.ndarray:
            positions = starts[:, None] + offsets[None, :]
            lower = np.floor(positions).astype(np.int64)
            fraction = positions - lower
            values = averaged[lower] * (1 - fraction) + averaged[lower + 1] * fraction
            return values @ template

        for first in range(0, max_start + 1, 512):
            starts = np.arange(first, min(first + 512, max_start + 1), dtype=np.float64)
            scores = correlation(starts)
            candidate = int(np.argmax(scores))
            if scores[candidate] > best_score:
                best_score = float(scores[candidate])
                best_start = float(starts[candidate])
        fine_starts = best_start + np.arange(-4, 5) * 0.25
        fine_starts = fine_starts[(fine_starts >= 0) & (fine_starts <= max_start)]
        fine_scores = correlation(fine_starts)
        candidate = int(np.argmax(fine_scores))
        best_score = float(fine_scores[candidate])
        best_start = float(fine_starts[candidate])
        positions = best_start + np.arange(physical_count) * width
        chip_scores = np.interp(positions, np.arange(len(averaged)), averaged).astype(np.float32)
    timing_phases = None
    if timing_track and payload_logical_bits_count:
        tracked_chips, timing_phases = _tracked_payload_chips(
            averaged,
            start_window=best_start,
            windows_per_chip=width,
            preamble_chip_count=len(preamble_bits),
            payload_logical_bits_count=payload_logical_bits_count,
        )
        # Keep the public-preamble samples fixed; only payload sampling moves.
        chip_scores = np.concatenate((chip_scores[:len(preamble_bits)].astype(np.float64), tracked_chips))
    pairs = chip_scores.reshape(-1, 2)
    soft = pairs[:, 1] - pairs[:, 0]
    preamble_logical = preamble_bits[::2]
    pre_soft = soft[:len(preamble_logical)]
    pre_sign = np.fromiter((1 if b == "0" else -1 for b in preamble_logical), dtype=np.float32)
    margins = pre_soft * pre_sign
    payload_soft = soft[len(preamble_logical):].copy()
    bits = "".join("0" if value > 0 else "1" for value in payload_soft)
    errors = None
    if expected_logical_bits is not None:
        errors = sum(a != b for a, b in zip(bits, expected_logical_bits))
    return ManchesterDecodeResult(
        start_window=best_start,
        windows_per_chip=width,
        chip_average_windows=chip_average_windows,
        preamble_score_db=best_score / len(preamble_bits),
        preamble_margin_db=float(margins.mean()),
        preamble_bit_errors=int(np.count_nonzero(margins <= 0)),
        logical_bits=bits,
        bit_errors=errors,
        soft_scores_db=payload_soft,
        chip_scores_db=chip_scores,
        timing_phase_windows=timing_phases,
    )


def energies_from_file(path: Path, receiver: EnergyReceiver, chunk_bytes: int = 400_032) -> np.ndarray:
    parts: list[np.ndarray] = []
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            parts.append(receiver.feed_iq_i16(block))
    if not parts:
        raise ValueError("empty IQ file")
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iq_file", type=Path)
    parser.add_argument("--center-hz", type=float, required=True)
    parser.add_argument("--carrier-hz", type=float, required=True)
    parser.add_argument("--sample-rate", type=int, default=2_000_000)
    parser.add_argument("--window-samples", type=int, default=500)
    parser.add_argument("--fft-size", type=int, default=1024)
    parser.add_argument("--bitrate", type=int, default=1000)
    parser.add_argument("--preamble", default="111100001010101011100010")
    parser.add_argument("--payload-bits", type=int, default=64)
    parser.add_argument("--expected-bits")
    args = parser.parse_args()
    receiver = EnergyReceiver(
        sample_rate=args.sample_rate,
        center_hz=args.center_hz,
        carrier_hz=args.carrier_hz,
        window_samples=args.window_samples,
        fft_size=args.fft_size,
    )
    energies = energies_from_file(args.iq_file, receiver)
    result = decode_packet(
        energies,
        sample_rate=args.sample_rate,
        window_samples=args.window_samples,
        bitrate=args.bitrate,
        preamble_bits=args.preamble,
        payload_bits_count=args.payload_bits,
        expected_bits=args.expected_bits,
    )
    print(f"windows={len(energies)} start={result.start_window} preamble_gap_db={result.preamble_gap_db:.2f}")
    print(f"threshold_db={result.threshold_db:.2f} payload={result.payload_bits}")
    if result.bit_errors is not None:
        print(f"BER={result.bit_errors}/{args.payload_bits}")


if __name__ == "__main__":
    main()
