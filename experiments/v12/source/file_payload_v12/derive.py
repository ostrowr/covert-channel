"""Derive a protocol-agnostic 100 kS/s RF-band IQ view of the selected capture.

This reads only the selected physical IQ and its public metadata. Frequency
selection uses an ordinary short-window power survey; no packet or sender
source is consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent

OUT_RATE_HZ = 100_000
SEARCH_WINDOW_SECONDS = 0.1
SEARCH_STEP_SECONDS = 0.5
SEARCH_NFFT = 16_384
SEARCH_BAND_HZ = (1_490_000.0, 1_520_000.0)
FIR_TAPS = 801
FIR_CUTOFF_HZ = 40_000.0
FIR_WINDOW = ("kaiser", 10.0)
BLOCK_INPUT_SAMPLES = 2_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_complex(raw: np.memmap, first: int, last: int) -> np.ndarray:
    pairs = raw[2 * first:2 * last].reshape(-1, 2)
    return pairs[:, 0].astype(np.float32) + 1j * pairs[:, 1].astype(np.float32)


def choose_center(raw: np.memmap, fs: int, source_center: float) -> tuple[int, float]:
    """Choose the strongest average short-window PSD bin in the predeclared band."""
    n_iq = raw.size // 2
    duration = n_iq / fs
    cumulative: np.ndarray | None = None
    frequencies: np.ndarray | None = None
    windows = 0
    seconds = np.arange(1.0, max(1.0, duration - SEARCH_WINDOW_SECONDS),
                        SEARCH_STEP_SECONDS)
    for second in seconds:
        first = int(second * fs)
        last = first + int(SEARCH_WINDOW_SECONDS * fs)
        if last > n_iq:
            break
        z = read_complex(raw, first, last)
        frequencies, power = signal.welch(
            z, fs=fs, nperseg=SEARCH_NFFT, noverlap=SEARCH_NFFT // 2,
            return_onesided=False,
        )
        if cumulative is None:
            cumulative = np.zeros_like(power, dtype=np.float64)
        cumulative += power
        windows += 1
    if cumulative is None or frequencies is None or windows < 3:
        raise ValueError("source IQ is too short for the generic spectral survey")
    rf_frequencies = source_center + frequencies
    candidates = np.flatnonzero(
        (rf_frequencies >= SEARCH_BAND_HZ[0]) &
        (rf_frequencies <= SEARCH_BAND_HZ[1])
    )
    if not len(candidates):
        raise ValueError("predeclared RF band lies outside the source IQ")
    peak = int(candidates[np.argmax(cumulative[candidates])])
    measured = float(rf_frequencies[peak])
    chosen = int(round(measured / 100.0) * 100)
    return chosen, measured


def filter_characteristics(taps: np.ndarray, fs: int) -> tuple[float, float]:
    frequencies, response = signal.freqz(taps, worN=131_072, fs=fs)
    passband = np.abs(response[frequencies <= 30_000])
    stopband = np.abs(response[frequencies >= OUT_RATE_HZ / 2])
    min_passband_db = float(20 * np.log10(max(float(passband.min()), 1e-30)))
    max_stopband_db = float(20 * np.log10(max(float(stopband.max()), 1e-30)))
    if min_passband_db < -0.5 or max_stopband_db > -80:
        raise ValueError("low-pass design does not meet the conversion limits")
    return min_passband_db, max_stopband_db


def derive(source_public: Path, public: Path) -> dict:
    source_iq = source_public / "range_01.iq"
    source_metadata = source_public / "metadata.json"
    out_iq = public / "range_01.iq"
    out_metadata = public / "metadata.json"
    source = json.loads(source_metadata.read_text())
    entry = next((item for item in source["ranges"]
                  if item.get("file") == source_iq.name), None)
    if entry is None:
        raise ValueError("source metadata has no matching IQ entry")
    fs = int(source["sample_rate_hz"])
    source_center = float(entry["center_frequency_hz"])
    n_iq = int(source["complex_samples_per_range"])
    if fs != 2_000_000 or n_iq % (fs // OUT_RATE_HZ):
        raise ValueError("unexpected source sample geometry")
    if source_iq.stat().st_size != n_iq * 4:
        raise ValueError("source IQ byte count disagrees with metadata")
    original_sha = sha256_file(source_iq)
    source_metadata_sha = sha256_file(source_metadata)
    if original_sha.lower() != str(entry["sha256"]).lower():
        raise ValueError("source IQ SHA-256 disagrees with metadata")
    if out_iq.exists() or out_metadata.exists():
        raise FileExistsError("derived public IQ or metadata already exists")
    raw = np.memmap(source_iq, dtype="<i2", mode="r")
    target_center, measured_peak = choose_center(raw, fs, source_center)
    shift_hz = target_center - source_center
    decimation = fs // OUT_RATE_HZ
    taps = signal.firwin(FIR_TAPS, FIR_CUTOFF_HZ, fs=fs,
                         window=FIR_WINDOW).astype(np.float32)
    min_passband_db, max_stopband_db = filter_characteristics(taps, fs)
    pad = (FIR_TAPS - 1) // 2
    if pad % decimation or BLOCK_INPUT_SAMPLES % decimation:
        raise AssertionError("FIR overlap and chunk length must align to decimation")
    public.mkdir(parents=True, exist_ok=True)
    stage = public / ".range_01.staging.iq"
    output_sha = hashlib.sha256()
    output_samples = 0
    clipped_components = 0
    with stage.open("xb") as stream:
        for first in range(0, n_iq, BLOCK_INPUT_SAMPLES):
            last = min(first + BLOCK_INPUT_SAMPLES, n_iq)
            left = max(0, first - pad)
            right = min(n_iq, last + pad)
            z = read_complex(raw, left, right)
            sample_numbers = np.arange(left, right, dtype=np.float64)
            oscillator = np.exp(
                (-2j * np.pi * shift_hz / fs) * sample_numbers
            ).astype(np.complex64)
            translated = z * oscillator
            down = signal.resample_poly(
                translated, 1, decimation, window=taps, padtype="constant")
            trim = (first - left) // decimation
            take = (last - first) // decimation
            selected = down[trim:trim + take]
            if len(selected) != take:
                raise AssertionError("polyphase output length is incomplete")
            components = np.column_stack((selected.real, selected.imag))
            clipped_components += int(np.count_nonzero(
                (components > 32767) | (components < -32768)))
            quantized = np.rint(np.clip(components, -32768, 32767)).astype("<i2")
            chunk = quantized.tobytes()
            stream.write(chunk)
            output_sha.update(chunk)
            output_samples += len(selected)
    if output_samples != n_iq // decimation or stage.stat().st_size != output_samples * 4:
        raise AssertionError("derived sample count or byte count is incorrect")
    os.link(stage, out_iq)
    stage.unlink()
    metadata = {
        "dataset_kind": "derived_selected_band_physical_iq",
        "sample_rate_hz": OUT_RATE_HZ,
        "iq_format": "interleaved signed 16-bit little-endian I,Q",
        "complex_samples_per_range": output_samples,
        "capture_seconds_per_range": output_samples / OUT_RATE_HZ,
        "ranges": [{
            "file": out_iq.name,
            "center_frequency_hz": target_center,
            "sha256": output_sha.hexdigest(),
        }],
        "derivation": {
            "kind": "filtered and decimated physical IQ; no new SDR capture",
            "source_capture_file": source_iq.name,
            "source_capture_sha256": original_sha,
            "original_capture_sha256": original_sha,
            "source_capture_metadata_sha256": source_metadata_sha,
            "source_sample_rate_hz": fs,
            "source_center_frequency_hz": source_center,
            "source_complex_samples": n_iq,
            "source_iq_format": source["iq_format"],
            "frequency_translation_hz": shift_hz,
            "measured_short_window_peak_hz": measured_peak,
            "center_selection": (
                "largest average Welch PSD bin from 100 ms windows every 500 ms "
                "within the predeclared 1.490-1.520 MHz band validated in a "
                "prior capture; rounded to 100 Hz"
            ),
            "decimation": decimation,
            "filter": {
                "type": "zero-phase polyphase FIR",
                "num_taps": FIR_TAPS,
                "cutoff_hz": FIR_CUTOFF_HZ,
                "window": "Kaiser beta 10",
                "input_overlap_samples_each_side": pad,
                "minimum_gain_db_through_30khz": min_passband_db,
                "maximum_gain_db_at_or_above_output_nyquist": max_stopband_db,
            },
            "output_quantization": "round to nearest int16, saturating",
            "clipped_iq_components": clipped_components,
        },
    }
    with out_metadata.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return {
        "source_sha256_matches_metadata": True,
        "source_complex_samples": n_iq,
        "derived_complex_samples": output_samples,
        "derived_bytes": out_iq.stat().st_size,
        "derived_center_hz": target_center,
        "derived_sample_rate_hz": OUT_RATE_HZ,
        "derived_sha256": output_sha.hexdigest(),
        "clipped_components": clipped_components,
        "max_stopband_gain_db": max_stopband_db,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=ROOT / "private_candidate")
    args = parser.parse_args()
    packet_dir = args.packet_dir.expanduser().resolve()
    print(json.dumps(derive(packet_dir / "capture_public",
                            packet_dir / "derived_public"), indent=2))
