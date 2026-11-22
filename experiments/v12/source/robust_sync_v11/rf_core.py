"""Protocol-independent high-band RF front end and Manchester timing helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, signal


OUT_RATE_HZ = 100_000
DECIMATION = 20
SEARCH_BAND_HZ = (1_490_000.0, 1_520_000.0)
CHIP_FRACTIONS = np.array((0.30, 0.40, 0.50, 0.60, 0.70))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_capture(public: Path, filename: str) -> tuple[Path, int, float, int, bool, str]:
    metadata = json.loads((public / "metadata.json").read_text())
    entry = next((item for item in metadata["ranges"] if item["file"] == filename), None)
    if entry is None:
        raise ValueError(f"capture metadata has no {filename} entry")
    iq_path = public / filename
    if not iq_path.is_file():
        raise FileNotFoundError(iq_path)
    sample_rate = int(metadata["sample_rate_hz"])
    sample_count = int(metadata["complex_samples_per_range"])
    if iq_path.stat().st_size != sample_count * 4:
        raise ValueError("IQ byte count disagrees with metadata")
    actual_sha = sha256_file(iq_path).lower()
    return (iq_path, sample_rate, float(entry["center_frequency_hz"]), sample_count,
            actual_sha == str(entry["sha256"]).lower(), actual_sha)


def read_complex(raw: np.memmap, first: int, last: int) -> np.ndarray:
    pairs = raw[2 * first:2 * last].reshape(-1, 2)
    return pairs[:, 0].astype(np.float32) + 1j * pairs[:, 1].astype(np.float32)


def spectral_peak(raw: np.memmap, sample_rate: int, center_hz: float) -> tuple[float, float]:
    """Choose the strongest measured RF line in a predeclared high-band window."""
    sample_count = raw.size // 2
    cumulative = None
    offsets = None
    windows = 0
    for second in np.arange(2.0, sample_count / sample_rate - 1.0, 3.0):
        first = int(second * sample_rate)
        last = first + sample_rate // 10
        if last > sample_count:
            break
        z = read_complex(raw, first, last)
        nperseg = min(16_384, len(z))
        offsets, power = signal.welch(z, fs=sample_rate, nperseg=nperseg,
                                      noverlap=nperseg // 2, return_onesided=False)
        if cumulative is None:
            cumulative = np.zeros_like(power, dtype=np.float64)
        cumulative += power
        windows += 1
    if windows < 3 or cumulative is None or offsets is None:
        raise ValueError("capture is too short for RF band survey")
    frequencies = center_hz + offsets
    near = (frequencies >= SEARCH_BAND_HZ[0]) & (frequencies <= SEARCH_BAND_HZ[1])
    if not np.any(near):
        raise ValueError("high RF band is outside the recorded bandwidth")
    indices = np.flatnonzero(near)
    winner = indices[np.argmax(cumulative[indices])]
    floor = np.median(cumulative[indices])
    contrast_db = 10.0 * np.log10(max(float(cumulative[winner]), 1e-30) / max(float(floor), 1e-30))
    return float(frequencies[winner]), float(contrast_db)


def envelope(raw: np.memmap, sample_rate: int, center_hz: float, rf_hz: float) -> np.ndarray:
    """Translate, anti-alias filter, and decimate physical IQ in bounded chunks."""
    if sample_rate == OUT_RATE_HZ:
        # A separately documented selected-band derivative is already
        # anti-alias filtered and decimated. Magnitude is invariant to its
        # residual center offset, so no second resampling is needed.
        sample_count = raw.size // 2
        output = np.empty(sample_count, dtype=np.float32)
        for first in range(0, sample_count, OUT_RATE_HZ * 10):
            last = min(first + OUT_RATE_HZ * 10, sample_count)
            output[first:last] = np.abs(read_complex(raw, first, last))
        return output
    if sample_rate != OUT_RATE_HZ * DECIMATION:
        raise ValueError("expected 2 MS/s physical IQ or 100 kS/s selected-band IQ")
    sample_count = raw.size // 2
    output_count = (sample_count + DECIMATION - 1) // DECIMATION
    output = np.empty(output_count, dtype=np.float32)
    block = sample_rate
    pad = 128 * DECIMATION
    frequency = rf_hz - center_hz
    for first in range(0, sample_count, block):
        last = min(first + block, sample_count)
        left = max(0, first - pad)
        right = min(sample_count, last + pad)
        z = read_complex(raw, left, right)
        samples = np.arange(left, right, dtype=np.float64)
        oscillator = np.exp((-2j * np.pi * frequency / sample_rate) * samples).astype(np.complex64)
        filtered = signal.resample_poly(z * oscillator, 1, DECIMATION,
                                        window=("kaiser", 8.0))
        trim = (first - left) // DECIMATION
        take = (last - first + DECIMATION - 1) // DECIMATION
        output[first // DECIMATION:first // DECIMATION + take] = np.abs(filtered[trim:trim + take])
    return output


def find_packet_start(amp: np.ndarray, physical: np.ndarray, chip_rate: int) -> tuple[float, float]:
    """Source-aware physical start alignment; no candidate bytes are selected."""
    template_chips = physical[:int(3.0 * chip_rate)]
    n_template = round(len(template_chips) * OUT_RATE_HZ / chip_rate)
    sample_chip = np.minimum(((np.arange(n_template) + 0.5) * chip_rate / OUT_RATE_HZ).astype(np.intp),
                             len(template_chips) - 1)
    template = 2.0 * template_chips[sample_chip].astype(np.float32) - 1.0
    template -= template.mean()
    search = amp[:min(len(amp), n_template + 4 * OUT_RATE_HZ)]
    feature = search - ndimage.uniform_filter1d(search, size=OUT_RATE_HZ // 2)
    correlation = signal.fftconvolve(feature, template[::-1], mode="valid")
    peak = int(np.argmax(correlation))
    quality = float(correlation[peak] / (len(template) * max(float(np.std(feature[peak:peak+n_template])), 1e-12)))
    return peak / OUT_RATE_HZ, quality


def chip_values(amp: np.ndarray, start: float, chip_rate: float,
                indices: np.ndarray) -> np.ndarray:
    positions = (start + (indices[:, None] + CHIP_FRACTIONS[None, :]) / chip_rate) * OUT_RATE_HZ
    if positions.size and (positions.min() < 0 or positions.max() >= len(amp)):
        raise ValueError("chip samples fall outside IQ")
    lower = np.floor(positions).astype(np.intp)
    fraction = positions - lower
    upper = np.minimum(lower + 1, len(amp) - 1)
    samples = amp[lower] * (1.0 - fraction) + amp[upper] * fraction
    return samples.mean(axis=1)


def known_prefix_quality(amp: np.ndarray, start: float, chip_rate: float,
                         known_chips: np.ndarray) -> float:
    indices = np.arange(len(known_chips), dtype=np.float64)
    values = chip_values(amp, start, chip_rate, indices)
    signs = 2.0 * known_chips.astype(np.float64) - 1.0
    signs -= signs.mean()
    return float(np.dot(signs, values) / (len(signs) * max(float(np.std(values)), 1e-12)))


def align_stage(amp: np.ndarray, predicted: float, chip_rate: float,
                known_chips: np.ndarray, search_chips: float = 80.0) -> tuple[float, float]:
    """Find a local known-prefix start without reading secret payload bits."""
    coarse = np.arange(-search_chips, search_chips + 0.5, 1.0)
    scores = np.array([known_prefix_quality(amp, predicted + shift / chip_rate,
                                             chip_rate, known_chips)
                       for shift in coarse])
    winner = float(coarse[int(np.argmax(scores))])
    fine = np.arange(winner - 1.0, winner + 1.01, 0.1)
    fine_scores = np.array([known_prefix_quality(amp, predicted + shift / chip_rate,
                                                  chip_rate, known_chips)
                            for shift in fine])
    index = int(np.argmax(fine_scores))
    return predicted + float(fine[index]) / chip_rate, float(fine_scores[index])


def manchester_margins(amp: np.ndarray, start: float, chip_rate: float,
                       prefix_chips: int, logical_count: int,
                       phase_chips: float = 0.0) -> np.ndarray:
    indices = prefix_chips + phase_chips + np.arange(2 * logical_count, dtype=np.float64)
    values = chip_values(amp, start, chip_rate, indices)
    return values[1::2] - values[0::2]  # >0 favors logical 1 (physical 01)


def repeated_manchester_margins(amp: np.ndarray, start: float, chip_rate: float,
                                prefix_chips: int, logical_count: int,
                                pair_repetitions: int,
                                phase_chips: float = 0.0) -> np.ndarray:
    """Demodulate 0=(10)^R, 1=(01)^R by combining each bit's R pairs."""
    if pair_repetitions < 1:
        raise ValueError("Manchester pair repetitions must be positive")
    chips_per_bit = 2 * pair_repetitions
    indices = (prefix_chips + phase_chips +
               np.arange(chips_per_bit * logical_count, dtype=np.float64))
    values = chip_values(amp, start, chip_rate, indices).reshape(-1, chips_per_bit)
    return values[:, 1::2].mean(axis=1) - values[:, ::2].mean(axis=1)


def fit_body_rate(amp: np.ndarray, start: float, nominal_rate: float,
                  prefix_chips: int, logical_count: int) -> tuple[float, float]:
    """Fit one block's linear chip clock using data-independent contrast."""
    probes = np.linspace(0, max(0, logical_count - 1), min(4096, logical_count)).astype(np.int64)
    probes = np.unique(probes)
    indices = prefix_chips + (2 * probes[:, None] + np.array([0, 1])[None, :]).ravel()
    rates = np.arange(nominal_rate - 1.0, nominal_rate + 1.001, 0.05)
    scores = []
    for rate in rates:
        values = chip_values(amp, start, float(rate), indices).reshape(-1, 2)
        differences = values[:, 1] - values[:, 0]
        scores.append(float(np.mean(np.abs(differences))))
    best = int(np.argmax(scores))
    return float(rates[best]), float(scores[best])


def tracked_manchester_margins(amp: np.ndarray, start: float, chip_rate: float,
                               prefix_chips: int, logical_count: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Track residual subchip phase using unsigned contrast, never payload bits."""
    chunk = 256
    phases = np.linspace(-0.75, 0.75, 25)
    spans = [(i, min(i + chunk, logical_count)) for i in range(0, logical_count, chunk)]
    differences: list[np.ndarray] = []
    contrast = np.empty((len(spans), len(phases)), dtype=np.float64)
    for row, (first, stop) in enumerate(spans):
        indices = prefix_chips + np.arange(2 * first, 2 * stop, dtype=np.float64)
        choices = np.empty((len(phases), stop - first), dtype=np.float64)
        for col, phase in enumerate(phases):
            values = chip_values(amp, start, chip_rate, indices + float(phase))
            choices[col] = values[1::2] - values[0::2]
        differences.append(choices)
        contrast[row] = np.mean(np.abs(choices), axis=1)
    penalty = 0.12 * max(float(np.median(np.max(contrast, axis=1))), 1e-9)
    scores = contrast[0] - 2 * penalty * np.abs(phases)
    predecessors = np.zeros(contrast.shape, dtype=np.int16)
    for row in range(1, len(spans)):
        transitions = scores[:, None] - penalty * np.abs(phases[:, None] - phases[None, :])
        predecessors[row] = np.argmax(transitions, axis=0)
        scores = contrast[row] + transitions[predecessors[row], np.arange(len(phases))]
    selected = np.empty(len(spans), dtype=np.int16)
    selected[-1] = int(np.argmax(scores))
    for row in range(len(spans) - 1, 0, -1):
        selected[row - 1] = predecessors[row, selected[row]]
    merged = np.concatenate([differences[row][col] for row, col in enumerate(selected)])
    applied = phases[selected]
    return merged, (float(applied.min()), float(applied.max()))
