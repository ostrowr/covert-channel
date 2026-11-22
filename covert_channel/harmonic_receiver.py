"""Experimental preamble-weighted combination of two RF harmonic envelopes.

The SDR capture is centered at 1.4 MHz with a 2 MS/s sample rate, so both
1.505 MHz and 754 kHz fall in the same IQ recording. This module selects a
low-band weight using the known Manchester preamble only; no payload bits or
FEC result contribute to that choice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fast_receiver import EnergyReceiver, decode_manchester, energies_from_file


LOW_CARRIER_HZ = 754_000
LOW_HALF_BANDWIDTH_HZ = 15_000
MAX_LOW_WEIGHT = 0.75


def preamble_low_weight(
    high: np.ndarray,
    low: np.ndarray,
    *,
    sample_rate: int,
    window_samples: int,
    hop_samples: int,
    physical_rate: int,
    preamble_bits: str,
    payload_logical_bits_count: int,
) -> tuple[float, dict]:
    """Estimate relative low-band quality from preamble margins and scatter."""
    signs = np.fromiter((1 if bit == "0" else -1 for bit in preamble_bits[::2]),
                        dtype=np.float64)
    band_summaries = []
    for energy in (high, low):
        result = decode_manchester(
            energy,
            sample_rate=sample_rate,
            window_samples=window_samples,
            hop_samples=hop_samples,
            physical_rate=physical_rate,
            preamble_bits=preamble_bits,
            payload_logical_bits_count=payload_logical_bits_count,
        )
        chip_pairs = result.chip_scores_db.reshape(-1, 2)
        margins = (chip_pairs[:, 1] - chip_pairs[:, 0])[:len(signs)] * signs
        median = float(np.median(margins))
        robust_scatter = float(1.4826 * np.median(np.abs(margins - median)))
        quality = max(0.0, median) / (robust_scatter ** 2 + 1.0)
        band_summaries.append({
            "median_preamble_margin_db": median,
            "robust_scatter_db": robust_scatter,
            "quality": quality,
            "start_window": result.start_window,
        })
    if abs(band_summaries[0]["start_window"] - band_summaries[1]["start_window"]) > 2:
        weight = 0.0
    else:
        weight = float(np.clip(
            band_summaries[1]["quality"] / max(band_summaries[0]["quality"], 1e-9),
            0.0,
            MAX_LOW_WEIGHT,
        ))
    return weight, {"high": band_summaries[0], "low": band_summaries[1]}


def combine_preamble_weighted(
    iq_path: Path,
    high: np.ndarray,
    *,
    sample_rate: int,
    center_hz: int,
    window_samples: int,
    hop_samples: int,
    fft_size: int,
    physical_rate: int,
    preamble_bits: str,
    payload_logical_bits_count: int,
) -> tuple[np.ndarray, dict]:
    """Read the low harmonic from the same IQ and return combined dB envelopes."""
    low_receiver = EnergyReceiver(
        sample_rate=sample_rate,
        center_hz=center_hz,
        carrier_hz=LOW_CARRIER_HZ,
        window_samples=window_samples,
        hop_samples=hop_samples,
        fft_size=fft_size,
        half_bandwidth_hz=LOW_HALF_BANDWIDTH_HZ,
        energy_mode="signal",
    )
    low = energies_from_file(iq_path, low_receiver)
    if high.shape != low.shape:
        raise ValueError("harmonic envelopes have different analysis grids")
    weight, preamble_quality = preamble_low_weight(
        high,
        low,
        sample_rate=sample_rate,
        window_samples=window_samples,
        hop_samples=hop_samples,
        physical_rate=physical_rate,
        preamble_bits=preamble_bits,
        payload_logical_bits_count=payload_logical_bits_count,
    )
    return high + weight * low, {
        "mode": "preamble",
        "low_carrier_hz": LOW_CARRIER_HZ,
        "low_half_bandwidth_hz": LOW_HALF_BANDWIDTH_HZ,
        "low_weight": weight,
        "preamble_quality": preamble_quality,
    }
