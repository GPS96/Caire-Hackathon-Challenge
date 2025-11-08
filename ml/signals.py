"""Signal processing helpers for rPPG cleanup and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch


def filter_bandpass(signal: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    if signal.size == 0:
        return signal
    nyquist = fs / 2
    low = max(low_hz / nyquist, 0.01)
    high = min(high_hz / nyquist, 0.99)
    b, a = butter(N=2, Wn=[low, high], btype="band")
    return filtfilt(b, a, signal)


def compute_heart_rate_and_ibi(signal: np.ndarray, fs: int) -> Tuple[float, list[float]]:
    if signal.size < fs:
        raise ValueError("Not enough samples to compute heart rate")
    peaks, _ = find_peaks(signal, distance=int(fs * 0.4))
    if peaks.size < 2:
        raise ValueError("Insufficient peaks detected in signal")
    ibi_samples = np.diff(peaks) / fs * 1000.0
    heart_rate = 60000.0 / np.mean(ibi_samples)
    return float(heart_rate), ibi_samples.tolist()


def estimate_signal_quality(signal: np.ndarray) -> float:
    """Legacy, variance-based quality metric in [0,1].

    Kept for backward compatibility; not recommended for display.
    """
    if signal.size == 0:
        return 0.0
    variance = float(np.var(signal))
    normalized = min(variance / 2.0, 1.0)
    return max(normalized, 0.0)


def estimate_signal_quality_psd(signal: np.ndarray, fs: int) -> float:
    """Estimate signal quality using power spectral density SNR in the heart-rate band.

    Approach:
    - Compute Welch PSD over 0.5–4.0 Hz.
    - Find dominant frequency f0 in this band.
        - Define signal band around the fundamental and its second harmonic:
                fundamental: f0 ± 0.20 Hz
                second harmonic: 2*f0 ± 0.25 Hz
            Each band is clipped to [0.5, 4.0].
        - Quality = (power_in_fundamental + power_in_harmonic) / total_power_in_0.5–4.0, clipped to [0,1].

    Returns:
        float in [0,1], where values > 0.7 typically indicate a clean, periodic pulse.
    """
    if signal.size < max(fs // 2, 64):
        return 0.0
    # Welch PSD
    nperseg = min(1024, max(64, signal.size // 2))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    if f.size == 0 or Pxx.size == 0:
        return 0.0
    # Band of interest
    band_lo, band_hi = 0.5, 4.0
    band_mask = (f >= band_lo) & (f <= band_hi)
    if not np.any(band_mask):
        return 0.0
    f_band = f[band_mask]
    P_band = Pxx[band_mask]
    total_power = float(np.trapz(P_band, f_band))
    if total_power <= 0:
        return 0.0
    # Dominant peak
    peak_idx = int(np.argmax(P_band))
    f0 = float(f_band[peak_idx])
    # Fundamental band
    win_f = 0.20
    sig_mask_f = (f_band >= max(band_lo, f0 - win_f)) & (f_band <= min(band_hi, f0 + win_f))
    P_f = float(np.trapz(P_band[sig_mask_f], f_band[sig_mask_f])) if np.any(sig_mask_f) else 0.0
    # Second harmonic band (optional)
    f2 = 2.0 * f0
    win_h = 0.25
    sig_mask_h = (f_band >= max(band_lo, f2 - win_h)) & (f_band <= min(band_hi, f2 + win_h))
    P_h = float(np.trapz(P_band[sig_mask_h], f_band[sig_mask_h])) if np.any(sig_mask_h) else 0.0
    P_signal = P_f + P_h
    quality = P_signal / total_power
    # Clamp to [0,1]
    return float(max(0.0, min(1.0, quality)))
