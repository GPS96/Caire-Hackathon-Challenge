"""
Signal cleaning utilities to convert a noisy rPPG/BVP trace into a cleaner PPG-like waveform.

Pipeline (default):
- Detrend (linear)
- Band-pass filter (Butterworth, 0.7–4.0 Hz ~ 42–240 bpm)
- Z-score normalization
- Optional moving-average smoothing

Usage:
from rppg.cleaning import clean_bvp
clean, info = clean_bvp(raw, fs=30.0)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.signal import butter, filtfilt, detrend


@dataclass
class CleanOptions:
    low_hz: float = 0.7
    high_hz: float = 4.0
    order: int = 3
    smooth_win_sec: float = 0.3  # moving average window in seconds (0 disables)


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 3):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if high <= low:
        high = min(0.999, low + 0.01)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs, order)
    # Use filtfilt for zero-phase filtering
    return filtfilt(b, a, x, method='pad')


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    y = (c[w:] - c[:-w]) / float(w)
    # pad to original length
    pad_left = w // 2
    pad_right = len(x) - len(y) - pad_left
    return np.pad(y, (pad_left, pad_right), mode='edge')


def estimate_fs_from_time(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[~np.isnan(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Cannot estimate sampling rate from time vector")
    return float(1.0 / np.median(dt))


def clean_bvp(
    rppg: np.ndarray,
    fs: Optional[float] = None,
    *,
    options: Optional[CleanOptions] = None,
    time: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Clean an rPPG/BVP signal into a PPG-like waveform.

    Args:
        rppg: 1D array-like raw rPPG signal
        fs: sampling frequency in Hz; if None and time provided, infer from time
        options: CleanOptions
        time: optional time vector for fs estimation

    Returns:
        (cleaned_signal, info_dict)
    """
    x = np.asarray(rppg, dtype=float).ravel()
    if options is None:
        options = CleanOptions()

    if fs is None:
        if time is None:
            raise ValueError("fs must be provided when time is None")
        fs = estimate_fs_from_time(np.asarray(time, dtype=float).ravel())

    # Step 1: Detrend
    x_dt = detrend(x, type='linear')

    # Step 2: Band-pass
    x_bp = bandpass_filter(x_dt, fs, options.low_hz, options.high_hz, options.order)

    # Step 3: Z-score
    std = np.std(x_bp) if np.std(x_bp) > 1e-12 else 1.0
    x_z = (x_bp - np.mean(x_bp)) / std

    # Step 4: Optional smoothing
    if options.smooth_win_sec and options.smooth_win_sec > 0:
        win = max(1, int(round(options.smooth_win_sec * fs)))
        x_sm = moving_average(x_z, win)
    else:
        x_sm = x_z

    info = {
        'fs': float(fs),
        'low_hz': float(options.low_hz),
        'high_hz': float(options.high_hz),
        'order': float(options.order),
        'smooth_win_sec': float(options.smooth_win_sec or 0.0),
    }
    return x_sm.astype(float), info
