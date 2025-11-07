"""Utilities to compute a driver fatigue score from facial features and HRV.

This module defines a small, explainable scoring algorithm that combines
facial movement indicators (blink_rate, avg_blink_duration, yawns_count,
mouth_open_duration, head_nod_count, head_pitch_std) with HRV-related
indicators (heart_rate, mean_rr, std_rr).

The API is deliberately simple: callers provide per-window aggregated
facial features and HRV features (one vector per time window). The
module returns a normalized fatigue score in range [0, 100] and an
alert decision when the score exceeds a configurable threshold.

Assumptions made:
- Inputs are aggregated per analysis window (e.g., 30s or 60s).
- HRV features can be the rows returned by `extract_hrv_features` in
  the arrhythmia project (heart_rate, mean_rr, std_rr are available).

This file is self-contained and has no runtime-side effects.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def normalize(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Map value from [lo,hi] to [0,1]. If invert True, lower values map to 1.

    Returns clipped result in [0,1].
    """
    if hi == lo:
        return 0.0
    v = (value - lo) / (hi - lo)
    if invert:
        v = 1.0 - v
    return _clamp01(v)


def facial_score_from_features(features: Dict[str, float]) -> float:
    """Compute a facial movement subs-score in [0,1].

    Expected keys in `features` (all optional, reasonable defaults used):
      - blink_rate: blinks per minute
      - avg_blink_duration: seconds
      - yawns_count: yawns per window
      - mouth_open_duration: fraction of window mouth open (0..1)
      - head_nod_count: number of nods per window
      - head_pitch_std: standard deviation of head pitch (degrees)

    The returned score increases with signs of drowsiness (high yawns,
    long blinks, frequent nods) and is normalized to [0,1].
    """
    # defaults
    blink_rate = float(features.get("blink_rate", 12.0))
    avg_blink_duration = float(features.get("avg_blink_duration", 0.12))
    yawns = float(features.get("yawns_count", 0.0))
    mouth_open = float(features.get("mouth_open_duration", 0.0))
    nods = float(features.get("head_nod_count", 0.0))
    pitch_std = float(features.get("head_pitch_std", 0.5))

    # heuristic normalization ranges (tunable)
    # blink_rate: low blink rate may indicate focus; very low or very high may be bad
    blink_rate_score = normalize(blink_rate, 6.0, 30.0, invert=False)

    # avg blink duration: longer blinks more drowsy
    blink_dur_score = normalize(avg_blink_duration, 0.05, 0.5, invert=False)

    # yawns per window (0..5 typical); more yawns -> more drowsy
    yawns_score = normalize(yawns, 0.0, 3.0, invert=False)

    # mouth open fraction in window
    mouth_score = _clamp01(mouth_open)

    # nods: frequent nodding indicates micro-sleep
    nods_score = normalize(nods, 0.0, 4.0, invert=False)

    # head pitch variability: high std may indicate unstable head movement (drowsy)
    pitch_score = normalize(pitch_std, 0.1, 6.0, invert=False)

    # weights (tunable); yawns and nods are stronger indicators
    w = dict(
        blink_rate=0.12,
        blink_dur=0.18,
        yawns=0.30,
        mouth=0.10,
        nods=0.20,
        pitch=0.10,
    )

    score = (
        w["blink_rate"] * blink_rate_score
        + w["blink_dur"] * blink_dur_score
        + w["yawns"] * yawns_score
        + w["mouth"] * mouth_score
        + w["nods"] * nods_score
        + w["pitch"] * pitch_score
    )

    return _clamp01(score)


def hrv_score_from_features(hrv_features: Dict[str, float]) -> float:
    """Compute HRV-derived subs-score in [0,1].

    Expects keys (at least one present):
      - std_rr: standard deviation of RR intervals (s)
      - mean_rr: mean RR interval (s)
      - heart_rate: beats per minute
      - rmssd: if available, provides a sensitive HRV metric

    Lower HRV (low std_rr/rmssd) maps to higher fatigue score.
    """
    std_rr = float(hrv_features.get("std_rr", 0.0))
    mean_rr = float(hrv_features.get("mean_rr", 0.0))
    heart_rate = float(hrv_features.get("heart_rate", 0.0))
    rmssd = float(hrv_features.get("rmssd", -1.0))

    # Use rmssd if provided, otherwise use std_rr
    if rmssd >= 0:
        hrv_raw = rmssd
        # typical RMSSD range: 10 (low) - 80 (high)
        hrv_norm = normalize(hrv_raw, 10.0, 80.0, invert=True)
    else:
        # std_rr typical: 0.02 - 0.2 s depending on data
        hrv_norm = normalize(std_rr, 0.02, 0.18, invert=True)

    # heart rate: extreme high or low might indicate stress; here we map higher HR -> more fatigue
    hr_norm = normalize(heart_rate, 50.0, 120.0, invert=False)

    # mean_rr is inverse of HR; smaller mean_rr -> higher HR
    mean_rr_norm = normalize(mean_rr, 0.4, 1.2, invert=False)

    # weights
    w_hr = dict(hrv=0.7, hr=0.2, mean_rr=0.1)
    score = w_hr["hrv"] * hrv_norm + w_hr["hr"] * hr_norm + w_hr["mean_rr"] * mean_rr_norm
    return _clamp01(score)


def combined_fatigue_score(
    facial_feats: Dict[str, float],
    hrv_feats: Dict[str, float],
    facial_weight: float = 0.6,
) -> float:
    """Combine facial and HRV subscores into a final fatigue score (0..100).

    facial_weight controls importance of facial cues (0..1). HRV weight = 1 - facial_weight.
    """
    f = facial_score_from_features(facial_feats)
    h = hrv_score_from_features(hrv_feats)
    combined = float(_clamp01(facial_weight * f + (1.0 - facial_weight) * h))
    return combined * 100.0


def alert_from_score(score: float, threshold: float = 70.0) -> Dict[str, object]:
    """Return alert dict with decision and recommended action based on score.

    - score: 0..100
    - threshold: numeric threshold to trigger an immediate alert
    """
    if score >= 90.0:
        level = "critical"
        action = "Stop driving now and take a break / switch driver"
    elif score >= threshold:
        level = "warning"
        action = "Take a break soon; pull over when safe"
    elif score >= 40.0:
        level = "notice"
        action = "Take a short break and monitor yourself"
    else:
        level = "normal"
        action = "OK"

    return {"score": float(score), "level": level, "action": action}


__all__ = [
    "facial_score_from_features",
    "hrv_score_from_features",
    "combined_fatigue_score",
    "alert_from_score",
]
