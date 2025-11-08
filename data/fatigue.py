"""
Driver fatigue scoring utilities (facial + optional HRV).

Expose:
- facial_score_from_features
- hrv_score_from_features
- compute_subscores_and_score
- combined_fatigue_score
- alert_from_score
"""
from typing import Dict, Optional, Tuple
import math
import numpy as np


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def normalize(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Map value in [lo,hi] -> [0,1]; if invert True, lower values map to 1."""
    lo = float(lo); hi = float(hi)
    v = _safe_float(value, default=lo)
    if hi == lo:
        return 0.0
    u = (v - lo) / (hi - lo)
    if invert:
        u = 1.0 - u
    return _clamp01(u)


def facial_score_from_features(features: Dict[str, float]) -> float:
    """Return facial sub-score in [0,1] (higher = more drowsy)."""
    blink_rate      = _safe_float(features.get("blink_rate", 12.0))
    avg_blink_dur   = _safe_float(features.get("avg_blink_duration", 0.12))
    yawns           = _safe_float(features.get("yawns_count", 0.0))
    mouth_open      = _safe_float(features.get("mouth_open_duration", 0.0))
    nods            = _safe_float(features.get("head_nod_count", 0.0))
    pitch_std       = _safe_float(features.get("head_pitch_std", 0.5))

    s_blink   = normalize(blink_rate, 6.0, 30.0)
    s_blink_d = normalize(avg_blink_dur, 0.05, 0.5)
    s_yawn    = normalize(yawns, 0.0, 3.0)
    s_mouth   = _clamp01(mouth_open)
    s_nods    = normalize(nods, 0.0, 4.0)
    s_pitch   = normalize(pitch_std, 0.1, 6.0)

    # Heuristic weights (sum=1)
    score = (
        0.12 * s_blink +
        0.18 * s_blink_d +
        0.30 * s_yawn +
        0.10 * s_mouth +
        0.20 * s_nods +
        0.10 * s_pitch
    )
    return _clamp01(score)


def hrv_score_from_features(hrv_features: Dict[str, float]) -> float:
    """Return HRV sub-score in [0,1]; lower HRV -> higher score."""
    std_rr     = _safe_float(hrv_features.get("std_rr", 0.0))
    mean_rr    = _safe_float(hrv_features.get("mean_rr", 0.0))
    heart_rate = _safe_float(hrv_features.get("heart_rate", 0.0))
    rmssd      = _safe_float(hrv_features.get("rmssd", -1.0), default=-1.0)

    if rmssd >= 0:
        hrv_norm = normalize(rmssd, 10.0, 80.0, invert=True)
    else:
        hrv_norm = normalize(std_rr, 0.02, 0.18, invert=True)

    hr_norm      = normalize(heart_rate, 50.0, 120.0)
    mean_rr_norm = normalize(mean_rr, 0.4, 1.2)

    score = 0.7 * hrv_norm + 0.2 * hr_norm + 0.1 * mean_rr_norm
    return _clamp01(score)


def compute_subscores_and_score(
    facial_feats: Dict[str, float],
    hrv_feats: Dict[str, float],
    facial_weight: float = 0.6,
    baseline: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Return (combined_score_0_100, subscores_dict)."""

    # Facial subscores
    blink_rate    = _safe_float(facial_feats.get("blink_rate", 12.0))
    blink_dur     = _safe_float(facial_feats.get("avg_blink_duration", 0.12))
    yawns         = _safe_float(facial_feats.get("yawns_count", 0.0))
    mouth_open    = _safe_float(facial_feats.get("mouth_open_duration", 0.0))
    nods          = _safe_float(facial_feats.get("head_nod_count", 0.0))
    pitch_std     = _safe_float(facial_feats.get("head_pitch_std", 0.5))

    s_blink     = normalize(blink_rate, 6.0, 30.0)
    s_blink_dur = normalize(blink_dur, 0.05, 0.5)
    s_yawns     = normalize(yawns, 0.0, 3.0)
    s_mouth     = _clamp01(mouth_open)
    s_nods      = normalize(nods, 0.0, 4.0)
    s_pitch     = normalize(pitch_std, 0.1, 6.0)

    facial_subscores = {
        "blink_rate": s_blink * 100.0,
        "avg_blink_duration": s_blink_dur * 100.0,
        "yawns_count": s_yawns * 100.0,
        "mouth_open_duration": s_mouth * 100.0,
        "head_nod_count": s_nods * 100.0,
        "head_pitch_std": s_pitch * 100.0,
    }
    facial_score = facial_score_from_features(facial_feats) * 100.0

    # HRV subscores
    std_rr     = _safe_float(hrv_feats.get("std_rr", 0.0))
    mean_rr    = _safe_float(hrv_feats.get("mean_rr", 0.0))
    heart_rate = _safe_float(hrv_feats.get("heart_rate", 0.0))
    rmssd      = _safe_float(hrv_feats.get("rmssd", -1.0), default=-1.0)

    if rmssd >= 0:
        hrv_norm = normalize(rmssd, 10.0, 80.0, invert=True)
    else:
        hrv_norm = normalize(std_rr, 0.02, 0.18, invert=True)

    if baseline and _safe_float(baseline.get("heart_rate", 0.0)) > 0:
        hr_delta  = abs(heart_rate - _safe_float(baseline["heart_rate"]))
        s_hr      = normalize(hr_delta, 0.0, 12.0)
        s_mean_rr = normalize(abs(mean_rr - _safe_float(baseline.get("mean_rr", mean_rr))), 0.0, 0.2)
    else:
        s_hr      = normalize(heart_rate, 55.0, 90.0)
        s_mean_rr = normalize(mean_rr, 0.4, 1.2)

    hrv_subscores = {
        "heart_rate": s_hr * 100.0,
        "std_rr": hrv_norm * 100.0,
        "mean_rr": s_mean_rr * 100.0,
    }
    hrv_score = _clamp01(0.7 * hrv_norm + 0.2 * s_hr + 0.1 * s_mean_rr) * 100.0

    # Fuse
    combined = _clamp01((facial_weight * (facial_score / 100.0)) +
                        ((1.0 - facial_weight) * (hrv_score / 100.0))) * 100.0

    subs = {**facial_subscores, **hrv_subscores,
            "facial_score": facial_score, "hrv_score": hrv_score}
    return combined, subs


def combined_fatigue_score(
    facial_feats: Dict[str, float],
    hrv_feats: Dict[str, float],
    facial_weight: float = 0.6,
) -> float:
    """Convenience wrapper: final score in [0,100]."""
    f = facial_score_from_features(facial_feats)
    h = hrv_score_from_features(hrv_feats)
    return _clamp01(facial_weight * f + (1.0 - facial_weight) * h) * 100.0


def alert_from_score(score: float, threshold: float = 70.0) -> Dict[str, object]:
    """Return alert dict with decision and recommended action."""
    s = _safe_float(score, 0.0)
    if s >= 90.0:
        level = "critical"; action = "Stop driving now and take a break / switch driver"
    elif s >= threshold:
        level = "warning";  action = "Take a break soon; pull over when safe"
    elif s >= 40.0:
        level = "notice";   action = "Take a short break and monitor yourself"
    else:
        level = "normal";   action = "OK"
    return {"score": float(s), "level": level, "action": action}


__all__ = [
    "facial_score_from_features",
    "hrv_score_from_features",
    "compute_subscores_and_score",
    "combined_fatigue_score",
    "alert_from_score",
]