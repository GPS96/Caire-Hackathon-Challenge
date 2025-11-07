"""Unit tests for the fatigue scoring utilities."""
from __future__ import annotations

from fatigue import (
    facial_score_from_features,
    hrv_score_from_features,
    combined_fatigue_score,
    alert_from_score,
)


def test_facial_score_basic():
    feats = {
        "blink_rate": 20.0,
        "avg_blink_duration": 0.25,
        "yawns_count": 2.0,
        "mouth_open_duration": 0.2,
        "head_nod_count": 1.0,
        "head_pitch_std": 2.0,
    }
    s = facial_score_from_features(feats)
    assert 0.0 <= s <= 1.0


def test_hrv_score_low_high():
    low_hrv = {"std_rr": 0.02, "mean_rr": 0.8, "heart_rate": 75}
    high_hrv = {"std_rr": 0.15, "mean_rr": 1.0, "heart_rate": 60}
    s_low = hrv_score_from_features(low_hrv)
    s_high = hrv_score_from_features(high_hrv)
    # lower HRV should yield higher fatigue score (invert=True used)
    assert 0.0 <= s_low <= 1.0
    assert 0.0 <= s_high <= 1.0
    assert s_low >= s_high - 1e-6


def test_combined_and_alert():
    facial = {"yawns_count": 3.0, "avg_blink_duration": 0.3}
    hrv = {"std_rr": 0.02, "mean_rr": 0.6, "heart_rate": 85}
    score = combined_fatigue_score(facial, hrv, facial_weight=0.5)
    assert 0.0 <= score <= 100.0
    alert = alert_from_score(score, threshold=60.0)
    assert "score" in alert and "level" in alert and "action" in alert
