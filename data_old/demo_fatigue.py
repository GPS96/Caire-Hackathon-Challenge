"""Small demo that computes a fatigue score from sample facial+HRV data.

Run this from the `ppr_project/Caire-Hackathon-Challenge/data` folder.
"""
from __future__ import annotations

import numpy as np

try:
    # prefer project extractor when available
    from arrhythmia_project.models import extract_hrv_features  # type: ignore
except Exception:
    # fall back to a minimal HRV extractor that does not require torch
    from typing import List

    import numpy as np

    try:
        from scipy.signal import find_peaks
    except Exception:
        find_peaks = None  # type: ignore

    def _simple_find_peaks(x: np.ndarray, distance: int = 1) -> np.ndarray:
        # naive local-maximum detector (safe fallback when scipy not present)
        peaks = []
        n = x.size
        for i in range(1, n - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1]:
                peaks.append(i)
        return np.array(peaks, dtype=int)


    def extract_hrv_features(windows: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Fallback, minimal HRV feature extraction for the demo.

        This produces the same column ordering used in the project's
        function so the demo can run without heavy dependencies.
        """
        feature_vectors: List[np.ndarray] = []
        for window in windows:
            amp_mean = float(np.mean(window))
            amp_std = float(np.std(window))
            amp_min = float(np.min(window))
            amp_max = float(np.max(window))
            amp_range = amp_max - amp_min
            amp_median = float(np.median(window))
            amp_q25 = float(np.percentile(window, 25))
            amp_q75 = float(np.percentile(window, 75))

            if find_peaks is not None:
                peaks, _ = find_peaks(window, distance=int(0.3 * sampling_rate))
            else:
                peaks = _simple_find_peaks(window, distance=int(0.3 * sampling_rate))

            rr_intervals = np.diff(peaks) / sampling_rate if peaks.size >= 2 else np.array([])
            rr_intervals = rr_intervals[~np.isnan(rr_intervals)]

            if rr_intervals.size >= 1:
                mean_rr = float(np.mean(rr_intervals))
                std_rr = float(np.std(rr_intervals)) if rr_intervals.size > 1 else 0.0
                heart_rate = 60.0 / mean_rr if mean_rr > 0 else 0.0
                rr_max = float(np.max(rr_intervals))
                rr_min = float(np.min(rr_intervals))
                rr_median = float(np.median(rr_intervals))
            else:
                mean_rr = std_rr = heart_rate = rr_max = rr_min = rr_median = 0.0

            feature_vectors.append(
                np.array(
                    [
                        amp_mean,
                        amp_std,
                        amp_min,
                        amp_max,
                        amp_range,
                        amp_median,
                        amp_q25,
                        amp_q75,
                        heart_rate,
                        mean_rr,
                        std_rr,
                        rr_max,
                        rr_min,
                        rr_median,
                    ],
                    dtype=np.float32,
                )
            )
        return np.vstack(feature_vectors)


from fatigue import combined_fatigue_score, alert_from_score


def main() -> None:
    # Simulate PPG windows (two windows) with simple synthetic pulses
    sr = 50
    t = np.linspace(0, 10, 10 * sr)
    # synthetic pulse-like signals (two windows)
    pulse = 0.5 * (1.0 + np.sin(2 * np.pi * 1.2 * t))
    windows = np.stack([pulse, 0.9 * pulse])

    hrv = extract_hrv_features(windows, sr)

    # Convert first window's HRV features to dict expected by fatigue.hrv_score_from_features
    # extract_hrv_features returns columns matching docs in models.py
    # indices: heart_rate (8), mean_rr (9), std_rr (10)
    hrv_feats = {
        "heart_rate": float(hrv[0, 8]),
        "mean_rr": float(hrv[0, 9]),
        "std_rr": float(hrv[0, 10]),
    }

    # Placeholder facial features collected from a hypothetical detector
    facial_feats = {
        "blink_rate": 18.0,
        "avg_blink_duration": 0.18,
        "yawns_count": 1.0,
        "mouth_open_duration": 0.05,
        "head_nod_count": 0.5,
        "head_pitch_std": 1.2,
    }

    score = combined_fatigue_score(facial_feats, hrv_feats, facial_weight=0.6)
    alert = alert_from_score(score, threshold=70.0)

    print("HRV features:", hrv_feats)
    print("Facial features:", facial_feats)
    print(f"Fatigue score: {score:.1f} / 100")
    print("Alert:", alert)


if __name__ == "__main__":
    main()
