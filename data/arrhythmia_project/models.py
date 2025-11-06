"""Model definitions for feature-based and deep learning classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks


def extract_hrv_features(windows: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Compute amplitude and HRV descriptors from segmented PPG windows."""
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

        peaks, _ = find_peaks(window, distance=int(0.3 * sampling_rate))
        rr_intervals = np.diff(peaks) / sampling_rate
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

def build_feature_classifier(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


class CNNBiLSTM(nn.Module):
    """1D CNN front-end followed by BiLSTM for sequence classification."""

    def __init__(self, input_length: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        conv_output = input_length // 4  # two pooling layers
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(conv_output * 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        outputs, _ = self.lstm(x)
        flat = outputs.reshape(outputs.size(0), -1)
        return self.classifier(flat)

class CNNATLSTM(nn.Module):
    """1D CNN front-end + Attention LSTM for sequence classification."""

    def __init__(self, input_length: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        conv_output = input_length // 4  # pooling reduces length
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.attn = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 256)
        return self.classifier(context)

@dataclass
class TrainingArtifacts:
    """Bundle containing trained models and metadata."""

    feature_model: Pipeline
    deep_model_state: Dict[str, torch.Tensor]
    index_label_mapping: Dict[int, str]
    index_to_original_id: Dict[int, int]

    @property
    def label_mapping(self) -> Dict[int, str]:
        """Return mapping from contiguous dataset indices to label names."""

        return dict(self.index_label_mapping)

    @property
    def original_label_mapping(self) -> Dict[int, str]:
        """Return mapping from original label ids to their human-readable names."""

        return {
            original_id: self.index_label_mapping[idx]
            for idx, original_id in self.index_to_original_id.items()
        }


__all__ = [
    "extract_hrv_features",
    "build_feature_classifier",
    "CNNBiLSTM",
    "CNNATLSTM",
    "TrainingArtifacts",
]
