"""Dataset utilities for multi-class arrhythmia classification."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .data_loader import LABEL_ID_MAP, PPGRecord


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration controlling signal conditioning and windowing."""

    window_seconds: float = 10.0
    sampling_rate: int = 125
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 8.0
    normalize: str = "zscore"
    augment: bool = False
    gaussian_noise_std: float = 0.01
    amplitude_scale_range: Tuple[float, float] = (0.9, 1.1)
    time_warp_sigma: float = 0.2  # Fraction of window duration


class PPGPreprocessor:
    """Applies filtering, normalization, and segmentation to raw PPG arrays."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._b, self._a = butter(
            N=4,
            Wn=[config.bandpass_low_hz / (config.sampling_rate / 2), config.bandpass_high_hz / (config.sampling_rate / 2)],
            btype="bandpass",
        )

    def prepare_record(self, record: PPGRecord) -> Tuple[np.ndarray, np.ndarray]:
        signal = self._resample(record.signal, record.sampling_rate)
        signal = self._bandpass(signal)
        signal = self._normalize(signal)
        windows = self._segment(signal)
        labels = np.full((windows.shape[0],), record.label_id, dtype=np.int64)
        if self.config.augment:
            windows = self._augment_batch(windows)
        return windows, labels

    def _resample(self, signal: np.ndarray, original_fs: int) -> np.ndarray:
        if original_fs == self.config.sampling_rate:
            return signal
        target_length = max(1, int(round(self.config.sampling_rate * len(signal) / original_fs)))
        return resample(signal, target_length)

    def _bandpass(self, signal: np.ndarray) -> np.ndarray:
        return filtfilt(self._b, self._a, signal)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        if self.config.normalize == "zscore":
            mean = np.mean(signal)
            std = np.std(signal) or 1.0
            return (signal - mean) / std
        if self.config.normalize == "minmax":
            min_val, max_val = float(np.min(signal)), float(np.max(signal))
            if math.isclose(max_val, min_val):
                return signal
            return (signal - min_val) / (max_val - min_val)
        return signal

    def _segment(self, signal: np.ndarray) -> np.ndarray:
        window_length = int(self.config.window_seconds * self.config.sampling_rate)
        total = len(signal)
        if total < window_length:
            padding = np.zeros(window_length - total, dtype=signal.dtype)
            signal = np.concatenate([signal, padding])
            total = len(signal)
        num_windows = total // window_length
        trimmed = signal[: num_windows * window_length]
        return trimmed.reshape(num_windows, window_length)

    def _augment_batch(self, windows: np.ndarray) -> np.ndarray:
        augmented = []
        for window in windows:
            augmented.append(self._augment(window))
        return np.stack(augmented)

    def _augment(self, window: np.ndarray) -> np.ndarray:
        data = window.copy()
        if self.config.amplitude_scale_range:
            scale = random.uniform(*self.config.amplitude_scale_range)
            data *= scale
        if self.config.gaussian_noise_std > 0:
            noise = np.random.normal(0, self.config.gaussian_noise_std, size=data.shape)
            data += noise
        if self.config.time_warp_sigma > 0:
            data = self._time_warp(data)
        return data

    def _time_warp(self, window: np.ndarray) -> np.ndarray:
        sigma = self.config.time_warp_sigma
        indices = np.arange(len(window))
        random_curve = np.random.normal(loc=0.0, scale=sigma, size=len(window))
        warped_indices = np.clip(indices + random_curve * len(window), 0, len(window) - 1)
        return np.interp(indices, warped_indices, window)


class PPGDataset(Dataset):
    """Torch dataset wrapping individual PPG windows."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        index_to_name: Dict[int, str],
        index_to_original_id: Dict[int, int],
    ) -> None:
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.index_to_name = index_to_name
        self.index_to_original_id = index_to_original_id
        self.name_to_index = {name: idx for idx, name in index_to_name.items()}
        self.num_classes = len(index_to_name)

    def __len__(self) -> int:  # noqa: D401
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def build_dataset(records: Dict[str, PPGRecord], config: PreprocessingConfig) -> PPGDataset:
    preprocessor = PPGPreprocessor(config)
    all_windows: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for record in records.values():
        windows, labels = preprocessor.prepare_record(record)
        all_windows.append(windows)
        all_labels.append(labels)
    if not all_windows:
        raise ValueError("No windows were produced. Check the source data and labels.")
    windows_concat = np.concatenate(all_windows, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)
    unique_ids = sorted(int(x) for x in np.unique(labels_concat))
    id_to_index = {original: new_idx for new_idx, original in enumerate(unique_ids)}
    remapped_labels = np.vectorize(id_to_index.get)(labels_concat)

    index_to_original = {new_idx: original for original, new_idx in id_to_index.items()}
    index_to_name: Dict[int, str] = {}
    for label_name, original_id in LABEL_ID_MAP.items():
        if original_id in id_to_index:
            index = id_to_index[original_id]
            index_to_name[index] = label_name

    return PPGDataset(
        windows_concat,
        remapped_labels.astype(np.int64),
        index_to_name,
        index_to_original,
    )


def split_dataset(
    dataset: PPGDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    total = len(dataset)
    if total < 3:
        raise ValueError("Dataset too small for stratified splitting.")

    indices = np.arange(total)
    labels = dataset.labels.numpy()

    label_counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    if np.any(label_counts < 2):
        LOGGER.warning(
            "Unable to stratify split due to rare classes (counts: %s). Falling back to random split.",
            label_counts.tolist(),
        )
        generator = torch.Generator().manual_seed(seed)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size
        return tuple(
            random_split(dataset, [train_size, val_size, test_size], generator=generator)
        )

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    remaining_ratio = 1.0 - train_ratio
    if remaining_ratio <= 0:
        raise ValueError("Train ratio must be < 1.0")
    val_fraction = val_ratio / remaining_ratio
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_fraction,
        stratify=labels[temp_indices],
        random_state=seed,
    )

    return (
        Subset(dataset, train_indices.tolist()),
        Subset(dataset, val_indices.tolist()),
        Subset(dataset, test_indices.tolist()),
    )


def create_dataloaders(
    dataset: PPGDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    train_set, val_set, test_set = split_dataset(dataset, seed=seed)
    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


__all__ = [
    "PreprocessingConfig",
    "PPGPreprocessor",
    "PPGDataset",
    "build_dataset",
    "create_dataloaders",
]
