"""Dataset utilities for CAIRE arrhythmia classification."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, Dataset, random_split

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration for CAIRE PPG signal preprocessing."""
    window_seconds: float = 10.0
    sampling_rate: int = 100  # CAIRE uses 100 Hz
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 8.0
    normalize: str = "zscore"
    augment: bool = False
    gaussian_noise_std: float = 0.01
    amplitude_scale_range: Tuple[float, float] = (0.9, 1.1)
    time_warp_sigma: float = 0.2


class PPGPreprocessor:
    """Applies preprocessing and augmentation to PPG signals."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.window_length = int(config.window_seconds * config.sampling_rate)

    def _bandpass(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to PPG signal."""
        try:
            nyquist = self.config.sampling_rate / 2
            low = self.config.bandpass_low_hz / nyquist
            high = self.config.bandpass_high_hz / nyquist
            low = np.clip(low, 0.001, 0.999)
            high = np.clip(high, low + 0.001, 0.999)
            
            b, a = butter(4, [low, high], btype="band")
            filtered = filtfilt(b, a, signal)
            return filtered
        except Exception as e:
            LOGGER.warning(f"Bandpass filter failed: {e}. Returning original signal.")
            return signal

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal using specified method."""
        if self.config.normalize == "zscore":
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 0:
                return (signal - mean) / std
            return signal
        elif self.config.normalize == "minmax":
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val - min_val > 0:
                return (signal - min_val) / (max_val - min_val)
            return signal
        return signal

    def _augment(self, signal: np.ndarray) -> np.ndarray:
        """Apply data augmentation to signal."""
        # Gaussian noise
        signal = signal + np.random.normal(0, self.config.gaussian_noise_std, signal.shape)
        
        # Amplitude scaling
        scale = np.random.uniform(*self.config.amplitude_scale_range)
        signal = signal * scale
        
        # Time warping
        if self.config.time_warp_sigma > 0:
            n = len(signal)
            warp_path = np.cumsum(np.random.normal(1.0, self.config.time_warp_sigma, n))
            warp_path = np.clip(warp_path, 0, n - 1)
            signal = np.interp(np.arange(n), warp_path, signal)
        
        return signal

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline."""
        signal = self._bandpass(signal)
        signal = self._normalize(signal)
        if self.config.augment:
            signal = self._augment(signal)
        return signal


class PPGDataset(Dataset):
    """PyTorch Dataset for PPG signals."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        index_to_name: Dict[int, str],
        index_to_original_id: Dict[int, int],
    ):
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.index_to_name = index_to_name
        self.index_to_original_id = index_to_original_id
        self.num_classes = len(index_to_name)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def build_dataset_caire(
    train_segments: np.ndarray,
    train_labels: np.ndarray,
    config: PreprocessingConfig,
) -> PPGDataset:
    """Build dataset from CAIRE pre-segmented data."""
    
    preprocessor = PPGPreprocessor(config)
    
    LOGGER.info("Preprocessing training segments...")
    all_windows = []
    
    for i, segment in enumerate(train_segments):
        if i % 5000 == 0:
            LOGGER.info(f"Processed {i}/{len(train_segments)} segments")
        
        # Segment is already 1000 points (10 seconds at 100 Hz)
        processed = preprocessor.preprocess(segment)
        all_windows.append(processed)
    
    windows_concat = np.array(all_windows)
    
    # Binary classification mapping
    index_to_name = {0: "Healthy", 1: "Arrhythmic"}
    index_to_original_id = {0: 0, 1: 1}
    
    LOGGER.info(f"Dataset shape: {windows_concat.shape}")
    LOGGER.info(f"Labels shape: {train_labels.shape}")
    
    return PPGDataset(
        windows_concat,
        train_labels.astype(np.int64),
        index_to_name,
        index_to_original_id,
    )


def create_dataloaders(
    dataset: PPGDataset,
    batch_size: int = 128,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create train/val dataloaders from dataset."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    
    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return {"train": train_loader, "val": val_loader}
