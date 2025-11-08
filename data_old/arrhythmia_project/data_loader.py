"""Data loader for CAIRE hackathon PPG dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class CAIRERecord:
    """Container for CAIRE dataset PPG signal and label."""
    
    def __init__(self, signal: np.ndarray, label: int, segment_id: str):
        self.signal = signal
        self.label = label
        self.segment_id = segment_id
        self.sampling_rate = 100  # CAIRE uses 100 Hz


def load_caire_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CAIRE dataset from subdirectories.
    
    Expected structure:
    data_dir/
    ├── train/
    │   ├── train_segments.npy (shape: 35120, 1000)
    │   └── train_labels.npy (shape: 35120,)
    └── test/
        ├── test_segments.npy
        └── test_labels.npy
    """
    
    data_dir = Path(data_dir)
    
    # Updated to match actual file names from CAIRE dataset
    train_segments_path = data_dir / "train" / "train_segments.npy"
    train_labels_path = data_dir / "train" / "train_labels.npy"
    
    if not train_segments_path.exists():
        raise FileNotFoundError(f"Training segments not found at {train_segments_path}")
    if not train_labels_path.exists():
        raise FileNotFoundError(f"Training labels not found at {train_labels_path}")
    
    train_segments = np.load(train_segments_path)
    train_labels = np.load(train_labels_path)
    
    LOGGER.info(f"Loaded training data: {train_segments.shape}, labels: {train_labels.shape}")
    
    # Check class distribution
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = 100.0 * count / len(train_labels)
        LOGGER.info(f"  Label {label}: {count} samples ({percentage:.1f}%)")
    
    return train_segments, train_labels


def load_test_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data from CAIRE dataset."""
    
    data_dir = Path(data_dir)
    
    # Updated to match actual file names from CAIRE dataset
    test_segments_path = data_dir / "test" / "test_segments.npy"
    test_labels_path = data_dir / "test" / "test_labels.npy"
    
    if not test_segments_path.exists():
        LOGGER.warning(f"Test segments not found at {test_segments_path}")
        return None, None
    if not test_labels_path.exists():
        LOGGER.warning(f"Test labels not found at {test_labels_path}")
        return None, None
    
    test_segments = np.load(test_segments_path)
    test_labels = np.load(test_labels_path)
    
    LOGGER.info(f"Loaded test data: {test_segments.shape}, labels: {test_labels.shape}")
    
    return test_segments, test_labels
