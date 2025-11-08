"""
PROPER DOMAIN ADAPTATION WITH CORRECT TRAIN-TEST SPLIT
- Resample rPPG: 30 Hz → 100 Hz
- Split rPPG using ORGANIZERS' RATIO
- Combine: Org_Train + rPPG_Train, Org_Test + rPPG_Test
- Train on combined, Test on combined (NO MISMATCH)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.interpolate import interp1d
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def resample_rppg_30_to_100hz(csv_path: Path):
    """Resample rPPG from 30 Hz → 100 Hz."""
    logger.info("📥 Loading and resampling rPPG...")
    df = pd.read_csv(csv_path)
    signal = df["ppg"].values.astype(np.float32)
    
    # Resample 30 Hz → 100 Hz
    time_old = np.arange(len(signal)) / 30.0
    time_new = np.arange(0, len(signal) / 30.0, 1 / 100.0)
    f = interp1d(time_old, signal, kind="linear", fill_value="extrapolate")
    resampled = f(time_new).astype(np.float32)
    
    logger.info(f"✅ Resampled: {len(signal)} → {len(resampled)} samples")
    return resampled


def segment_signal(signal: np.ndarray, window_size: int = 1000):
    """Segment into 1000-sample windows."""
    num_windows = len(signal) // window_size
    segments = np.stack([signal[i * window_size:(i + 1) * window_size] for i in range(num_windows)]).astype(np.float32)
    logger.info(f"✅ Segmented: {num_windows} windows of {window_size} samples")
    return segments


def get_organizer_split_ratio(train_dir: Path, test_dir: Path):
    """Get train/test ratio from organizers' data."""
    train_segments = np.load(train_dir / "train_segments.npy")
    test_segments = np.load(test_dir / "test_segments.npy")
    
    num_train = len(train_segments)
    num_test = len(test_segments)
    total = num_train + num_test
    
    train_ratio = num_train / total
    test_ratio = num_test / total
    
    logger.info(f"\n📊 Organizers' split ratio:")
    logger.info(f"   Train: {train_ratio:.4f} ({num_train} samples)")
    logger.info(f"   Test:  {test_ratio:.4f} ({num_test} samples)")
    
    return train_ratio, test_ratio


def create_combined_train_test(train_dir: Path, test_dir: Path, rppg_segments: np.ndarray):
    """Create combined train/test using organizers' ratio."""
    
    logger.info("📂 Loading organizers' data...")
    org_train_segments = np.load(train_dir / "train_segments.npy").astype(np.float32)
    org_train_labels = np.load(train_dir / "train_labels.npy").astype(np.int64)
    org_test_segments = np.load(test_dir / "test_segments.npy").astype(np.float32)
    org_test_labels = np.load(test_dir / "test_labels.npy").astype(np.int64)
    
    # Get split ratio
    train_ratio, test_ratio = get_organizer_split_ratio(train_dir, test_dir)
    
    # Split rPPG using SAME ratio
    num_rppg_train = int(len(rppg_segments) * train_ratio)
    rppg_train_segments = rppg_segments[:num_rppg_train]
    rppg_test_segments = rppg_segments[num_rppg_train:]
    
    logger.info(f"\n✂️ Split rPPG using organizers' ratio:")
    logger.info(f"   rPPG Train: {len(rppg_train_segments)} samples")
    logger.info(f"   rPPG Test:  {len(rppg_test_segments)} samples")
    
    # We need pseudo-labels for rPPG (use original trained model)
    logger.info("\n🤖 Auto-labeling rPPG segments...")
    from arrhythmia_project.models import CNNATLSTM
    
    device = torch.device("cuda")
    model = CNNATLSTM(input_length=1000, num_classes=2)
    model_path = Path("caire_weights/deep_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Label all rPPG
    all_rppg_labels = []
    with torch.no_grad():
        for i in range(0, len(rppg_segments), 128):
            batch = torch.from_numpy(rppg_segments[i:i+128]).float().to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_rppg_labels.extend(preds)
    
    rppg_train_labels = np.array(all_rppg_labels[:num_rppg_train], dtype=np.int64)
    rppg_test_labels = np.array(all_rppg_labels[num_rppg_train:], dtype=np.int64)
    
    # COMBINE: Org + rPPG
    combined_train_segments = np.vstack([org_train_segments, rppg_train_segments]).astype(np.float32)
    combined_train_labels = np.hstack([org_train_labels, rppg_train_labels]).astype(np.int64)
    
    combined_test_segments = np.vstack([org_test_segments, rppg_test_segments]).astype(np.float32)
    combined_test_labels = np.hstack([org_test_labels, rppg_test_labels]).astype(np.int64)
    
    # Shuffle
    idx_train = np.random.permutation(len(combined_train_labels))
    combined_train_segments = combined_train_segments[idx_train]
    combined_train_labels = combined_train_labels[idx_train]
    
    idx_test = np.random.permutation(len(combined_test_labels))
    combined_test_segments = combined_test_segments[idx_test]
    combined_test_labels = combined_test_labels[idx_test]
    
    # Report
    logger.info(f"\n📊 COMBINED TRAIN SET:")
    logger.info(f"   Org PPG: {len(org_train_segments)}")
    logger.info(f"   rPPG: {len(rppg_train_segments)}")
    logger.info(f"   Total: {len(combined_train_segments)}")
    u, c = np.unique(combined_train_labels, return_counts=True)
    for k, v in zip(u, c):
        pct = 100.0 * v / len(combined_train_labels)
        logger.info(f"   {'Healthy' if k==0 else 'Arrhythmic'}: {v} ({pct:.1f}%)")
    
    logger.info(f"\n📊 COMBINED TEST SET:")
    logger.info(f"   Org PPG: {len(org_test_segments)}")
    logger.info(f"   rPPG: {len(rppg_test_segments)}")
    logger.info(f"   Total: {len(combined_test_segments)}")
    u, c = np.unique(combined_test_labels, return_counts=True)
    for k, v in zip(u, c):
        pct = 100.0 * v / len(combined_test_labels)
        logger.info(f"   {'Healthy' if k==0 else 'Arrhythmic'}: {v} ({pct:.1f}%)")
    
    # Save
    out_dir = Path("domain_adapt_data_corrected")
    out_dir.mkdir(exist_ok=True)
    
    np.save(out_dir / "combined_train_segments.npy", combined_train_segments)
    np.save(out_dir / "combined_train_labels.npy", combined_train_labels)
    np.save(out_dir / "combined_test_segments.npy", combined_test_segments)
    np.save(out_dir / "combined_test_labels.npy", combined_test_labels)
    
    logger.info(f"\n✅ Saved to {out_dir}/")


if __name__ == "__main__":
    train_dir = Path(r"D:\Caire-Hackathon-Challenge\data\data\train")
    test_dir = Path(r"D:\Caire-Hackathon-Challenge\data\data\test")
    rppg_csv = Path("live_ppg.csv")
    
    logger.info("\n" + "=" * 72)
    logger.info("CREATING PROPERLY SPLIT COMBINED DATASET")
    logger.info("=" * 72 + "\n")
    
    # Resample rPPG
    rppg_resampled = resample_rppg_30_to_100hz(rppg_csv)
    
    # Segment
    rppg_segments = segment_signal(rppg_resampled, window_size=1000)
    
    # Create combined train/test with correct split
    create_combined_train_test(train_dir, test_dir, rppg_segments)
    
    logger.info("=" * 72)
    logger.info("✅ DONE! Use the combined dataset for training and testing")
    logger.info("=" * 72 + "\n")
