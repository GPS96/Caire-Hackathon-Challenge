"""
FAST Domain Adaptation: rPPG → Pseudo-labeled dataset
- Resample 30 Hz rPPG to 100 Hz
- Segment into 10s windows (1000 samples)
- Auto-label with your trained AT-LSTM (pseudo-labels)
- Mix 30% rPPG + 70% original PPG
- Focal Loss will handle class imbalance during retraining
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.interpolate import interp1d
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def fast_resample_and_segment(csv_path: Path, output_dir="domain_adapt_data"):
    """Resample rPPG from 30 Hz → 100 Hz and segment into 1000-sample windows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("📥 Loading rPPG CSV...")
    df = pd.read_csv(csv_path)
    source_file = csv_path.name
    signal = df["ppg"].values.astype(np.float32)

    original_fs = 30.0
    target_fs = 100.0

    logger.info(f"📈 Resampling {len(signal)} samples: {original_fs}Hz → {target_fs}Hz...")
    time_old = np.arange(len(signal)) / original_fs
    time_new = np.arange(0, len(signal) / original_fs, 1 / target_fs)

    f = interp1d(time_old, signal, kind="linear", fill_value="extrapolate")
    resampled = f(time_new).astype(np.float32)

    logger.info(f"✅ Resampled: {len(signal)} → {len(resampled)} samples")

    window_size = 1000
    num_windows = len(resampled) // window_size
    logger.info(f"🔪 Creating {num_windows} segments of {window_size} samples...")

    segments = np.stack([resampled[i * window_size:(i + 1) * window_size] for i in range(num_windows)]).astype(np.float32)
    logger.info(f"✅ Created segments: {segments.shape}")

    metadata = pd.DataFrame({
        "segment_id": np.arange(num_windows, dtype=np.int64),
        "source_file": [source_file] * num_windows,
        "label": [-1] * num_windows,
    })

    return segments, metadata, output_dir


def fast_autolabel(segments: np.ndarray, model_path: Path, device: str = "cuda"):
    """Auto-label rPPG segments using trained model (ALL segments, no filtering)."""
    from arrhythmia_project.models import CNNATLSTM

    logger.info("🤖 Loading trained AT-LSTM model...")
    model = CNNATLSTM(input_length=1000, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    logger.info(f"📊 Auto-labeling {len(segments)} segments...")

    all_labels = []
    all_probs = []
    batch_size = 128

    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            x = torch.from_numpy(batch).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_labels.extend(preds)
            all_probs.extend(probs.cpu().numpy())

    labels = np.array(all_labels, dtype=np.int64)
    probs = np.array(all_probs, dtype=np.float32)

    u, c = np.unique(labels, return_counts=True)
    count_map = {int(k): int(v) for k, v in zip(u, c)}
    logger.info(f"✅ Pseudo-labels: {count_map}")

    return labels, probs


def save_rppg_dataset_like_organizers(output_dir: Path, segments: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame):
    """Save rPPG dataset in organizer's format."""
    out_segments = output_dir / "rppg_train_segments.npy"
    out_labels = output_dir / "rppg_train_labels.npy"
    out_meta = output_dir / "rppg_train_metadata.csv"

    metadata = metadata.copy()
    metadata["label"] = labels.astype(np.int64)

    np.save(out_segments, segments.astype(np.float32))
    np.save(out_labels, labels.astype(np.int64))
    metadata.to_csv(out_meta, index=False)

    logger.info(f"💾 Saved rPPG dataset:")
    logger.info(f"  {out_segments}")
    logger.info(f"  {out_labels}")
    logger.info(f"  {out_meta}")


def fast_merge_with_original(original_train_path: Path,
                             new_segments: np.ndarray,
                             new_labels: np.ndarray,
                             output_dir="domain_adapt_data",
                             mix_ratio=0.30):
    """Merge rPPG data with original PPG training data (30% rPPG + 70% PPG)."""
    output_dir = Path(output_dir)

    logger.info("📂 Loading original training data...")
    orig_segments = np.load(original_train_path / "train_segments.npy")
    orig_labels = np.load(original_train_path / "train_labels.npy")

    num_new = int(len(new_segments) * mix_ratio)
    new_segments_subset = new_segments[:num_new]
    new_labels_subset = new_labels[:num_new]

    logger.info(f"\n📊 Creating mixed dataset:")
    logger.info(f"   Original PPG: {len(orig_segments)} samples ({100-mix_ratio*100:.0f}%)")
    logger.info(f"   rPPG (auto-labeled): {num_new} samples ({mix_ratio*100:.0f}%)")
    logger.info(f"   Total: {len(orig_segments) + num_new} samples")

    merged_segments = np.vstack([orig_segments, new_segments_subset]).astype(np.float32)
    merged_labels = np.hstack([orig_labels, new_labels_subset]).astype(np.int64)

    # Shuffle
    idx = np.random.permutation(len(merged_labels))
    merged_segments = merged_segments[idx]
    merged_labels = merged_labels[idx]

    # Class distribution
    u, c = np.unique(merged_labels, return_counts=True)
    for k, v in zip(u, c):
        pct = 100.0 * v / len(merged_labels)
        label_name = "Healthy" if k == 0 else "Arrhythmic"
        logger.info(f"   {label_name}: {v} ({pct:.1f}%)")

    np.save(output_dir / "merged_train_segments.npy", merged_segments)
    np.save(output_dir / "merged_train_labels.npy", merged_labels)

    logger.info(f"\n✅ Merged dataset saved:")
    logger.info(f"  {output_dir / 'merged_train_segments.npy'}")
    logger.info(f"  {output_dir / 'merged_train_labels.npy'}\n")

    return merged_segments, merged_labels


if __name__ == "__main__":
    csv_path = Path("live_ppg.csv")
    model_path = Path("caire_weights/deep_model.pth")
    original_train_path = Path(r"D:\Caire-Hackathon-Challenge\data\data\train")

    logger.info("\n" + "=" * 72)
    logger.info("🚀 DOMAIN ADAPTATION (30% rPPG + 70% PPG)")
    logger.info("=" * 72 + "\n")

    # Step 1: Resample & segment
    segments, metadata, out_dir = fast_resample_and_segment(csv_path, "domain_adapt_data")

    # Step 2: Auto-label (ALL segments)
    labels, probs = fast_autolabel(segments, model_path, device="cuda")

    # Step 3: Save rPPG dataset
    save_rppg_dataset_like_organizers(out_dir, segments, labels, metadata)

    # Step 4: Merge with original (30% rPPG)
    fast_merge_with_original(original_train_path, segments, labels, out_dir, mix_ratio=0.30)

    logger.info("=" * 72)
    logger.info("✅ READY FOR RETRAINING WITH FOCAL LOSS")
    logger.info("=" * 72)
    logger.info(f"Run:  python -m arrhythmia_project.domain_adapt_retrain\n")
