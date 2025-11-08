"""
DOMAIN-ADAPTED RETRAINING: 1.5x Arrhythmia Weight
- Load from domain_adapt_data/merged_train_*.npy (30% rPPG + 70% PPG)
- Asymmetric weights: 1.5x for arrhythmia
- Target: 71-74% accuracy with balanced recalls
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import logging
from sklearn.metrics import accuracy_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

from arrhythmia_project.models import CNNATLSTM


def fast_retrain_safety_critical(merged_segments_path: Path,
                                 merged_labels_path: Path,
                                 pretrained_model_path: Path,
                                 output_dir="caire_weights_robust",
                                 epochs=15,
                                 batch_size=64,
                                 lr=5e-5,
                                 device="cuda"):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load merged dataset
    logger.info("📂 Loading merged dataset...")
    segments = np.load(merged_segments_path).astype(np.float32)
    labels = np.load(merged_labels_path).astype(np.int64)

    logger.info(f"📊 Dataset: {segments.shape}, Labels: {labels.shape}")
    u, c = np.unique(labels, return_counts=True)
    for k, v in zip(u, c):
        pct = 100.0 * v / len(labels)
        label_name = "Healthy" if k == 0 else "Arrhythmic"
        logger.info(f"   {label_name}: {v} ({pct:.1f}%)")

    # Torch dataset
    X = torch.from_numpy(segments).float()
    y = torch.from_numpy(labels).long()
    ds = TensorDataset(X, y)

    # Split train/val
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Load pretrained model
    logger.info("🤖 Loading pretrained base model...")
    device = torch.device(device)
    model = CNNATLSTM(input_length=1000, num_classes=2)
    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # Optimizer/scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ASYMMETRIC CLASS WEIGHTS: 1.5x for arrhythmia
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2
    class_weights[1] *= 1.5  # Arrhythmia gets 1.5x weight
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger.info(f"\n⚡ Finetuning with DOMAIN-ADAPTED WEIGHTS:")
    logger.info(f"   epochs={epochs}, lr={lr}, batch={batch_size}")
    logger.info(f"   Class weights: Healthy={class_weights[0]:.3f}, Arrhythmic={class_weights[1]:.3f}")
    logger.info(f"   Asymmetric: 1.5x for arrhythmia (balanced approach)")
    logger.info("=" * 72)

    best_val_arr_recall = 0.0
    best_state = None
    patience = 5
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_preds, tr_tgts = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item()
            tr_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            tr_tgts.extend(yb.detach().cpu().numpy())

        tr_loss /= len(train_loader)
        tr_acc = accuracy_score(tr_tgts, tr_preds)

        # Val
        model.eval()
        va_loss = 0.0
        va_preds, va_tgts = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item()
                va_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                va_tgts.extend(yb.detach().cpu().numpy())

        va_loss /= len(val_loader)
        va_acc = accuracy_score(va_tgts, va_preds)
        va_recall_healthy = recall_score(va_tgts, va_preds, labels=[0], average=None)[0]
        va_recall_arr = recall_score(va_tgts, va_preds, labels=[1], average=None)[0]

        logger.info(f"Epoch {epoch+1} | Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | Val: acc={va_acc:.4f} | Recall: H={va_recall_healthy:.3f}, A={va_recall_arr:.3f}")
        scheduler.step()

        if va_recall_arr > best_val_arr_recall:
            best_val_arr_recall = va_recall_arr
            best_state = model.state_dict().copy()
            patience_ctr = 0
            logger.info(f"   ✅ New best arrhythmia recall: {best_val_arr_recall:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break

    torch.save(best_state, output_dir / "deep_model.pth")
    logger.info(f"\n✅ Saved domain-adapted model to {output_dir}/deep_model.pth")
    logger.info(f"   Best Val Arrhythmia Recall: {best_val_arr_recall:.4f}\n")


if __name__ == "__main__":
    fast_retrain_safety_critical(
        merged_segments_path=Path("domain_adapt_data/merged_train_segments.npy"),
        merged_labels_path=Path("domain_adapt_data/merged_train_labels.npy"),
        pretrained_model_path=Path("caire_weights/deep_model.pth"),
        output_dir="caire_weights_robust",
        epochs=15,
        batch_size=64,
        lr=5e-5,
        device="cuda",
    )
