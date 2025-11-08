"""End-to-end training script for CAIRE arrhythmia detection."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import data_loader
from .dataset import PreprocessingConfig, build_dataset_caire, create_dataloaders
from .models import CNNATLSTM, TrainingArtifacts

LOGGER = logging.getLogger(__name__)

DEEP_MODEL_FILENAME = "deep_model.pth"
FEATURE_MODEL_FILENAME = "feature_classifier.pkl"


@dataclass(slots=True)
class DeepTrainingConfig:
    """Configuration for deep model training."""
    epochs: int = 75
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def print_gpu_info(device: torch.device) -> None:
    """Print GPU information."""
    print(f"🎮 Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU Available: True")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {total_mem:.2f} GB")


def train_deep_model(
    dataset,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    output_dir: Path,
    label_names: Dict[int, str],
    config: DeepTrainingConfig,
    class_weights: torch.Tensor = None,
) -> Dict:
    """Train AT-LSTM model."""
    
    model = CNNATLSTM(input_length=1000, num_classes=dataset.num_classes)
    model = model.to(device)
    
    LOGGER.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    training_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            #batch_x = batch_x.to(device).unsqueeze(1)  # Add channel dimension
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
            
            avg_loss = train_loss / (pbar.n + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})


        
        train_loss /= len(loaders["train"])
        train_acc = accuracy_score(train_targets, train_preds)
        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            pbar = tqdm(loaders["val"], desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(device)

                #batch_x = batch_x.to(device).unsqueeze(1)
                batch_y = batch_y.to(device)
                
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
                val_targets.extend(batch_y.detach().cpu().numpy())
                
                avg_loss = val_loss / (pbar.n + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        
        val_loss /= len(loaders["val"])
        val_acc = accuracy_score(val_targets, val_preds)
        training_history["val_loss"].append(val_loss)
        training_history["val_acc"].append(val_acc)
        
        LOGGER.info(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            LOGGER.info(f"✓ New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            LOGGER.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save best model
    if best_model_state is not None:
        checkpoint_path = output_dir / DEEP_MODEL_FILENAME
        torch.save(best_model_state, checkpoint_path)
        LOGGER.info(f"Saved best model to {checkpoint_path}")
    
    # Compute final metrics on validation set
    model.load_state_dict(best_model_state)
    model.eval()
    final_preds = []
    final_targets = []
    final_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in loaders["val"]:
            batch_x = batch_x.to(device)

            #batch_x = batch_x.to(device).unsqueeze(1)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            
            final_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            final_probs.extend(probs.detach().cpu().numpy())
            final_targets.extend(batch_y.detach().cpu().numpy())
    
    final_preds = np.array(final_preds)
    final_targets = np.array(final_targets)
    final_probs = np.array(final_probs)
    
    val_acc = accuracy_score(final_targets, final_preds)
    val_f1 = f1_score(final_targets, final_preds, average="weighted")
    
    if dataset.num_classes == 2:
        val_auc = roc_auc_score(final_targets, final_probs[:, 1])
    else:
        val_auc = roc_auc_score(final_targets, final_probs, multi_class="ovr", average="weighted")
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Final Validation Results:")
    LOGGER.info(f"  Accuracy: {val_acc:.4f}")
    LOGGER.info(f"  F1 Score: {val_f1:.4f}")
    LOGGER.info(f"  ROC-AUC: {val_auc:.4f}")
    LOGGER.info(f"\nClassification Report:\n{classification_report(final_targets, final_preds, target_names=[label_names[i] for i in range(dataset.num_classes)])}")
    LOGGER.info(f"{'='*60}\n")
    
    return {
        "accuracy": float(val_acc),
        "f1_score": float(val_f1),
        "roc_auc": float(val_auc),
        "training_history": training_history,
    }


def run_pipeline_caire(
    data_root: Path,
    output_dir: Path,
    seed: int = 42,
    device_str: str = "auto",
) -> TrainingArtifacts:
    """Training pipeline for CAIRE dataset."""
    
    set_seed(seed)
    
    # Load CAIRE data
    train_segments, train_labels = data_loader.load_caire_dataset(data_root)
    
    # Build dataset
    config = PreprocessingConfig(
        sampling_rate=100,
        augment=True,
    )
    dataset = build_dataset_caire(train_segments, train_labels, config)
    
    # Create dataloaders
    deep_config = DeepTrainingConfig(
        epochs=75,
        batch_size=64,
        lr=1e-4,
    )
    loaders = create_dataloaders(dataset, batch_size=deep_config.batch_size)
    
    # Resolve device
    device = resolve_device(device_str)
    print_gpu_info(device)
    
    # Calculate class weights for imbalanced data
    label_counts = np.bincount(dataset.labels.numpy(), minlength=dataset.num_classes)
    class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * dataset.num_classes
    class_weights = class_weights.to(device)
    
    LOGGER.info(f"Class distribution: {label_counts}")
    LOGGER.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Train model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    deep_metrics = train_deep_model(
        dataset,
        loaders,
        device,
        output_dir,
        dataset.index_to_name,
        config=deep_config,
        class_weights=class_weights,
    )
    
    # Save metadata
    metrics_summary = {"deep_model": deep_metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )
    
    label_payload = {
        "index_to_name": dataset.index_to_name,
        "index_to_original_id": dataset.index_to_original_id,
    }
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_payload, indent=2), encoding="utf-8"
    )
    
    # Load best model
    checkpoint_path = output_dir / DEEP_MODEL_FILENAME
    model_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    artifacts = TrainingArtifacts(
        feature_model=None,
        deep_model_state=model_state,
        index_label_mapping=dict(dataset.index_to_name),
        index_to_original_id=dict(dataset.index_to_original_id),
    )
    
    return artifacts


def create_dataloaders(dataset, batch_size: int = 128, val_split: float = 0.2, num_workers: int = 0, seed: int = 42) -> Dict[str, DataLoader]:
    """Create train/val dataloaders from dataset."""
    from .dataset import create_dataloaders as create_dataloaders_impl
    return create_dataloaders_impl(dataset, batch_size=batch_size, val_split=val_split, num_workers=num_workers, seed=seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AT-LSTM arrhythmia detection model")
    
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Caire-Hackathon-Challenge\data\data"),
        help="Path to CAIRE dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./caire_weights"),
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda, cuda:0)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    args = parse_args()
    
    LOGGER.info(f"Data root: {args.data_root}")
    LOGGER.info(f"Output directory: {args.output}")
    LOGGER.info(f"Device: {args.device}")
    LOGGER.info(f"Seed: {args.seed}")
    
    artifacts = run_pipeline_caire(
        args.data_root,
        args.output,
        seed=args.seed,
        device_str=args.device,
    )
    
    print(f"\n✅ Training complete!")
    print(f"📦 Saved to: {args.output}")
    print(f"🏷️  Labels: {artifacts.index_label_mapping}")


if __name__ == "__main__":
    main()