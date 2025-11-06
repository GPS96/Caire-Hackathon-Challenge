"""End-to-end training script for multi-class arrhythmia detection."""

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

try:
    from . import data_loader
    from .dataset import PreprocessingConfig, build_dataset, create_dataloaders
    from .models import CNNATLSTM, CNNBiLSTM, TrainingArtifacts, build_feature_classifier, extract_hrv_features
except ImportError:  # pragma: no cover - fallback for script execution
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PACKAGE_ROOT.parent))

    from arrhythmia_project import data_loader
    from arrhythmia_project.dataset import PreprocessingConfig, build_dataset, create_dataloaders
    from arrhythmia_project.models import (
        CNNATLSTM,
        CNNBiLSTM,
        TrainingArtifacts,
        build_feature_classifier,
        extract_hrv_features,
    )


LOGGER = logging.getLogger(__name__)

DEEP_MODEL_FILENAME = "cnn_atlstm.pt"
LEGACY_DEEP_MODEL_FILENAME = "cnn_bilstm.pt"


@dataclass(slots=True)
class DeepTrainingConfig:
    """Hyperparameters controlling deep model optimisation."""

    epochs: int = 75
    patience: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    # scheduler_factor: float = 0.5
    # scheduler_patience: int = 3
    grad_clip_norm: float | None = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cpu")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    try:
        metrics["auc_ovo"] = float(roc_auc_score(y_true, y_prob, multi_class="ovo"))
    except ValueError:
        metrics["auc_ovo"] = float("nan")
    return metrics


def _subset_arrays(dataset, subset) -> Tuple[np.ndarray, np.ndarray]:
    indices = subset.indices  # torch.utils.data.Subset
    windows = dataset.windows[indices].numpy()
    labels = dataset.labels[indices].numpy()
    return windows, labels


def train_feature_model(
    dataset,
    dataloaders: Dict[str, DataLoader],
    sampling_rate: int,
    output_dir: Path,
) -> Dict[str, float]:
    LOGGER.info(
        "Training feature classifier on %d windows (val=%d, test=%d)",
        len(dataloaders["train"].dataset),
        len(dataloaders["val"].dataset),
        len(dataloaders["test"].dataset),
    )
    train_windows, train_labels = _subset_arrays(dataset, dataloaders["train"].dataset)
    val_windows, val_labels = _subset_arrays(dataset, dataloaders["val"].dataset)
    test_windows, test_labels = _subset_arrays(dataset, dataloaders["test"].dataset)

    fe_train = extract_hrv_features(train_windows, sampling_rate)
    fe_val = extract_hrv_features(val_windows, sampling_rate)
    fe_test = extract_hrv_features(test_windows, sampling_rate)

    model = build_feature_classifier()
    model.fit(fe_train, train_labels)

    val_pred = model.predict(fe_val)
    val_prob = model.predict_proba(fe_val)
    metrics = compute_metrics(val_labels, val_pred, val_prob)

    test_pred = model.predict(fe_test)
    test_prob = model.predict_proba(fe_test)
    metrics.update({
        "test_accuracy": float(accuracy_score(test_labels, test_pred)),
        "test_macro_f1": float(f1_score(test_labels, test_pred, average="macro")),
    })
    try:
        metrics["test_auc_ovo"] = float(roc_auc_score(test_labels, test_prob, multi_class="ovo"))
    except ValueError:
        metrics["test_auc_ovo"] = float("nan")

    label_ids = list(range(dataset.num_classes))
    target_names = [dataset.index_to_name[idx] for idx in label_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "random_forest.joblib")
    report = classification_report(
        test_labels,
        test_pred,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
    )
    (output_dir / "feature_classification_report.txt").write_text(report, encoding="utf-8")
    LOGGER.info("Feature model metrics: %s", metrics)
    return metrics


def train_deep_model(
    dataset,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    output_dir: Path,
    label_names: Dict[int, str],
    config: DeepTrainingConfig,
    class_weights: torch.Tensor,  # Add this argument
) -> Dict[str, float]:
    input_length = dataset.windows.shape[1]
    num_classes = dataset.num_classes
    model = CNNATLSTM(input_length=input_length, num_classes=num_classes).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     patience=config.scheduler_patience,
    #     factor=config.scheduler_factor,
    # )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    LOGGER.info(
        "Training deep model with %s (optimizer=%s)",
        model.__class__.__name__,
        optimizer.__class__.__name__,
    )
    LOGGER.info("Deep training hyperparameters: %s", asdict(config))

    best_loss = float("inf")
    patience_counter = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_loader_iter = tqdm(
            dataloaders["train"],
            desc=f"Epoch {epoch}/{config.epochs} [train]",
            leave=False,
        )
        for batch_x, batch_y in train_loader_iter:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(dataloaders["train"].dataset)

        val_loss, val_metrics = evaluate_model(
            model,
            dataloaders["val"],
            device,
            criterion,
            label_names,
            progress_desc=f"Epoch {epoch}/{config.epochs} [val]",
        )
        # scheduler.step(val_loss)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            state_dict = model.state_dict()
            torch.save(state_dict, output_dir / DEEP_MODEL_FILENAME)
            # Maintain legacy filename for compatibility with older tooling and tests.
            torch.save(state_dict, output_dir / LEGACY_DEEP_MODEL_FILENAME)
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

        LOGGER.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | val_macro_f1=%.3f",
            epoch,
            train_loss,
            val_loss,
            val_metrics.get("accuracy", float("nan")),
            val_metrics.get("macro_f1", float("nan")),
        )

    checkpoint_path = output_dir / DEEP_MODEL_FILENAME
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / LEGACY_DEEP_MODEL_FILENAME
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    _, test_metrics = evaluate_model(
        model,
        dataloaders["test"],
        device,
        criterion,
        label_names,
        include_report=True,
        output_dir=output_dir,
        progress_desc="Testing",
    )
    LOGGER.info("Deep model test metrics: %s", test_metrics)
    return {
        "train_loss": train_loss,
        "val_loss": best_loss,
        **val_metrics,
        **test_metrics,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    label_names: Dict[int, str],
    include_report: bool = False,
    output_dir: Path | None = None,
    progress_desc: str | None = None,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    preds: List[int] = []
    probs: List[np.ndarray] = []
    labels: List[int] = []
    iterator = tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader
    with torch.no_grad():
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            probability = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(probability)
            preds.append(np.argmax(probability, axis=1))
            labels.append(batch_y.cpu().numpy())
    total_loss /= len(loader.dataset)
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    if include_report and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        label_ids = sorted(label_names.keys())
        target_names = [label_names[idx] for idx in label_ids]
        report = classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
        )
        (output_dir / "deep_classification_report.txt").write_text(report, encoding="utf-8")
    return total_loss, metrics


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    seed: int = 42,
    device_str: str = "auto",
) -> TrainingArtifacts:
    set_seed(seed)
    records = data_loader.load_records(data_root)
    if not records:
        raise RuntimeError("No labeled records discovered. Check input data and annotations.")

    config = PreprocessingConfig(sampling_rate=125, augment=True)
    dataset = build_dataset(records, config)
    deep_config = DeepTrainingConfig()
    loaders = create_dataloaders(dataset, batch_size=deep_config.batch_size)

    device = resolve_device(device_str)
    LOGGER.info("Using device %s", device)

    label_to_records: Dict[str, List[str]] = {}
    for record in records.values():
        label_to_records.setdefault(record.label_name, []).append(record.name)
    missing_labels = set(data_loader.LABEL_ID_MAP.keys()) - set(label_to_records)
    for label_name, names in label_to_records.items():
        LOGGER.info("Discovered %d records for label '%s' (examples: %s)", len(names), label_name, names[:3])
    if missing_labels:
        LOGGER.warning("No records found for labels: %s", ", ".join(sorted(missing_labels)))

    label_counts = np.bincount(dataset.labels.numpy(), minlength=dataset.num_classes)
    # --- ADD THIS CODE ---
    # Calculate weights inversely proportional to class frequency
    class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float32)
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * dataset.num_classes
    class_weights = class_weights.to(device)

    LOGGER.info(f"Using class weights: {class_weights.cpu().numpy()}")
    # --- END OF ADDITION ---

    for new_idx, name in dataset.index_to_name.items():
        original_id = dataset.index_to_original_id[new_idx]
        LOGGER.info(
            "Class '%s' (original id=%d): %d windows",
            name,
            original_id,
            int(label_counts[new_idx]),
        )

    # device = resolve_device(device_str)
    # LOGGER.info("Using device %s", device)

    feature_metrics = train_feature_model(dataset, loaders, config.sampling_rate, output_dir)
    deep_metrics = train_deep_model(
                    dataset,
                    loaders,
                    device,
                    output_dir,
                    dataset.index_to_name,
                    config=deep_config,
                    class_weights=class_weights  # Pass the weights here
                )

    summary = {
        "feature_model": feature_metrics,
        "deep_model": deep_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    label_payload = {
        "index_to_name": dataset.index_to_name,
        "index_to_original_id": dataset.index_to_original_id,
    }
    (output_dir / "label_mapping.json").write_text(json.dumps(label_payload, indent=2), encoding="utf-8")

    checkpoint_path = output_dir / DEEP_MODEL_FILENAME
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / LEGACY_DEEP_MODEL_FILENAME
    model_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    artifacts = TrainingArtifacts(
        feature_model=joblib.load(output_dir / "random_forest.joblib"),
        deep_model_state=model_state,
        index_label_mapping=dict(dataset.index_to_name),
        index_to_original_id=dict(dataset.index_to_original_id),
    )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train arrhythmia detection models")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "training")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("weights"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for deep model training (auto, cpu, cuda, cuda:<index>)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    artifacts = run_pipeline(
        args.data_root,
        args.output,
        seed=args.seed,
        device_str=args.device,
    )
    print("Training complete. Saved artifacts to", args.output)
    print("Available labels:", artifacts.label_mapping)


if __name__ == "__main__":
    main()
