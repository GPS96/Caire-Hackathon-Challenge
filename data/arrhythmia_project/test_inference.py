"""Evaluate trained AT-LSTM model on test dataset."""

import logging
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from . import data_loader
from .models import CNNATLSTM
from .dataset import PreprocessingConfig, PPGPreprocessor

LOGGER = logging.getLogger(__name__)


def evaluate_on_test_set(
    data_root: Path,
    model_path: Path,
    device: str = "cuda",
) -> None:
    """Evaluate trained model on test dataset."""
    
    # Load test data
    LOGGER.info("Loading test data...")
    test_segments, test_labels = data_loader.load_test_data(data_root)
    
    if test_segments is None:
        LOGGER.error("Test data not found!")
        return
    
    LOGGER.info(f"Test data shape: {test_segments.shape}, labels: {test_labels.shape}")
    
    # Preprocess test data (same as training)
    config = PreprocessingConfig(sampling_rate=100, augment=False)
    preprocessor = PPGPreprocessor(config)
    
    LOGGER.info("Preprocessing test segments...")
    all_windows = []
    for i, segment in enumerate(test_segments):
        if i % 2000 == 0:
            LOGGER.info(f"Processed {i}/{len(test_segments)} segments")
        processed = preprocessor.preprocess(segment)
        all_windows.append(processed)
    
    test_data = np.array(all_windows)
    LOGGER.info(f"Test data preprocessed: {test_data.shape}")
    
    # Load trained model
    device = torch.device(device)
    model = CNNATLSTM(input_length=1000, num_classes=2)
    model = model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    LOGGER.info(f"Model loaded from {model_path}")
    
    # Make predictions in batches
    batch_size = 128
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            if i % 2000 == 0:
                LOGGER.info(f"Processed {min(i + batch_size, len(test_data))}/{len(test_data)} test samples")
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels, all_preds)
    test_f1 = f1_score(test_labels, all_preds, average="weighted")
    test_auc = roc_auc_score(test_labels, all_probs[:, 1])
    
    # Class distribution
    unique_labels, counts = np.unique(test_labels, return_counts=True)
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Test Dataset Statistics:")
    for label, count in zip(unique_labels, counts):
        percentage = 100.0 * count / len(test_labels)
        label_name = "Healthy" if label == 0 else "Arrhythmic"
        LOGGER.info(f"  {label_name}: {count} samples ({percentage:.1f}%)")
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Test Set Performance:")
    LOGGER.info(f"  Accuracy: {test_acc:.4f}")
    LOGGER.info(f"  F1 Score: {test_f1:.4f}")
    LOGGER.info(f"  ROC-AUC: {test_auc:.4f}")
    LOGGER.info(f"\nClassification Report:")
    LOGGER.info(classification_report(test_labels, all_preds, target_names=["Healthy", "Arrhythmic"]))
    LOGGER.info(f"{'='*60}\n")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, all_preds)
    LOGGER.info(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\Caire-Hackathon-Challenge\data\data")
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("caire_weights/deep_model.pth")
    
    evaluate_on_test_set(data_root, model_path)
