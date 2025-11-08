"""
TEST ON COMBINED DATASET (Org PPG + rPPG Test Split)
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from arrhythmia_project.models import CNNATLSTM


def test_combined_dataset(model_path: Path, device: str = "cuda"):
    """Test on combined test dataset."""
    
    device = torch.device(device)
    
    # Load combined test data
    logger.info("📂 Loading combined test dataset...")
    test_segments = np.load("domain_adapt_data_corrected/combined_test_segments.npy").astype(np.float32)
    test_labels = np.load("domain_adapt_data_corrected/combined_test_labels.npy").astype(np.int64)
    
    logger.info(f"✅ Test dataset: {test_segments.shape}")
    
    # Load model
    logger.info("🤖 Loading model...")
    model = CNNATLSTM(input_length=1000, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Inference
    logger.info("🔍 Running inference...")
    ds = TensorDataset(torch.from_numpy(test_segments).float(), 
                       torch.from_numpy(test_labels).long())
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    recall_healthy = recall_score(all_targets, all_preds, labels=[0], average=None)[0]
    recall_arr = recall_score(all_targets, all_preds, labels=[1], average=None)[0]
    f1 = f1_score(all_targets, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
    cm = confusion_matrix(all_targets, all_preds)
    
    # Report
    logger.info("\n" + "=" * 72)
    logger.info("TEST RESULTS (Combined PPG + rPPG Test Set)")
    logger.info("=" * 72)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"\nPer-Class Recall:")
    logger.info(f"  Healthy:    {recall_healthy:.4f}")
    logger.info(f"  Arrhythmic: {recall_arr:.4f}")
    logger.info("\n" + classification_report(all_targets, all_preds, 
                                             target_names=['Healthy', 'Arrhythmic']))
    logger.info("Confusion Matrix:")
    logger.info(str(cm))
    logger.info("=" * 72 + "\n")


if __name__ == "__main__":
    model_path = Path("caire_weights_robust/deep_model.pth")
    test_combined_dataset(model_path, device="cuda")
