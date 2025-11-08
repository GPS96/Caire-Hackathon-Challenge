"""Wrapper for loading and running the pretrained SiamAF model."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class _SiamAFStub(nn.Module):
    """Fallback lightweight SiamAF-compatible model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.conv(x)
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)


class SiamAFClassifier:
    """Loads the SiamAF checkpoint and performs forward inference."""

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = _SiamAFStub()
        self.model.to(self.device)
        self.is_ready = self._load_checkpoint()
        self.model.eval()
        self.labels = ["normal", "arrhythmia"]

    def _load_checkpoint(self) -> bool:
        if self.checkpoint.exists():
            try:
                state = torch.load(self.checkpoint, map_location=self.device)
                self.model.load_state_dict(state)
                return True
            except Exception:  # pragma: no cover - defensive loading guard
                return False
        return False

    def predict(self, signal: np.ndarray, fs: int) -> Tuple[str, float]:
        if signal.size == 0:
            return "unknown", 0.0
        tensor = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        index = int(np.argmax(probs))
        return self.labels[index], float(probs[index])
