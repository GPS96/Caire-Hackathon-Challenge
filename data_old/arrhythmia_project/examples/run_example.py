"""Minimal example showing how to train and run inference."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from arrhythmia_project import data_loader
from arrhythmia_project.inference import predict
from arrhythmia_project.train import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "training"
WEIGHTS_DIR = ROOT / "weights"


def main() -> None:
    artifacts = run_pipeline(DATA_ROOT, WEIGHTS_DIR)
    records = data_loader.load_records(DATA_ROOT)
    sample = random.choice(list(records.values()))
    label_name = artifacts.original_label_mapping.get(sample.label_id, "unknown")
    print(f"Running inference on sample {sample.name} ({label_name})")
    results = predict(sample.signal, sample.sampling_rate, WEIGHTS_DIR, plot=False)
    for model_name, payload in results.items():
        print(f"{model_name}: {payload['label_name']} (confidence={payload['confidence']:.2f})")


if __name__ == "__main__":
    main()
