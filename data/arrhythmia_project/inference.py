"""Inference helpers for multi-class arrhythmia detection models."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data_loader import LABEL_ID_MAP, PPGRecord
from .dataset import PPGPreprocessor, PreprocessingConfig, build_dataset
from . import data_loader as dl
from .models import CNNATLSTM, extract_hrv_features

DEEP_MODEL_FILENAME = "cnn_atlstm.pt"
LEGACY_DEEP_MODEL_FILENAME = "cnn_bilstm.pt"


def _load_label_mapping(weights_dir: Path) -> Dict[int, str]:
    mapping_path = weights_dir / "label_mapping.json"
    if mapping_path.exists():
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        raw_mapping = data.get("index_to_name", {})
        mapping: Dict[int, str] = {int(idx): name for idx, name in raw_mapping.items()}
        if mapping:
            return mapping
    default_mapping = {idx: name for name, idx in LABEL_ID_MAP.items()}
    return default_mapping


def load_models(
    weights_dir: Path,
    input_length: int,
    device: torch.device,
    label_mapping: Dict[int, str],
) -> Tuple[torch.nn.Module, joblib.BaseEstimator]:
    rf_model = joblib.load(weights_dir / "random_forest.joblib")
    checkpoint_path = weights_dir / DEEP_MODEL_FILENAME
    if not checkpoint_path.exists():
        checkpoint_path = weights_dir / LEGACY_DEEP_MODEL_FILENAME
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    deep_model = CNNATLSTM(input_length=input_length, num_classes=len(label_mapping))
    deep_model.load_state_dict(state_dict)
    deep_model.to(device)
    deep_model.eval()
    return deep_model, rf_model


def predict(
    signal: np.ndarray,
    sampling_rate: int,
    weights_dir: Path,
    config: PreprocessingConfig | None = None,
    device: torch.device | None = None,
    plot: bool = False,
) -> Dict[str, Dict[str, float]]:
    if config is None:
        config = PreprocessingConfig(sampling_rate=sampling_rate)
    preprocessor = PPGPreprocessor(config)
    record = PPGRecord(name="input", signal=signal, sampling_rate=sampling_rate, label_name="Normal", label_id=0)
    windows, _ = preprocessor.prepare_record(record)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_mapping = _load_label_mapping(weights_dir)
    deep_model, rf_model = load_models(weights_dir, windows.shape[1], device, label_mapping)

    rf_features = extract_hrv_features(windows, config.sampling_rate)
    rf_probs = rf_model.predict_proba(rf_features)
    rf_mean = rf_probs.mean(axis=0)

    tensor = torch.from_numpy(windows).float().to(device)
    with torch.no_grad():
        logits = deep_model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().mean(axis=0)

    predictions = {
        "feature_model": _format_prediction(rf_mean, label_mapping),
        "deep_model": _format_prediction(probs, label_mapping),
    }

    if plot:
        _plot_signal(signal, config)

    return predictions


# NOTE: Streaming mode for demo/simulation
# This loop opens a live camera feed but, on the backend, selects random PPG windows
# from the training+validation dataset to run predictions. Replace the sampling logic
# with real-time PPG extraction when the rPPG pipeline is ready.
def stream_with_dataset_predictions(
    weights_dir: Path,
    data_root: Path,
    device: torch.device | None = None,
    camera_index: int = 0,
    fps_limit: float = 10.0,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset once and cache windows for random sampling
    records = dl.load_records(data_root)
    if not records:
        raise RuntimeError(f"No paired records found under {data_root}")
    prep = PreprocessingConfig(sampling_rate=125, augment=False)
    dataset = build_dataset(records, prep)
    windows = dataset.windows.numpy()  # shape: (N, T)

    # Load models once
    label_mapping = _load_label_mapping(weights_dir)
    deep_model, rf_model = load_models(weights_dir, windows.shape[1], device, label_mapping)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera device")

    last_time = 0.0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Terminate once video stopped coming
                break

            # Throttle processing to roughly fps_limit
            now = time.time()
            if fps_limit > 0 and (now - last_time) < (1.0 / fps_limit):
                # Show frame without inference to keep UI responsive
                cv2.imshow("PPG Stream (demo)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_time = now

            # --- DEMO ONLY: sample a random window from dataset ---
            # TODO: Replace this with real-time PPG produced from the frame
            idx = random.randrange(0, windows.shape[0])
            win = windows[idx:idx+1]  # shape (1, T)

            # Feature model
            rf_features = extract_hrv_features(win, prep.sampling_rate)
            rf_probs = rf_model.predict_proba(rf_features)[0]

            # Deep model
            tensor = torch.from_numpy(win).float().to(device)
            with torch.no_grad():
                logits = deep_model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Format prediction
            deep_pred = _format_prediction(probs, label_mapping)
            rf_pred = _format_prediction(rf_probs, label_mapping)

            # TODO: Hook this into your dashboard pipeline. For now we overlay on the frame.
            text = f"Deep: {deep_pred['label_name']} ({deep_pred['confidence']:.2f}) | RF: {rf_pred['label_name']} ({rf_pred['confidence']:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2, cv2.LINE_AA)
            cv2.imshow("PPG Stream (demo)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _format_prediction(probabilities: np.ndarray, label_mapping: Dict[int, str]) -> Dict[str, float]:
    best_idx = int(np.argmax(probabilities))
    return {
        "label_id": best_idx,
        "label_name": label_mapping.get(best_idx, f"class_{best_idx}"),
        "confidence": float(probabilities[best_idx]),
    }


def _plot_signal(signal: np.ndarray, config: PreprocessingConfig) -> None:
    duration = len(signal) / config.sampling_rate
    time_axis = np.linspace(0, duration, len(signal))
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal, color="#2563eb")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("PPG Waveform")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a PPG waveform")
    parser.add_argument("input", nargs="?", type=Path, help="Path to numpy .npy file containing a PPG waveform")
    parser.add_argument("--sampling-rate", type=int, default=125)
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().with_name("weights"))
    parser.add_argument("--plot", action="store_true")
    # Streaming/demo options
    parser.add_argument("--stream", action="store_true", help="Open a live camera and run simulated predictions")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "training", help="Data root used to sample PPG windows in stream mode")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0, help="Max FPS for inference overlay in stream mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stream:
        # DEMO MODE: camera input + dataset-based predictions (replace with real rPPG later)
        stream_with_dataset_predictions(
            weights_dir=args.weights,
            data_root=args.data_root,
            camera_index=args.camera_index,
            fps_limit=args.fps,
        )
    else:
        if args.input is None:
            raise SystemExit("--stream not set and no input file provided")
        signal = np.load(args.input)
        predictions = predict(signal, args.sampling_rate, args.weights, plot=args.plot)
        for model_name, result in predictions.items():
            print(f"{model_name}: {result['label_name']} (confidence={result['confidence']:.2f})")


if __name__ == "__main__":
    main()
