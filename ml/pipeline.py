"""Core rPPG and arrhythmia inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Sequence

import cv2
import numpy as np

from .signals import compute_heart_rate_and_ibi, estimate_signal_quality, filter_bandpass
from .siamaf import SiamAFClassifier

try:  # pragma: no cover - optional dependency at runtime
    from rppg_toolbox.utils.face_detector import OpenCVDetector
    from rppg_toolbox.utils.preprocessing import crop_face
except ImportError:  # pragma: no cover
    OpenCVDetector = None  # type: ignore[assignment]
    crop_face = None  # type: ignore[assignment]

from collections import deque


@dataclass(slots=True)
class SignalSample:
    timestamp: datetime
    value: float


@dataclass(slots=True)
class PipelineResult:
    heart_rate_bpm: float
    ibi_ms: List[float]
    arrhythmia_state: str
    confidence: float
    signal_quality: float
    status: str
    waveform: Sequence[SignalSample]


@dataclass(slots=True)
class PipelineConfig:
    heartbeat_window_seconds: int = 15
    frame_rate: int = 30
    model_checkpoint: str = "ml/models/siamaf_pretrained.pt"
    device: str = "cuda"
    bandpass_low_hz: float = 0.7
    bandpass_high_hz: float = 4.0


class RPPGInferencePipeline:
    """Extracts rPPG from frames and predicts arrhythmia states."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._model = SiamAFClassifier(config.model_checkpoint, device=config.device)
        self._frame_detector = OpenCVDetector() if OpenCVDetector else None
        self._buffer: Deque[SignalSample] = deque(maxlen=config.heartbeat_window_seconds * config.frame_rate)
        self.model_ready = self._model.is_ready
        self.device = self._model.device

    def process_frame(self, frame_bgr: np.ndarray) -> PipelineResult:
        if frame_bgr is None or not frame_bgr.size:
            raise ValueError("Empty frame provided")

        roi = self._extract_face_roi(frame_bgr)
        sample = self._sample_signal(roi)
        self._buffer.append(sample)

        cleaned_signal = self._filter_signal()
        try:
            heart_rate, ibi = compute_heart_rate_and_ibi(
                cleaned_signal, fs=self.config.frame_rate
            )
        except ValueError:
            heart_rate, ibi = 0.0, []
        quality = estimate_signal_quality(cleaned_signal)
        arr_state, confidence = self._model.predict(cleaned_signal, fs=self.config.frame_rate)
        status = self._status_from_confidence(arr_state, confidence)

        waveform = self._build_waveform(cleaned_signal)
        return PipelineResult(
            heart_rate_bpm=heart_rate,
            ibi_ms=ibi,
            arrhythmia_state=arr_state,
            confidence=confidence,
            signal_quality=quality,
            status=status,
            waveform=waveform,
        )

    def _extract_face_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._frame_detector and crop_face:
            detections = self._frame_detector.detect(frame_bgr)
            if detections:
                x, y, w, h = detections[0]
                return crop_face(frame_bgr, (x, y, w, h))
        return frame_bgr

    def _sample_signal(self, roi: np.ndarray) -> SignalSample:
        if roi.ndim == 3:
            green_channel = roi[:, :, 1]
        else:
            green_channel = roi
        intensity = float(np.mean(green_channel))
        return SignalSample(timestamp=datetime.utcnow(), value=intensity)

    def _filter_signal(self) -> np.ndarray:
        values = np.array([item.value for item in self._buffer], dtype=np.float32)
        if values.size < self.config.frame_rate * 3:
            return values
        return filter_bandpass(
            values,
            fs=self.config.frame_rate,
            low_hz=self.config.bandpass_low_hz,
            high_hz=self.config.bandpass_high_hz,
        )

    @staticmethod
    def _status_from_confidence(arr_state: str, confidence: float) -> str:
        if confidence > 0.85:
            return "Emergency" if arr_state != "normal" else "Info"
        if confidence > 0.6:
            return "Caution"
        return "Info"

    def _build_waveform(self, cleaned_signal: np.ndarray) -> Sequence[SignalSample]:
        samples = list(self._buffer)
        if cleaned_signal.size == 0 or not samples:
            return samples
        if cleaned_signal.size != len(samples):
            diff = len(samples) - cleaned_signal.size
            samples = samples[max(diff, 0) :]
        trimmed_values = cleaned_signal[-len(samples) :]
        waveform: list[SignalSample] = []
        for sample, value in zip(samples, trimmed_values, strict=True):
            waveform.append(SignalSample(timestamp=sample.timestamp, value=float(value)))
        return waveform[-self.config.frame_rate * 5 :]
