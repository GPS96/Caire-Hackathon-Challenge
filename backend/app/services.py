"""Service layer bridging FastAPI and the ML pipeline.

Supports two modes:
- Normal: Use RPPGInferencePipeline on incoming frames.
- Simulation: Ignore frames and sample PPG windows from the dataset to produce predictions
    with the arrhythmia_project models. This is useful until real-time rPPG is finalized.
"""

import asyncio
import base64
from pathlib import Path
from typing import AsyncIterator, Optional
import importlib
import sys

import cv2
import numpy as np
import torch

from .config import get_settings
from .models import ArrhythmiaPush, InferRequest, InferResponse, SignalSample

# Signal processing utilities for deriving HR/IBI/signal quality and a cleaned waveform
try:
    from ml.signals import (
        filter_bandpass,
        compute_heart_rate_and_ibi,
        estimate_signal_quality,
        estimate_signal_quality_psd,
    )
except Exception:  # pragma: no cover - optional during early bootstrap
    filter_bandpass = None
    compute_heart_rate_and_ibi = None
    estimate_signal_quality = None
    estimate_signal_quality_psd = None

try:
    from ml.pipeline import PipelineConfig, PipelineResult, RPPGInferencePipeline
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "ML pipeline package is not importable. Ensure the repository root is on PYTHONPATH."
    ) from exc


class PipelineService:
    """Coordinates synchronous inference requests and websocket broadcasts."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._pipeline = RPPGInferencePipeline(
            PipelineConfig(
                heartbeat_window_seconds=settings.heartbeat_window_seconds,
                model_checkpoint=settings.model_checkpoint_path,
                device=settings.device,
            )
        )
        self._subscribers: set["asyncio.Queue[ArrhythmiaPush]"] = set()
        # Simulation resources
        self._sim_ready = False
        self._sim_windows: Optional[np.ndarray] = None
        self._sim_labels: Optional[np.ndarray] = None
        self._sim_pool_normal: Optional[np.ndarray] = None
        self._sim_pool_arr: Optional[np.ndarray] = None
        self._sim_normal_index: Optional[int] = None
        self._sim_label_map: Optional[dict[int, str]] = None
        self._sim_deep_model = None
        self._sim_rf_model = None
        # Simulation state to stream windows smoothly
        self._sim_idx = None
        self._sim_cursor = 0
        self._sim_cleaned_win = None
        self._sim_last_quality = None
        self._sim_cached_hr = 0.0
        self._sim_cached_ibi = []
        self._sim_cached_quality = 0.0
        self._sim_next_timestamp = None  # persistent time cursor per window
        self._sim_roll = None  # rolling buffer of streamed cleaned samples
        if settings.simulation_mode:
            self._initialize_simulation()
            # Start a lightweight background pusher so dashboards receive updates
            # even if no /infer requests arrive.
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._simulation_loop())
            except RuntimeError:
                # No running loop (e.g., during tests); ignore.
                pass

    async def run_inference(self, request: InferRequest) -> InferResponse:
        if self._settings.simulation_mode and self._sim_ready:
            state = await asyncio.to_thread(self._simulate_prediction)
            await self._broadcast(self._map_sim_push(state))
            return self._map_sim_result(state)

        frame_bytes = base64.b64decode(request.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode frame payload")
        result = await asyncio.to_thread(self._pipeline.process_frame, frame)
        await self._broadcast(self._map_push(result))
        return self._map_result(result)

    async def get_state_stream(self) -> AsyncIterator[ArrhythmiaPush]:
        queue: "asyncio.Queue[ArrhythmiaPush]" = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                payload = await queue.get()
                yield payload
        finally:
            self._subscribers.discard(queue)

    def health_status(self) -> dict[str, object]:
        if self._settings.simulation_mode:
            return {
                "model_loaded": bool(self._sim_ready),
                "device": self._settings.device,
                "ml_pipeline": "simulated" if self._sim_ready else "sim_uninitialized",
            }
        return {
            "model_loaded": self._pipeline.model_ready,
            "device": self._pipeline.device,
            "ml_pipeline": "ready" if self._pipeline.model_ready else "uninitialized",
        }

    def _map_result(self, result: PipelineResult) -> InferResponse:
        samples = [
            SignalSample(timestamp=sample.timestamp, value=sample.value)
            for sample in result.waveform
        ]
        return InferResponse(
            heart_rate_bpm=result.heart_rate_bpm,
            ibi_ms=result.ibi_ms,
            arrhythmia_state=result.arrhythmia_state,
            confidence=result.confidence,
            signal_quality=result.signal_quality,
            status=result.status,
            waveform=samples,
        )

    def _map_push(self, result: PipelineResult) -> ArrhythmiaPush:
        return ArrhythmiaPush(
            arrhythmia_state=result.arrhythmia_state,
            confidence=result.confidence,
            heart_rate_bpm=result.heart_rate_bpm,
            status=result.status,
        )

    async def _broadcast(self, payload: ArrhythmiaPush) -> None:
        if not self._subscribers:
            return
        coros = [queue.put(payload) for queue in self._subscribers]
        await asyncio.gather(*coros, return_exceptions=True)

    # --- Simulation helpers -------------------------------------------------
    def _initialize_simulation(self) -> None:
        # Ensure 'data' directory is importable so 'arrhythmia_project' can be resolved
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data"
        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))

        try:
            ap_dl = importlib.import_module("arrhythmia_project.data_loader")
            ap_ds = importlib.import_module("arrhythmia_project.dataset")
            ap_inf = importlib.import_module("arrhythmia_project.inference")
            ap_models = importlib.import_module("arrhythmia_project.models")
        except Exception:
            return

        data_root = Path(self._settings.sim_data_root)
        weights_dir = Path(self._settings.sim_weights_dir)
        records = ap_dl.load_records(data_root)
        if not records:
            return
        prep = ap_ds.PreprocessingConfig(sampling_rate=self._settings.sim_sampling_rate, augment=False)
        dataset = ap_ds.build_dataset(records, prep)
        # Optional split selection for simulation
        sim_split = (self._settings.sim_split or "all").lower()
        seed = int(self._settings.sim_seed)
        if sim_split in {"train", "test"}:
            train_set, _val_set, test_set = ap_ds.split_dataset(dataset, seed=seed)
            if sim_split == "train":
                indices = np.array(train_set.indices, dtype=int)
            else:
                indices = np.array(test_set.indices, dtype=int)
            self._sim_windows = dataset.windows[indices].numpy()
            self._sim_labels = dataset.labels[indices].numpy()
        else:
            self._sim_windows = dataset.windows.numpy()
            self._sim_labels = dataset.labels.numpy()
        # Build pools for class ratio sampling
        normal_idx = None
        for idx, name in dataset.index_to_name.items():
            if name.lower() == "normal":
                normal_idx = int(idx)
                break
        self._sim_normal_index = normal_idx
        if self._sim_labels is not None and normal_idx is not None:
            normal_mask = (self._sim_labels == normal_idx)
            arr_mask = ~normal_mask
            n_pool = np.nonzero(normal_mask)[0]
            a_pool = np.nonzero(arr_mask)[0]
            self._sim_pool_normal = n_pool if n_pool.size else None
            self._sim_pool_arr = a_pool if a_pool.size else None
        else:
            self._sim_pool_normal = None
            self._sim_pool_arr = None
        self._sim_label_map = ap_inf._load_label_mapping(weights_dir)
        device = torch.device(self._settings.device if torch.cuda.is_available() else "cpu")
        deep_model, rf_model, effective_map = ap_inf.load_models(
            weights_dir, self._sim_windows.shape[1], device, self._sim_label_map
        )
        self._sim_deep_model = deep_model
        self._sim_rf_model = rf_model
        # Update the simulation label map in case the checkpoint implies a different class set
        self._sim_label_map = effective_map
        self._sim_device = device
        self._sim_prep = prep
        self._sim_ready = True

    def _simulate_prediction(self) -> dict:
        assert self._sim_windows is not None and self._sim_ready
        # Select or continue streaming a window
        if self._sim_idx is None or self._sim_cleaned_win is None:
            # Choose a window guided by desired Normal ratio with occasional inversion
            total = int(self._sim_windows.shape[0])
            normal_ratio = float(getattr(self._settings, "sim_normal_ratio", 0.8))
            flip_prob = float(getattr(self._settings, "sim_flip_probability", 0.25))
            if np.random.rand() < flip_prob:
                normal_ratio = 1.0 - normal_ratio
            pick_normal = np.random.rand() < max(0.0, min(1.0, normal_ratio))
            chosen_idx = None
            if pick_normal and self._sim_pool_normal is not None and self._sim_pool_normal.size:
                chosen_idx = int(np.random.choice(self._sim_pool_normal))
            elif (not pick_normal) and self._sim_pool_arr is not None and self._sim_pool_arr.size:
                chosen_idx = int(np.random.choice(self._sim_pool_arr))
            else:
                chosen_idx = int(np.random.randint(0, total))
            self._sim_idx = chosen_idx
            self._sim_cursor = 0
            self._sim_cleaned_win = None
            # initialize streaming time origin
            from datetime import datetime
            self._sim_next_timestamp = datetime.utcnow()
            self._sim_roll = None
        idx = int(self._sim_idx)
        win = self._sim_windows[idx:idx+1]
        # RF
        # Import locally to avoid keeping module-level globals
        ap_models = importlib.import_module("arrhythmia_project.models")
        rf_features = ap_models.extract_hrv_features(win, self._sim_prep.sampling_rate)
        rf_probs = self._sim_rf_model.predict_proba(rf_features)[0]
        # Deep
        tensor = torch.from_numpy(win).float().to(self._sim_device)
        with torch.no_grad():
            logits = self._sim_deep_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # Choose deep prediction as primary
        label_idx = int(np.argmax(probs))
        label_name = self._sim_label_map.get(label_idx, f"class_{label_idx}")
        confidence = float(probs[label_idx])
        # Derive HR/IBI and signal quality from the window when possible
        fs = int(self._sim_prep.sampling_rate)
        raw = win[0]
        # Clean once per window and cache
        if self._sim_cleaned_win is None:
            cleaned = raw
            if filter_bandpass is not None:
                try:
                    cleaned = filter_bandpass(raw, fs, 0.7, 3.0)
                except Exception:
                    cleaned = raw
            self._sim_cleaned_win = cleaned
        else:
            cleaned = self._sim_cleaned_win
        # HR/IBI/quality: compute on rolling buffer to stay in sync with streamed chunk
        heart_rate_bpm = 0.0
        ibi_ms = []
        signal_quality = 0.0
        # Stream a small chunk of the waveform to avoid UI freezes
        from datetime import timedelta
        target_stream_hz = None  # we will match the sampling rate pace by default
        interval = float(self._settings.websocket_broadcast_interval)
        # default: match the underlying sampling rate over the interval
        chunk_size = max(1, int(fs * interval))
        start = int(self._sim_cursor)
        end = int(min(start + chunk_size, cleaned.shape[0]))
        slice_arr = cleaned[start:end]
        # Update rolling buffer with streamed samples and cap to one model window length
        roll_max = int(self._sim_windows.shape[1])  # use dataset window length (e.g., ~10s)
        if self._sim_roll is None:
            self._sim_roll = slice_arr.copy()
        else:
            if slice_arr.size > 0:
                self._sim_roll = np.concatenate([self._sim_roll, slice_arr])
                if self._sim_roll.shape[0] > roll_max:
                    self._sim_roll = self._sim_roll[-roll_max:]
        # Advance cursor; if at end, pick new window next time
        self._sim_cursor = end
        if self._sim_cursor >= cleaned.shape[0]:
            # reset to trigger a new window selection next call
            self._sim_idx = None
            self._sim_cleaned_win = None
            self._sim_last_quality = None
            self._sim_cached_hr = 0.0
            self._sim_cached_ibi = []
        # Build timestamps for the chunk
        dt = 1.0 / fs
        # Use a persistent timebase so timestamps are strictly increasing across pushes
        start_time = self._sim_next_timestamp
        waveform = []
        if slice_arr.size > 0:
            for i, v in enumerate(slice_arr.tolist()):
                ts = start_time + timedelta(seconds=i * dt)
                waveform.append({"timestamp": ts, "value": float(v)})
            # advance the next-timestamp cursor
            self._sim_next_timestamp = start_time + timedelta(seconds=len(slice_arr) * dt)

        # Recompute HR/IBI/quality on current rolling buffer
        roll = self._sim_roll if self._sim_roll is not None else cleaned[: end - start]
        if roll is not None and roll.size >= max(int(fs * 3), 128):
            if compute_heart_rate_and_ibi is not None:
                try:
                    heart_rate_bpm, ibi_ms = compute_heart_rate_and_ibi(roll, fs)
                except Exception:
                    heart_rate_bpm, ibi_ms = 0.0, []
            if estimate_signal_quality_psd is not None:
                try:
                    signal_quality = float(estimate_signal_quality_psd(roll, fs))
                except Exception:
                    signal_quality = 0.0
            if (not heart_rate_bpm) and rf_features.size > 0:
                heart_rate_bpm = float(rf_features[0][8])
        else:
            # startup fallback
            if estimate_signal_quality_psd is not None:
                try:
                    signal_quality = float(estimate_signal_quality_psd(cleaned, fs))
                except Exception:
                    signal_quality = 0.0

        # Light smoothing on quality only
        if self._sim_last_quality is None:
            self._sim_last_quality = signal_quality
        else:
            alpha = 0.2
            self._sim_last_quality = alpha * signal_quality + (1 - alpha) * self._sim_last_quality
        signal_quality = float(max(0.0, min(1.0, self._sim_last_quality)))

        return {
            "arrhythmia_state": label_name,
            "confidence": confidence,
            "heart_rate_bpm": float(heart_rate_bpm) if heart_rate_bpm else 0.0,
            "ibi_ms": [float(x) for x in ibi_ms] if ibi_ms else [],
            "signal_quality": signal_quality if signal_quality is not None else 0.0,
            "waveform": waveform,
            "status": "simulated",
        }

    def _map_sim_result(self, state: dict) -> InferResponse:
        samples = [
            SignalSample(timestamp=s["timestamp"], value=s["value"]) for s in state.get("waveform", [])
        ]
        return InferResponse(
            heart_rate_bpm=state["heart_rate_bpm"],
            ibi_ms=state.get("ibi_ms", []),
            arrhythmia_state=state["arrhythmia_state"],
            confidence=state["confidence"],
            signal_quality=float(state.get("signal_quality", 0.0)),
            status=state["status"],
            waveform=samples,
        )

    def _map_sim_push(self, state: dict) -> ArrhythmiaPush:
        samples = [
            SignalSample(timestamp=s["timestamp"], value=s["value"]) for s in state.get("waveform", [])
        ]
        return ArrhythmiaPush(
            arrhythmia_state=state["arrhythmia_state"],
            confidence=state["confidence"],
            heart_rate_bpm=state["heart_rate_bpm"],
            status=state["status"],
            signal_quality=float(state.get("signal_quality", 0.0)),
            ibi_ms=state.get("ibi_ms", []),
            waveform=samples,
        )

    async def _simulation_loop(self) -> None:
        """Periodically emit simulated updates over the websocket channel.

        Runs only when SIMULATION_MODE=true and models are initialized. This keeps
        the dashboard animated without requiring explicit /infer calls.
        """
        interval = float(self._settings.websocket_broadcast_interval)
        while True:
            try:
                await asyncio.sleep(interval)
                if not (self._settings.simulation_mode and self._sim_ready):
                    continue
                # Only push if someone is listening to avoid unnecessary work
                if not self._subscribers:
                    continue
                state = await asyncio.to_thread(self._simulate_prediction)
                await self._broadcast(self._map_sim_push(state))
            except asyncio.CancelledError:  # pragma: no cover - server shutdown
                break
            except Exception:
                # Swallow errors to keep the loop alive; logs can be added if needed
                continue


_service: Optional[PipelineService] = None


def get_service() -> PipelineService:
    global _service
    if _service is None:
        _service = PipelineService()
    return _service
