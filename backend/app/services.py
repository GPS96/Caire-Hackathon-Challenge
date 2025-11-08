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
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
from scipy import signal as scipy_signal

from .config import get_settings
from .models import ArrhythmiaPush, InferRequest, InferResponse, SignalSample

# ============================================================================
# AT-LSTM Signal Processing Functions (replacing dummy ml.signals)
# ============================================================================

def filter_bandpass(signal_data, fs=125, lowcut=0.7, highcut=3.0, order=4):
    """Bandpass filter for PPG signal (0.7-3.0 Hz for HR detection)."""
    try:
        nyquist = fs / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, signal_data)
    except Exception as e:
        print(f"[Backend] Bandpass filter error: {e}")
        return signal_data

def compute_heart_rate_from_lstm(waveform_segment, sampling_rate=125):
    """
    Compute HR from waveform using FFT-based peak detection.
    Input: PPG waveform segment (1D array)
    Output: Heart rate in BPM
    """
    try:
        if len(waveform_segment) < sampling_rate // 2:  # Need at least 0.5 seconds
            print(f"[DEBUG] Waveform too short: {len(waveform_segment)} < {sampling_rate // 2}")
            return 0.0
        
        # Normalize the signal
        sig = np.asarray(waveform_segment, dtype=float)
        sig = sig - np.mean(sig)
        if np.std(sig) > 0:
            sig = sig / np.std(sig)
        
        # FFT-based frequency detection
        freqs = np.fft.rfftfreq(len(sig), d=1/sampling_rate)
        fft_vals = np.abs(np.fft.rfft(sig))
        
        # Search in HR band (40-200 BPM = 0.67-3.33 Hz)
        hr_band_mask = (freqs >= 0.4) & (freqs <= 4.0)
        
        if not np.any(hr_band_mask):
            print(f"[DEBUG] No frequencies in HR band")
            return 0.0
        
        hr_band_fft = fft_vals[hr_band_mask]
        hr_band_freqs = freqs[hr_band_mask]
        
        # Find peak frequency
        peak_idx = np.argmax(hr_band_fft)
        peak_freq = hr_band_freqs[peak_idx]
        peak_power = hr_band_fft[peak_idx]
        
        # Convert Hz to BPM
        heart_rate_bpm = peak_freq * 60.0
        
        print(f"[DEBUG] HR Calculation: freq={peak_freq:.3f}Hz, power={peak_power:.2f}, HR={heart_rate_bpm:.1f}bpm")
        
        return float(heart_rate_bpm) if 40 <= heart_rate_bpm <= 200 else 0.0
    except Exception as e:
        print(f"[Backend] HR computation error: {e}")
        return 0.0


def compute_heart_rate_and_ibi(ppg_signal, fs):
    """
    Compute HR and IBI from PPG using frequency analysis.
    Output: (heart_rate_bpm, [ibi_ms_list])
    """
    try:
        hr_bpm = compute_heart_rate_from_lstm(ppg_signal, sampling_rate=fs)
        # Estimate IBI from HR (IBI in milliseconds)
        ibi_ms = [60000.0 / hr_bpm] if hr_bpm > 0 else []
        return hr_bpm, ibi_ms
    except Exception as e:
        print(f"[Backend] HR/IBI computation error: {e}")
        return 0.0, []

def estimate_signal_quality_psd(ppg_signal, fs):
    """
    Estimate signal quality using power spectral density (PSD).
    Quality = power in HR band / total power
    """
    try:
        # Compute PSD using Welch's method
        nperseg = min(256, len(ppg_signal))
        if nperseg < 4:
            return 0.0
        
        freqs, psd = scipy_signal.welch(ppg_signal, fs=fs, nperseg=nperseg)
        
        # Quality = ratio of power in HR band (0.5-3 Hz) to total power
        hr_band_mask = (freqs >= 0.5) & (freqs <= 3.0)
        hr_band_power = np.sum(psd[hr_band_mask])
        total_power = np.sum(psd)
        
        quality = hr_band_power / total_power if total_power > 0 else 0.0
        return float(np.clip(quality, 0.0, 1.0))
    except Exception as e:
        print(f"[Backend] Signal quality estimation error: {e}")
        return 0.0

def estimate_signal_quality(ppg_signal):
    """Wrapper for signal quality estimation at default sampling rate."""
    return estimate_signal_quality_psd(ppg_signal, fs=125)

# ============================================================================
# AT-LSTM arrhythmia project imports
# ============================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))

try:
    from data.arrhythmia_project.inference import get_detector
    ATLSTM_AVAILABLE = True
except ImportError:
    ATLSTM_AVAILABLE = False

# ============================================================================
# PipelineService
# ============================================================================

class PipelineService:
    """Coordinates synchronous inference requests and websocket broadcasts."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings

        if not settings.simulation_mode:
            try:
                from rppg.pipeline import RPPGInferencePipeline, PipelineConfig
                self._pipeline = RPPGInferencePipeline(
                    PipelineConfig(
                        heartbeat_window_seconds=settings.heartbeat_window_seconds,
                        model_checkpoint=settings.model_checkpoint_path,
                        device=settings.device,
                    )
                )
            except (NameError, ImportError):
                print("[Backend] RPPGInferencePipeline not available, skipping...")
                self._pipeline = None
        else:
            self._pipeline = None

        self._subscribers: set["asyncio.Queue[ArrhythmiaPush]"] = set()

        # Simulation resources
        self._sim_ready = False
        self._sim_windows: Optional[np.ndarray] = None
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

    async def _broadcast(self, payload: ArrhythmiaPush) -> None:
        if not self._subscribers:
            return

        coros = [queue.put(payload) for queue in self._subscribers]
        await asyncio.gather(*coros, return_exceptions=True)

    # ========================================================================
    # Simulation helpers
    # ========================================================================

    def _initialize_simulation(self) -> None:
        # Ensure 'data' directory is importable
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data"
        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))

        try:
            ap_dl = importlib.import_module("arrhythmia_project.data_loader")
            ap_ds = importlib.import_module("arrhythmia_project.dataset")
            ap_inf = importlib.import_module("arrhythmia_project.inference")
            ap_models = importlib.import_module("arrhythmia_project.models")
        except Exception as e:
            print(f"[Backend] Simulation init error: {e}")
            return

        data_root = Path(self._settings.sim_data_root)
        weights_dir = Path(self._settings.sim_weights_dir)

        records = ap_dl.load_records(data_root)
        if not records:
            print("[Backend] No records loaded for simulation")
            return

        prep = ap_ds.PreprocessingConfig(
            sampling_rate=self._settings.sim_sampling_rate, augment=False
        )
        dataset = ap_ds.build_dataset(records, prep)

        self._sim_windows = dataset.windows.numpy()
        self._sim_label_map = ap_inf._load_label_mapping(weights_dir)

        device = torch.device(
            self._settings.device if torch.cuda.is_available() else "cpu"
        )

        deep_model, rf_model = ap_inf.load_models(
            weights_dir, self._sim_windows.shape[1], device, self._sim_label_map
        )

        self._sim_deep_model = deep_model
        self._sim_rf_model = rf_model
        self._sim_device = device
        self._sim_prep = prep
        self._sim_ready = True
        print("[Backend] Simulation initialized with AT-LSTM models")

    def _simulate_prediction(self) -> dict:
        assert self._sim_windows is not None and self._sim_ready

        # Select or continue streaming a window
        if self._sim_idx is None or self._sim_cleaned_win is None:
            self._sim_idx = int(np.random.randint(0, self._sim_windows.shape[0]))
            self._sim_cursor = 0
            self._sim_cleaned_win = None
            self._sim_next_timestamp = datetime.utcnow()
            self._sim_roll = None

        idx = int(self._sim_idx)
        win = self._sim_windows[idx : idx + 1]

        # RF features
        ap_models = importlib.import_module("arrhythmia_project.models")
        rf_features = ap_models.extract_hrv_features(win, self._sim_prep.sampling_rate)
        rf_probs = self._sim_rf_model.predict_proba(rf_features)[0]

        # Deep model
        tensor = torch.from_numpy(win).float().to(self._sim_device)
        with torch.no_grad():
            logits = self._sim_deep_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Choose deep prediction as primary
        label_idx = int(np.argmax(probs))
        label_name = self._sim_label_map.get(label_idx, f"class_{label_idx}")
        confidence = float(probs[label_idx])

        # Derive HR/IBI and signal quality
        fs = int(self._sim_prep.sampling_rate)
        raw = win[0]

        # Clean once per window
        if self._sim_cleaned_win is None:
            cleaned = raw
            if filter_bandpass is not None:
                try:
                    cleaned = filter_bandpass(raw, fs=fs, lowcut=0.7, highcut=3.0)
                except Exception:
                    cleaned = raw
            self._sim_cleaned_win = cleaned
        else:
            cleaned = self._sim_cleaned_win

        # Stream a chunk
        interval = float(self._settings.websocket_broadcast_interval)
        chunk_size = max(1, int(fs * interval))

        start = int(self._sim_cursor)
        end = int(min(start + chunk_size, cleaned.shape[0]))

        slice_arr = cleaned[start:end]

        # Update rolling buffer
        roll_max = int(self._sim_windows.shape[1])
        if self._sim_roll is None:
            self._sim_roll = slice_arr.copy()
        else:
            if slice_arr.size > 0:
                self._sim_roll = np.concatenate([self._sim_roll, slice_arr])
            if self._sim_roll.shape[0] > roll_max:
                self._sim_roll = self._sim_roll[-roll_max:]

        # Advance cursor
        self._sim_cursor = end
        if self._sim_cursor >= cleaned.shape[0]:
            self._sim_idx = None
            self._sim_cleaned_win = None
            self._sim_last_quality = None

        # Build timestamps
        dt = 1.0 / fs
        start_time = self._sim_next_timestamp
        waveform = []

        if slice_arr.size > 0:
            for i, v in enumerate(slice_arr.tolist()):
                ts = start_time + timedelta(seconds=i * dt)
                waveform.append({"timestamp": ts.isoformat(), "value": float(v)})
            self._sim_next_timestamp = start_time + timedelta(seconds=len(slice_arr) * dt)

        # Compute HR/IBI/quality on rolling buffer
        heart_rate_bpm = 0.0
        ibi_ms = []
        signal_quality = 0.0

        roll = self._sim_roll if self._sim_roll is not None else cleaned[:end]

        if roll is not None and roll.size >= max(int(fs * 3), 128):
            heart_rate_bpm, ibi_ms = compute_heart_rate_and_ibi(roll, fs)
            signal_quality = estimate_signal_quality_psd(roll, fs)

        if not heart_rate_bpm and rf_features.size > 0:
            heart_rate_bpm = float(rf_features[0][8])

        # Light smoothing on quality
        if self._sim_last_quality is None:
            self._sim_last_quality = signal_quality
        else:
            alpha = 0.2
            self._sim_last_quality = (
                alpha * signal_quality + (1 - alpha) * self._sim_last_quality
            )
        signal_quality = float(max(0.0, min(1.0, self._sim_last_quality)))

        return {
            "arrhythmia_state": label_name,
            "confidence": confidence,
            "heart_rate_bpm": float(heart_rate_bpm) if heart_rate_bpm else 0.0,
            "ibi_ms": [float(x) for x in ibi_ms] if ibi_ms else [],
            "signal_quality": signal_quality,
            "waveform": waveform,
            "status": "simulated",
        }

    def _map_sim_result(self, state: dict) -> InferResponse:
        samples = [
            SignalSample(timestamp=s["timestamp"], value=s["value"])
            for s in state.get("waveform", [])
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
            SignalSample(timestamp=s["timestamp"], value=s["value"])
            for s in state.get("waveform", [])
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
        """Periodically emit simulated updates."""
        interval = float(self._settings.websocket_broadcast_interval)

        while True:
            try:
                await asyncio.sleep(interval)

                if not (self._settings.simulation_mode and self._sim_ready):
                    continue

                if not self._subscribers:
                    continue

                state = await asyncio.to_thread(self._simulate_prediction)
                await self._broadcast(self._map_sim_push(state))

            except asyncio.CancelledError:
                break
            except Exception:
                continue


_service: Optional[PipelineService] = None


def get_service() -> PipelineService:
    global _service
    if _service is None:
        _service = PipelineService()
    return _service
