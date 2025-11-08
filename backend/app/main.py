"""FastAPI entry point for the driver health assessment backend."""

import asyncio
import csv
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import get_settings
from .models import (
    ArrhythmiaPush,
    HealthResponse,
    InferRequest,
    InferResponse,
    SessionReportPayload,
    SessionReportRow,
    WeeklySummaryResponse,
)
from .services import PipelineService, get_service

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navigation import CONVERSATION_HISTORY, run_agent
from dotenv import load_dotenv

import json
import numpy as np
import websockets
import orjson
import os
import time
import torch

class AgentQueryPayload(BaseModel):
    query: str
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    session_id: Optional[str] = None

settings = get_settings()

_APP_DIR = Path(__file__).resolve().parent
load_dotenv(_APP_DIR / ".env", override=False)
load_dotenv(_APP_DIR.parent / ".env", override=False)

app = FastAPI(title=settings.app_name, version=settings.app_version)

_REPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "report_database"
_REPORT_FILE = _REPORT_DIR / "drive_reports.csv"
_REPORT_HEADERS = [
    "session_id",
    "started_at",
    "ended_at",
    "duration_seconds",
    "mean_signal_quality",
    "mean_confidence",
    "mean_ibi_ms",
    "mean_heart_rate",
    "dominant_arrhythmia",
    "score",
]

# Allow frontend dev servers to call REST endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def _stream_updates(service: PipelineService, websocket: WebSocket) -> None:
    async for payload in service.get_state_stream():
        await websocket.send_json(payload.model_dump(mode="json"))

@app.on_event("startup")
async def startup_event() -> None:
    _ = get_service()

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest, service: PipelineService = Depends(get_service)) -> InferResponse:
    try:
        return await service.run_inference(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/health", response_model=HealthResponse)
async def health(service: PipelineService = Depends(get_service)) -> HealthResponse:
    status = service.health_status()
    return HealthResponse(
        status="ok" if status["model_loaded"] else "degraded",
        backend="ok",
        ml_pipeline=status["ml_pipeline"],
        model_loaded=status["model_loaded"],
        device=status["device"],
    )

@app.post("/navigation/agent")
async def navigation_agent(payload: AgentQueryPayload) -> Dict[str, Any]:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    session_id = payload.session_id or "default"
    history = CONVERSATION_HISTORY.get(session_id, [])
    
    try:
        result = await asyncio.to_thread(
            run_agent,
            query,
            payload.user_lat,
            payload.user_lon,
            history,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    if result.get("ok") and result.get("history"):
        CONVERSATION_HISTORY[session_id] = result["history"]
    
    return result

@app.post("/reports/session")
async def save_session_report(payload: SessionReportPayload) -> Dict[str, Any]:
    try:
        session_id = await asyncio.to_thread(_append_session_report, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return {"ok": True, "session_id": session_id}

@app.get("/reports/weekly", response_model=WeeklySummaryResponse)
async def weekly_summary() -> WeeklySummaryResponse:
    try:
        summary = await asyncio.to_thread(_compute_weekly_summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return summary

@app.websocket("/ws/arrhythmia")
async def arrhythmia_socket(websocket: WebSocket, service: PipelineService = Depends(get_service)) -> None:
    await websocket.accept()
    
    try:
        from rppg.cleaning import clean_bvp, CleanOptions
        RPPG_AVAILABLE = True
    except ModuleNotFoundError:
        RPPG_AVAILABLE = False
        clean_bvp = None
        CleanOptions = None
    
    # Get Caire API config from env
    API_KEY = os.getenv("API_KEY", "")
    BACKEND_WS_BASE = os.getenv("BACKEND_WS_BASE", "")
    CLIENT = os.getenv("CLIENT", "livePython")
    
    # If Caire API is configured, use it; otherwise fall back to simulation
    use_caire_api = bool(API_KEY and BACKEND_WS_BASE)
    
    if not use_caire_api:
        print("[Backend] No Caire API configured, using simulation mode")
        try:
            await _stream_updates(service, websocket)
        except WebSocketDisconnect:
            return
        return
    
    # Build Caire API WebSocket URL
    from urllib.parse import urlencode
    params = {"api_key": API_KEY, "client": CLIENT}
    caire_ws_url = f"{BACKEND_WS_BASE.rstrip('/')}/?{urlencode(params)}"
    
    print(f"[Backend] Connecting to Caire API: {caire_ws_url}")
    
    rppg_buffer = []
    fs_inferred = None
    
    # Start the simulation stream as fallback
    simulation_task = asyncio.create_task(_stream_updates(service, websocket))
    real_data_received = False
    
    try:
        async with websockets.connect(caire_ws_url, max_size=2**22, compression=None) as caire_ws:
            print("[Backend] Connected to Caire API")
            
            async def forward_frames():
                """Receive frames from frontend, forward to Caire API"""
                frame_count = 0
                while True:
                    try:
                        data = await websocket.receive_text()
                        payload = json.loads(data)
                        frame_base64 = payload.get("frame_base64")
                        
                        if frame_base64:
                            frame_count += 1
                            caire_payload = {
                                "datapt_id": str(uuid.uuid4()),
                                "state": "stream",
                                "timestamp": f"{time.time():.6f}",
                                "frame_data": frame_base64,
                                "advanced": True,
                            }
                            await caire_ws.send(orjson.dumps(caire_payload).decode("utf-8"))
                            
                            if frame_count % 30 == 0:
                                print(f"[Backend] Forwarded {frame_count} frames to Caire API")
                    except WebSocketDisconnect:
                        break
                    except Exception as exc:
                        print(f"[Backend] Error forwarding frame: {exc}")
            
            async def receive_and_clean():
                """Receive rPPG from Caire API, run AT-LSTM, send to frontend"""
                nonlocal fs_inferred, real_data_received, simulation_task
                msg_count = 0
                
                while True:
                    try:
                        msg = await caire_ws.recv()
                        msg_count += 1
                        obj = orjson.loads(msg) if isinstance(msg, str) else orjson.loads(msg.decode("utf-8", errors="ignore"))
                        
                        # Extract rPPG
                        arr, fs = None, None
                        advanced = obj.get("advanced")
                        
                        if isinstance(advanced, dict) and "rppg" in advanced:
                            rppg_list = advanced.get("rppg")
                            if isinstance(rppg_list, (list, tuple)) and len(rppg_list) > 0:
                                arr = np.asarray(rppg_list, dtype=float)
                                timestamps = advanced.get("rppg_timestamps")
                                if isinstance(timestamps, (list, tuple)) and len(timestamps) > 1:
                                    dt = timestamps[-1] - timestamps[0]
                                    if dt > 0:
                                        fs = (len(timestamps) - 1) / dt
                        
                        if arr is None or arr.size == 0:
                            continue
                        
                        # First real data received - cancel simulation
                        if not real_data_received:
                            real_data_received = True
                            simulation_task.cancel()
                            print("[Backend] Received first real rPPG data, switching from simulation to real data")
                        
                        print(f"[Backend] Received rPPG: {arr.size} samples @ {fs} Hz")
                        
                        if fs and not fs_inferred:
                            fs_inferred = fs
                        
                        # Append to rolling buffer
                        rppg_buffer.extend(arr.tolist())
                        max_samples = int(fs_inferred * 20) if fs_inferred else 600
                        if len(rppg_buffer) > max_samples:
                            del rppg_buffer[:-max_samples]
                        
                        # Clean to PPG
                        tail_clean = None
                        if RPPG_AVAILABLE and fs_inferred and len(rppg_buffer) > 30:
                            try:
                                raw = np.asarray(rppg_buffer, dtype=float)
                                clean, info = clean_bvp(raw, fs=fs_inferred, options=CleanOptions())
                                n_samples = min(arr.size, len(clean))
                                tail_clean = clean[-n_samples:]
                            except Exception as e:
                                print(f"[Backend] Error cleaning rPPG: {e}")
                                tail_clean = np.asarray(rppg_buffer[-min(arr.size, len(rppg_buffer)):], dtype=float)
                        else:
                            tail_clean = np.asarray(rppg_buffer[-min(arr.size, len(rppg_buffer)):], dtype=float)
                        
                        # ============= RUN YOUR AT-LSTM MODEL ON PPG =============
                        arrhythmia_state = "healthy"
                        confidence = 0.0
                        heart_rate_bpm = 0.0
                        
                        if len(rppg_buffer) >= 1000:  # Need minimum 1000 samples for model
                            try:
                                # Import your actual AT-LSTM modules
                                from arrhythmia_project.models import CNNATLSTM
                                from arrhythmia_project.dataset import PPGPreprocessor, PreprocessingConfig
                                
                                fs_compute = int(fs_inferred) if fs_inferred else 125
                                config = PreprocessingConfig(sampling_rate=fs_compute)
                                preprocessor = PPGPreprocessor(config)
                                
                                # Get last 1000 samples for model input
                                raw_ppg = np.asarray(rppg_buffer[-1000:], dtype=np.float32)
                                
                                # Preprocess using YOUR preprocessor
                                windows = preprocessor.preprocess(raw_ppg)  # Returns preprocessed window
                                
                                # Load weights
                                weights_dir = Path(__file__).parent.parent.parent / "data/caire_weights_robust"
                                device = torch.device("cpu")
                                
                                # Load label mapping from your weights folder
                                label_map_path = weights_dir / "label_mapping.json"
                                if label_map_path.exists():
                                    with open(label_map_path, 'r') as f:
                                        label_data = json.load(f)
                                        label_mapping = label_data.get("index_to_name", {})
                                        # Ensure both 0 and 1 are mapped
                                        if "0" not in label_mapping:
                                            label_mapping["0"] = "healthy"
                                        if "1" not in label_mapping:
                                            label_mapping["1"] = "arrhythmic"
                                else:
                                    label_mapping = {"0": "healthy", "1": "arrhythmic"}
                                
                                # Load your trained AT-LSTM model
                                model = CNNATLSTM(input_length=1000, num_classes=2)
                                model_path = weights_dir / "cnn_atlstm.pt"
                                
                                if model_path.exists():
                                    state_dict = torch.load(model_path, map_location=device, weights_only=False)
                                    model.load_state_dict(state_dict)
                                    model.eval()
                                    
                                    # Run inference
                                    with torch.no_grad():
                                        tensor = torch.from_numpy(windows).float().to(device)
                                        logits = model(tensor)
                                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                                        
                                        pred_idx = int(np.argmax(probs[0]))
                                        confidence = float(probs[0][pred_idx])
                                        arrhythmia_state = label_mapping.get(str(pred_idx), "healthy" if pred_idx == 0 else "arrhythmic")
                                    
                                    print(f"[Backend] AT-LSTM loaded and ran successfully")
                                else:
                                    print(f"[Backend] Model file not found at {model_path}")
                                
                                # Compute HR from PPG using FFT
                                raw_all = np.asarray(rppg_buffer, dtype=float)
                                freqs = np.fft.rfftfreq(len(raw_all), d=1/fs_compute)
                                fft_vals = np.abs(np.fft.rfft(raw_all))
                                hr_band_mask = (freqs >= 0.4) & (freqs <= 4.0)
                                if np.any(hr_band_mask):
                                    peak_idx = np.argmax(fft_vals[hr_band_mask])
                                    peak_freq = freqs[hr_band_mask][peak_idx]
                                    heart_rate_bpm = float(peak_freq * 60.0)
                                    if not (40 <= heart_rate_bpm <= 200):
                                        heart_rate_bpm = 0.0
                                
                                print(f"[Backend] AT-LSTM: {arrhythmia_state} (confidence={confidence:.2f}), HR={heart_rate_bpm:.1f}bpm")
                            
                            except Exception as e:
                                print(f"[Backend] AT-LSTM inference error: {e}")
                                import traceback
                                traceback.print_exc()
                                arrhythmia_state = "healthy"
                                confidence = 0.0
                        else:
                            # Not enough samples yet
                            print(f"[Backend] Buffering: {len(rppg_buffer)}/1000 samples for AT-LSTM")
                        
                        # Build waveform payload
                        waveform = [
                            {"timestamp": datetime.utcnow().isoformat(), "value": float(v)}
                            for v in tail_clean
                        ]
                        
                        # Send to frontend with AT-LSTM predictions + dynamic HR
                        push = {
                            "arrhythmia_state": arrhythmia_state,
                            "confidence": confidence,
                            "heart_rate_bpm": heart_rate_bpm,
                            "status": "Info",
                            "generated_at": datetime.utcnow().isoformat(),
                            "signal_quality": 0.85,
                            "ibi_ms": [],
                            "waveform": waveform,
                        }
                        
                        await websocket.send_text(json.dumps(push))
                        print(f"[Backend] Sent PPG to frontend: {len(waveform)} samples")
                    
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as exc:
                        print(f"[Backend] Error processing rPPG: {exc}")
            
            # Run both tasks
            try:
                await asyncio.gather(forward_frames(), receive_and_clean())
            except asyncio.CancelledError:
                pass
            except WebSocketDisconnect:
                return
            except Exception as exc:
                print(f"[Backend] WebSocket error: {exc}")
    
    except Exception as exc:
        print(f"[Backend] Caire connection error: {exc}")
    
    # Fall back to simulation if needed
    if not real_data_received:
        print("[Backend] Caire API failed, falling back to simulation")
        try:
            await _stream_updates(service, websocket)
        except:
            pass

def _append_session_report(payload: SessionReportPayload) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    session_id = payload.session_id or uuid.uuid4().hex
    
    row = {
        "session_id": session_id,
        "started_at": payload.started_at.isoformat(),
        "ended_at": payload.ended_at.isoformat(),
        "duration_seconds": f"{payload.duration_seconds:.2f}",
        "mean_signal_quality": f"{payload.mean_signal_quality:.4f}",
        "mean_confidence": f"{payload.mean_confidence:.4f}",
        "mean_ibi_ms": f"{payload.mean_ibi_ms:.4f}",
        "mean_heart_rate": f"{payload.mean_heart_rate:.4f}",
        "dominant_arrhythmia": payload.dominant_arrhythmia,
        "score": f"{payload.score:.2f}",
    }
    
    file_exists = _REPORT_FILE.exists()
    with _REPORT_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_REPORT_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    return session_id

def _parse_iso8601(value: str) -> datetime:
    cleaned = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _compute_weekly_summary() -> WeeklySummaryResponse:
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=7)
    
    if not _REPORT_FILE.exists():
        return WeeklySummaryResponse(
            window_start=window_start,
            window_end=window_end,
            session_count=0,
            total_drive_time_seconds=0.0,
            average_signal_quality=0.0,
            average_confidence=0.0,
            average_ibi_ms=0.0,
            average_heart_rate=0.0,
            average_score=0.0,
            top_arrhythmia=None,
            sessions=[],
        )
    
    sessions: list[SessionReportRow] = []
    with _REPORT_FILE.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ended_at = _parse_iso8601(row["ended_at"])
            except (KeyError, ValueError):
                continue
            
            if ended_at < window_start:
                continue
            
            try:
                session = SessionReportRow(
                    session_id=row["session_id"],
                    started_at=_parse_iso8601(row["started_at"]),
                    ended_at=ended_at,
                    duration_seconds=float(row["duration_seconds"]),
                    mean_signal_quality=float(row["mean_signal_quality"]),
                    mean_confidence=float(row["mean_confidence"]),
                    mean_ibi_ms=float(row["mean_ibi_ms"]),
                    mean_heart_rate=float(row["mean_heart_rate"]),
                    dominant_arrhythmia=row["dominant_arrhythmia"],
                    score=float(row["score"]),
                )
            except (KeyError, ValueError):
                continue
            
            sessions.append(session)
    
    if not sessions:
        return WeeklySummaryResponse(
            window_start=window_start,
            window_end=window_end,
            session_count=0,
            total_drive_time_seconds=0.0,
            average_signal_quality=0.0,
            average_confidence=0.0,
            average_ibi_ms=0.0,
            average_heart_rate=0.0,
            average_score=0.0,
            top_arrhythmia=None,
            sessions=[],
        )
    
    total_drive_time = sum(s.duration_seconds for s in sessions)
    avg_signal_quality = sum(s.mean_signal_quality for s in sessions) / len(sessions)
    avg_confidence = sum(s.mean_confidence for s in sessions) / len(sessions)
    avg_ibi = sum(s.mean_ibi_ms for s in sessions) / len(sessions)
    avg_hr = sum(s.mean_heart_rate for s in sessions) / len(sessions)
    avg_score = sum(s.score for s in sessions) / len(sessions)
    
    arrhythmia_counts = Counter(s.dominant_arrhythmia for s in sessions if s.dominant_arrhythmia)
    top_arrhythmia = None
    if arrhythmia_counts:
        top_arrhythmia = arrhythmia_counts.most_common(1)[0][0]
    
    sessions_sorted = sorted(sessions, key=lambda x: x.ended_at, reverse=True)
    
    return WeeklySummaryResponse(
        window_start=window_start,
        window_end=window_end,
        session_count=len(sessions_sorted),
        total_drive_time_seconds=total_drive_time,
        average_signal_quality=avg_signal_quality,
        average_confidence=avg_confidence,
        average_ibi_ms=avg_ibi,
        average_heart_rate=avg_hr,
        average_score=avg_score,
        top_arrhythmia=top_arrhythmia,
        sessions=sessions_sorted,
    )
