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
from ..navigation import CONVERSATION_HISTORY, run_agent
from dotenv import load_dotenv
import sys
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

# Allow frontend dev servers to call REST endpoints (including navigation agent)
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
        # Ensure datetimes are encoded as ISO strings for WebSocket JSON frames
        try:
            await websocket.send_json(payload.model_dump(mode="json"))
        except WebSocketDisconnect:
            # client disconnected, stop streaming
            return
        except RuntimeError as exc:
            # Starlette may raise a RuntimeError if the socket is closed or not accepted.
            print(f"[Backend] _stream_updates: websocket send failed: {exc}")
            return
        except Exception as exc:
            # Defensive: log and stop the streaming task on unexpected errors
            print(f"[Backend] _stream_updates unexpected error: {exc}")
            return


@app.on_event("startup")
async def startup_event() -> None:  # pragma: no cover - FastAPI lifecycle hook
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
        # Common misconfiguration (e.g., missing API key). Surface clearly to the caller.
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - integration path
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.get("ok") and result.get("history"):
        CONVERSATION_HISTORY[session_id] = result["history"]

    return result


@app.post("/reports/session")
async def save_session_report(payload: SessionReportPayload) -> Dict[str, Any]:
    try:
        session_id = await asyncio.to_thread(_append_session_report, payload)
    except Exception as exc:  # pragma: no cover - file I/O safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "session_id": session_id}


@app.get("/reports/weekly", response_model=WeeklySummaryResponse)
async def weekly_summary() -> WeeklySummaryResponse:
    try:
        summary = await asyncio.to_thread(_compute_weekly_summary)
    except Exception as exc:  # pragma: no cover - file I/O safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return summary


@app.websocket("/ws/arrhythmia")
async def arrhythmia_socket(websocket: WebSocket, service: PipelineService = Depends(get_service)) -> None:
    await websocket.accept()
    import json
    from rppg.cleaning import clean_bvp, CleanOptions
    import numpy as np
    import websockets
    import orjson
    import os
    import uuid
    import time
    
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
    
    # Start the simulation stream as well (fallback while waiting for real data)
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
                        timestamp = payload.get("timestamp")
                        
                        if frame_base64:
                            frame_count += 1
                            # Build payload for Caire API (matching live_pipeline format)
                            caire_payload = {
                                "datapt_id": str(uuid.uuid4()),
                                "state": "stream",
                                "timestamp": f"{time.time():.6f}",
                                "frame_data": frame_base64,
                                "advanced": True,
                            }
                            await caire_ws.send(orjson.dumps(caire_payload).decode("utf-8"))
                            if frame_count % 30 == 0:  # Log every 30 frames
                                print(f"[Backend] Forwarded {frame_count} frames to Caire API")
                    except WebSocketDisconnect:
                        break
                    except Exception as exc:
                        print(f"[Backend] Error forwarding frame: {exc}")
            
            async def receive_and_clean():
                """Receive rPPG from Caire API, clean to PPG, send to frontend"""
                nonlocal fs_inferred, real_data_received, simulation_task
                msg_count = 0
                while True:
                    try:
                        msg = await caire_ws.recv()
                        msg_count += 1
                        obj = orjson.loads(msg) if isinstance(msg, str) else orjson.loads(msg.decode("utf-8", errors="ignore"))
                        
                        # Debug: print first few messages
                        if msg_count <= 10:
                            print(f"[Backend DEBUG] Caire message #{msg_count}: {obj}")
                        
                        # Extract rPPG (matching live_pipeline extraction logic)
                        arr, fs = None, None
                        
                        # Check under "advanced" field
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
                            if msg_count <= 10:
                                print(f"[Backend DEBUG] No rPPG extracted from message #{msg_count} (Caire API needs more frames to accumulate)")
                            continue
                        
                        # First time we receive real data, cancel simulation
                        if not real_data_received:
                            real_data_received = True
                            simulation_task.cancel()
                            print("[Backend] Received first real rPPG data, switching from simulation to real data")
                        
                        print(f"[Backend] Received rPPG: {arr.size} samples @ {fs} Hz")
                        
                        if fs and not fs_inferred:
                            fs_inferred = fs
                        
                        # Append to buffer
                        rppg_buffer.extend(arr.tolist())
                        max_samples = int(fs_inferred * 20) if fs_inferred else 600
                        if len(rppg_buffer) > max_samples:
                            del rppg_buffer[:-max_samples]
                        
                        # Clean to PPG
                        if fs_inferred and len(rppg_buffer) > 30:
                            raw = np.asarray(rppg_buffer, dtype=float)
                            clean, info = clean_bvp(raw, fs=fs_inferred, options=CleanOptions())
                            
                            # Get tail matching received rPPG size
                            n_samples = min(arr.size, len(raw), len(clean))
                            tail_clean = clean[-n_samples:]
                            
                            # Build waveform payload with PPG
                            from datetime import datetime
                            waveform = [
                                {"timestamp": datetime.utcnow().isoformat(), "value": float(v)}
                                for v in tail_clean
                            ]
                            
                            # Send PPG to frontend. Previously this was a hard-coded
                            # placeholder message (arrhythmia_state="normal", hr=75).
                            # Replace that with an actual inference using the
                            # arrhythmia_project inference helper when available so
                            # the frontend receives model-derived labels.
                            arrhythmia_state = "normal"
                            confidence = 0.0
                            heart_rate_value = None
                            try:
                                # Attempt to run model inference on the cleaned tail
                                from pathlib import Path
                                # Ensure the repository 'data' directory is on sys.path so
                                # the bundled arrhythmia_project package can be imported.
                                project_root = Path(__file__).resolve().parents[2]
                                data_dir = project_root / "data"
                                if str(data_dir) not in sys.path:
                                    sys.path.insert(0, str(data_dir))
                                try:
                                    from arrhythmia_project import inference as ap_inf
                                except Exception:
                                    ap_inf = None

                                if ap_inf is not None and fs:
                                    try:
                                        weights_dir = Path(settings.sim_weights_dir)
                                        preds = ap_inf.predict(tail_clean, sampling_rate=int(fs), weights_dir=weights_dir)
                                        deep = preds.get("deep_model") or {}
                                        arrhythmia_state = deep.get("label_name", arrhythmia_state)
                                        confidence = float(deep.get("confidence", 0.0))
                                    except Exception:
                                        # Fall back to unknown/placeholder if inference fails
                                        arrhythmia_state = "unknown"
                                        confidence = 0.0

                                # Try to compute a coarse heart-rate estimate if possible
                                try:
                                    # Simple heuristic: compute instantaneous HR from peaks
                                    import numpy as _np
                                    from scipy.signal import find_peaks as _find_peaks

                                    sig = _np.asarray(tail_clean)
                                    if sig.size > 10 and fs:
                                        peaks, _ = _find_peaks(sig, distance=int(0.3 * fs))
                                        if peaks.size >= 2:
                                            rr = _np.diff(peaks) / float(fs)
                                            mean_rr = float(_np.mean(rr)) if rr.size else None
                                            if mean_rr and mean_rr > 0:
                                                heart_rate_value = float(60.0 / mean_rr)
                                except Exception:
                                    heart_rate_value = None

                                # Compute HRV-based fatigue score (use fatigue utilities if available)
                                fatigue_score_val = None
                                try:
                                    project_root = Path(__file__).resolve().parents[2]
                                    data_dir = project_root / "data"
                                    if str(data_dir) not in sys.path:
                                        sys.path.insert(0, str(data_dir))
                                    try:
                                        from fatigue import compute_subscores_and_score
                                    except Exception:
                                        compute_subscores_and_score = None

                                    if compute_subscores_and_score is not None and fs and sig.size > 10:
                                        # compute RR intervals (seconds)
                                        peaks, _ = _find_peaks(sig, distance=int(0.3 * fs))
                                        if peaks.size >= 2:
                                            rr = _np.diff(peaks) / float(fs)
                                            mean_rr = float(_np.mean(rr)) if rr.size else 0.0
                                            std_rr = float(_np.std(rr)) if rr.size else 0.0
                                            # RMSSD: root mean square of successive differences of RR
                                            if rr.size >= 2:
                                                diff_rr = _np.diff(rr)
                                                rmssd = float(_np.sqrt(_np.mean(diff_rr ** 2))) if diff_rr.size else 0.0
                                            else:
                                                rmssd = 0.0

                                            hrv_feats = {
                                                "heart_rate": float(60.0 / mean_rr) if mean_rr else 0.0,
                                                "mean_rr": float(mean_rr),
                                                "std_rr": float(std_rr),
                                                "rmssd": float(rmssd),
                                            }
                                            # Use HRV-only by setting facial_weight=0.0
                                            try:
                                                combined_score, subs = compute_subscores_and_score({}, hrv_feats, facial_weight=0.0)
                                                fatigue_score_val = float(combined_score)
                                            except Exception:
                                                fatigue_score_val = None
                                except Exception:
                                    fatigue_score_val = None

                            except Exception:
                                arrhythmia_state = "unknown"
                                confidence = 0.0

                            push = {
                                "arrhythmia_state": arrhythmia_state,
                                "confidence": confidence,
                                "heart_rate_bpm": float(heart_rate_value) if heart_rate_value else 0.0,
                                # fatigue_score is 0-100 when computed; optional
                                "fatigue_score": float(fatigue_score_val) if fatigue_score_val is not None else None,
                                "status": "Info",
                                "generated_at": datetime.utcnow().isoformat(),
                                "signal_quality": 0.85,
                                "ibi_ms": [],
                                "waveform": waveform,
                            }
                            try:
                                await websocket.send_text(json.dumps(push))
                                print(f"[Backend] Sent PPG to frontend: {len(waveform)} samples (label={arrhythmia_state})")
                            except WebSocketDisconnect:
                                print("[Backend] Client disconnected while sending push; stopping sender loop")
                                break
                            except RuntimeError as exc:
                                # e.g., "websocket is not connected, need to call accept first"
                                print(f"[Backend] Failed to send push (socket state): {exc}")
                                break
                            except Exception as exc:
                                print(f"[Backend] Error sending push to frontend: {exc}")
                                break
                    
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as exc:
                        print(f"[Backend] Error processing rPPG: {exc}")
            
            # Run both tasks concurrently as managed background tasks. If one
            # completes (for example the client disconnects or the Caire API
            # closes), cancel the other and exit cleanly.
            t_forward = asyncio.create_task(forward_frames())
            t_receive = asyncio.create_task(receive_and_clean())
            try:
                done, pending = await asyncio.wait([t_forward, t_receive], return_when=asyncio.FIRST_COMPLETED)
                # Cancel any remaining tasks
                for p in pending:
                    p.cancel()
                    try:
                        await p
                    except asyncio.CancelledError:
                        pass
            finally:
                # Ensure simulation task is cancelled if still running
                if simulation_task and not simulation_task.done():
                    simulation_task.cancel()
                    try:
                        await simulation_task
                    except asyncio.CancelledError:
                        pass
                # Ensure websocket to Caire API is closed
                try:
                    await caire_ws.close()
                except Exception:
                    pass
            # end managed tasks
    
    except WebSocketDisconnect:
        return
    except Exception as exc:
        print(f"[Backend] WebSocket error: {exc}")
        # Fall back to simulation if Caire API fails
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
            except (KeyError, ValueError):  # pragma: no cover - malformed row
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
