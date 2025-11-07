"""Live pipeline: capture webcam frames, stream to backend WS, receive rPPG, convert to clean PPG,
append to CSV.

Requirements:
- OpenCV (cv2)
- websockets
- orjson
- numpy, scipy
- python-dotenv

Usage:
1. Copy .env.example to .env and fill in your API_KEY
2. Run: uv run live_pipeline.py

Output:
- CSV file live_ppg.csv with columns: time_s, rppg, ppg

Notes:
- We maintain a rolling buffer of raw rPPG values so the cleaning filter has context.
- If server messages already include a cleaned signal you can adjust extraction accordingly.
- This script assumes server sends JSON objects containing either:
  { "rppg": [..], "fs": <number> }  OR  { "data": { "rppg": [..], "fs": <number> } }
Adjust `extract_rppg` if your schema differs.
"""
from __future__ import annotations
import asyncio
import os
import uuid
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Load .env file if present
from dotenv import load_dotenv

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Load .env from the same directory as the script
dotenv_path = SCRIPT_DIR / ".env"
# Use override=True to replace existing environment variables
loaded = load_dotenv(dotenv_path, override=True)
print(f"[DEBUG] dotenv_path: {dotenv_path}")
print(f"[DEBUG] dotenv loaded: {loaded}")
print(f"[DEBUG] dotenv exists: {dotenv_path.exists()}")
if dotenv_path.exists():
    print(f"[DEBUG] First few lines of .env:")
    with open(dotenv_path) as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"  {line.rstrip()}")

import cv2
import numpy as np
import orjson
import websockets
import matplotlib
matplotlib.use('TkAgg')  # Use faster backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Allow running from python_demo without installing package
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rppg.cleaning import clean_bvp, CleanOptions

# -------------------- Config --------------------
API_KEY = os.getenv("API_KEY", "REPLACE_ME")
BACKEND_WS_BASE = os.getenv("BACKEND_WS_BASE", "ws://localhost:8000/ws/")
FPS = float(os.getenv("FPS", "30"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_FORMAT = os.getenv("FRAME_FORMAT", "jpeg").lower()  # raw|jpeg
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "75"))
CLIENT = os.getenv("CLIENT", "livePython")
OBJECT_ID = os.getenv("OBJECT_ID", "")
CALLBACK_URL = os.getenv("CALLBACK_URL", "")
CSV_PATH = Path(os.getenv("LIVE_CSV", "live_ppg.csv"))
MAX_RPPG_BUFFER_SEC = float(os.getenv("RPPG_BUFFER_SEC", "20"))  # seconds of history for filtering
PRINT_FPS_INTERVAL = 60
SHOW_PREVIEW = os.getenv("SHOW_PREVIEW", "1") not in ("0", "false", "False")
SHOW_PLOT = os.getenv("SHOW_PLOT", "1") not in ("0", "false", "False")
PLOT_WINDOW_SEC = 10.0  # Show last 10 seconds of data in plot

# Debug: Print loaded config
print(f"[DEBUG] Loaded config:")
print(f"  API_KEY: {API_KEY[:10]}..." if len(API_KEY) > 10 else f"  API_KEY: {API_KEY}")
print(f"  BACKEND_WS_BASE: {BACKEND_WS_BASE}")
print(f"  .env file path: {SCRIPT_DIR / '.env'}")
print(f"  .env exists: {(SCRIPT_DIR / '.env').exists()}")

WS_MAX_SIZE = 2**22

# -------------------- Helpers --------------------
def build_ws_url() -> str:
    from urllib.parse import urlencode
    params = {"api_key": API_KEY, "client": CLIENT}
    if OBJECT_ID:
        params["objectId"] = OBJECT_ID
    if CALLBACK_URL:
        params["callback_url"] = CALLBACK_URL
    return f"{BACKEND_WS_BASE.rstrip('/')}/?{urlencode(params)}"

def encode_frame(frame: np.ndarray) -> str:
    if FRAME_FORMAT == "raw":
        # PNG raw base64
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("PNG encode failed")
        return base64_b64(buf.tobytes())
    else:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64_b64(buf.tobytes())

def base64_b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")

def build_payload(ts: float, frame_b64: str) -> Dict[str, Any]:
    return {
        "datapt_id": str(uuid.uuid4()),
        "state": "stream",
        "timestamp": f"{ts:.6f}",
        "frame_data": frame_b64,
        "advanced": True,
    }

# Flexible extraction from server JSON
def extract_rppg(obj: Any) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not isinstance(obj, dict):
        return None, None
    
    # Check under "advanced" field (your server format)
    advanced = obj.get("advanced")
    if isinstance(advanced, dict) and "rppg" in advanced:
        rppg_list = advanced.get("rppg")
        if isinstance(rppg_list, (list, tuple)) and len(rppg_list) > 0:
            try:
                arr = np.asarray(rppg_list, dtype=float)
                # Infer fs from rppg_timestamps if available
                timestamps = advanced.get("rppg_timestamps")
                fs = None
                if isinstance(timestamps, (list, tuple)) and len(timestamps) > 1:
                    dt = timestamps[-1] - timestamps[0]
                    if dt > 0:
                        fs = (len(timestamps) - 1) / dt
                return arr, fs
            except Exception:
                pass
    
    # direct fields
    if "rppg" in obj and isinstance(obj["rppg"], (list, tuple)):
        fs = obj.get("fs")
        try:
            arr = np.asarray(obj["rppg"], dtype=float)
        except Exception:
            return None, None
        return arr, float(fs) if isinstance(fs, (int, float)) else None
    
    # nested under data
    data = obj.get("data")
    if isinstance(data, dict) and "rppg" in data:
        fs = data.get("fs")
        try:
            arr = np.asarray(data["rppg"], dtype=float)
        except Exception:
            return None, None
        return arr, float(fs) if isinstance(fs, (int, float)) else None
    
    return None, None

# -------------------- Live Pipeline --------------------
async def live_pipeline():
    ws_url = build_ws_url()
    print("Connecting WS:", ws_url)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if SHOW_PREVIEW:
        cv2.namedWindow("Live Pipeline", cv2.WINDOW_NORMAL)

    # Setup live plot if enabled
    plot_data = {
        'times': deque(maxlen=1000),
        'rppg': deque(maxlen=1000),
        'ppg': deque(maxlen=1000),
        't0': time.time(),
        'last_plot_update': 0,
        'update_interval': 0.1  # Update plot every 100ms max
    }
    
    if SHOW_PLOT:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle('Live rPPG and PPG Signals')
        
        line_rppg, = ax1.plot([], [], 'r-', label='rPPG (raw)', linewidth=1.5)
        ax1.set_ylabel('rPPG', fontsize=10)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        line_ppg, = ax2.plot([], [], 'b-', label='PPG (cleaned)', linewidth=1.5)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('PPG', fontsize=10)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.canvas.draw()
        plt.show(block=False)

    # Prepare CSV header
    write_header = not CSV_PATH.exists()
    f = CSV_PATH.open("a", buffering=1)
    if write_header:
        f.write("time_s,rppg,ppg\n")

    rppg_buffer: list[float] = []
    fs_inferred: Optional[float] = None
    last_server_message_ts = time.perf_counter()
    csv_sample_counter = 0  # Track number of samples written to CSV
    csv_start_time = time.time()  # Reference time for CSV timestamps

    try:
        async with websockets.connect(ws_url, max_size=WS_MAX_SIZE, compression=None) as ws:
            print("WS connected.")

            async def send_frames():
                sent = 0
                start = time.perf_counter()
                while True:
                    ts = time.time()
                    ok, frame = cap.read()
                    if not ok:
                        await asyncio.sleep(0.005)
                        continue

                    if SHOW_PREVIEW:
                        cv2.imshow("Live Pipeline", frame)
                        cv2.waitKey(1)

                    frame_b64 = encode_frame(frame)
                    payload = build_payload(ts, frame_b64)
                    await ws.send(orjson.dumps(payload).decode("utf-8"))
                    sent += 1
                    # regulate fps
                    next_target = start + sent / FPS
                    delay = next_target - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if sent % PRINT_FPS_INTERVAL == 0:
                        elapsed = time.perf_counter() - start
                        print(f">> sent {sent} frames @ {sent/elapsed:.2f} fps")

            async def receive_server():
                nonlocal fs_inferred
                nonlocal last_server_message_ts
                nonlocal csv_sample_counter
                msg_count = 0
                while True:
                    try:
                        msg = await ws.recv()
                    except websockets.exceptions.ConnectionClosed:
                        break
                    last_server_message_ts = time.perf_counter()
                    msg_count += 1
                    
                    try:
                        obj = orjson.loads(msg) if isinstance(msg, str) else orjson.loads(msg.decode("utf-8", errors="ignore"))
                    except Exception as e:
                        print(f"[!] JSON parse error: {e}")
                        continue
                    
                    # Debug: print first few messages
                    if msg_count <= 3:
                        print(f"[DEBUG] Server message #{msg_count}: {obj}")
                    
                    arr, fs = extract_rppg(obj)
                    if arr is None or arr.size == 0:
                        if msg_count <= 3:
                            print(f"[DEBUG] No rPPG data extracted from message #{msg_count}")
                        continue
                    
                    print(f"[+] Received rPPG data: {arr.size} samples @ {fs} Hz")
                    
                    if fs and not fs_inferred:
                        fs_inferred = fs
                    # append to buffer
                    rppg_buffer.extend(arr.tolist())
                    # trim buffer
                    if fs_inferred:
                        max_samples = int(fs_inferred * MAX_RPPG_BUFFER_SEC)
                        if len(rppg_buffer) > max_samples:
                            del rppg_buffer[:-max_samples]
                    # Clean latest segment
                    if fs_inferred and len(rppg_buffer) > 30:
                        raw = np.asarray(rppg_buffer, dtype=float)
                        try:
                            clean, info = clean_bvp(raw, fs=fs_inferred, options=CleanOptions())
                            
                            # Ensure we don't exceed array bounds
                            n_samples = min(arr.size, len(raw), len(clean))
                            
                            # Write only last received portion aligned with arr length
                            tail_raw = raw[-n_samples:]
                            tail_clean = clean[-n_samples:]
                            
                            # Calculate proper timestamps based on sampling rate
                            dt = 1.0 / fs_inferred
                            
                            # Batch write CSV and collect plot data
                            new_times = []
                            new_rppg = []
                            new_ppg = []
                            
                            for i in range(n_samples):
                                # Use continuous timestamp based on sample counter
                                t_rel = csv_start_time + (csv_sample_counter * dt)
                                csv_sample_counter += 1
                                
                                f.write(f"{t_rel:.6f},{tail_raw[i]:.6f},{tail_clean[i]:.6f}\n")
                                
                                # Collect plot data
                                if SHOW_PLOT:
                                    t_plot = t_rel - plot_data['t0']
                                    new_times.append(t_plot)
                                    new_rppg.append(tail_raw[i])
                                    new_ppg.append(tail_clean[i])
                            
                            f.flush()  # Ensure data is written to disk
                            
                            # Update plot data and visualization (batch update)
                            if SHOW_PLOT and new_times:
                                # Add new data to deques
                                plot_data['times'].extend(new_times)
                                plot_data['rppg'].extend(new_rppg)
                                plot_data['ppg'].extend(new_ppg)
                                
                                # Only update plot if enough time has passed (throttle updates)
                                current_time = time.time()
                                if current_time - plot_data['last_plot_update'] >= plot_data['update_interval']:
                                    plot_data['last_plot_update'] = current_time
                                    
                                    # Convert to lists for plotting
                                    times = list(plot_data['times'])
                                    rppg_vals = list(plot_data['rppg'])
                                    ppg_vals = list(plot_data['ppg'])
                                    
                                    if len(times) > 1:
                                        # Show only last PLOT_WINDOW_SEC seconds
                                        cutoff_time = times[-1] - PLOT_WINDOW_SEC
                                        visible_indices = [i for i, t in enumerate(times) if t >= cutoff_time]
                                        
                                        if len(visible_indices) > 1:
                                            vis_times = [times[i] for i in visible_indices]
                                            vis_rppg = [rppg_vals[i] for i in visible_indices]
                                            vis_ppg = [ppg_vals[i] for i in visible_indices]
                                            
                                            # Update plot lines
                                            line_rppg.set_data(vis_times, vis_rppg)
                                            line_ppg.set_data(vis_times, vis_ppg)
                                            
                                            # Set fixed x-axis limits for smooth scrolling
                                            ax1.set_xlim(vis_times[0], vis_times[-1])
                                            ax2.set_xlim(vis_times[0], vis_times[-1])
                                            
                                            # Auto-scale y-axis with padding
                                            rppg_range = max(vis_rppg) - min(vis_rppg)
                                            ppg_range = max(vis_ppg) - min(vis_ppg)
                                            rppg_pad = max(0.1, rppg_range * 0.1)
                                            ppg_pad = max(0.1, ppg_range * 0.1)
                                            
                                            ax1.set_ylim(min(vis_rppg) - rppg_pad, max(vis_rppg) + rppg_pad)
                                            ax2.set_ylim(min(vis_ppg) - ppg_pad, max(vis_ppg) + ppg_pad)
                                            
                                            # Use blit for faster rendering
                                            fig.canvas.draw_idle()
                                            fig.canvas.flush_events()
                        except Exception as e:
                            print(f"Cleaning error: {e}")

            send_task = asyncio.create_task(send_frames())
            recv_task = asyncio.create_task(receive_server())

            try:
                await asyncio.gather(send_task, recv_task)
            finally:
                f.close()
                cap.release()
                if SHOW_PREVIEW:
                    cv2.destroyAllWindows()
                if SHOW_PLOT:
                    plt.close('all')
                print("Pipeline closed. CSV saved.")
    except Exception as e:
        print(f"Error: {e}")
        f.close()
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        if SHOW_PLOT:
            plt.close('all')

# -------------------- Entry --------------------
if __name__ == "__main__":
    try:
        asyncio.run(live_pipeline())
    except KeyboardInterrupt:
        print("Interrupted by user.")
