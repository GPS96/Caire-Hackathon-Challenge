# System Architecture

## Overview

The driver health assessment platform consists of three primary components:

- **Backend** – FastAPI service that exposes REST and WebSocket interfaces for arrhythmia inference results.
- **ML Pipeline** – rPPG extraction and SiamAF arrhythmia classification logic orchestrated within `ml/`.
- **Frontend** – React (Vite) dashboard rendering live telemetry and control actions for operators.

## Data Flow

1. Video frames are captured from the in-cabin camera and sent to the `/infer` endpoint.
2. The backend decodes frames, runs the ML pipeline, and returns heart rate, IBI, and arrhythmia predictions.
3. Results stream to connected clients via `/ws/arrhythmia` WebSocket.
4. The dashboard visualises the waveform, vitals, and alerts while offering operator controls.
