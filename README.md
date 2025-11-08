# HackHealthAI Driver Monitoring Platform

An end-to-end in-vehicle driver health assessment system that extracts remote photoplethysmography (rPPG) signals from in-cabin video, computes heart-rate metrics, detects arrhythmias with a SiamAF model, and streams results to a React dashboard.

## Repository Layout

- `backend/` – FastAPI application exposing `/infer`, `/health`, and `/ws/arrhythmia`.
- `ml/` – Signal processing pipeline, rPPG utilities, and SiamAF wrapper.
- `frontend/` – Vite + React dashboard with live waveform and operator controls.
- `infra/` – Dockerfiles, docker-compose stack, and environment setup scripts.
- `tests/` – Pytest suite for ML utilities and backend endpoints.
- `docs/` – Architecture notes and supplementary documentation.

## Prerequisites

- Python 3.11.x
- Node.js 20+
- (Optional) CUDA-capable GPU for SiamAF inference

## Local Development

### Python Environment

```powershell
# Windows PowerShell
./infra/scripts/setup_venv.ps1

# Activate then run the backend
./.venv/Scripts/Activate.ps1
uvicorn backend.app.main:app --reload
```

```bash
# macOS / Linux
bash infra/scripts/setup_venv.sh
source .venv/bin/activate
uvicorn backend.app.main:app --reload
```

Set environment variables as needed (e.g. `MODEL_CHECKPOINT` pointing to the SiamAF weights in `ml/models/`).

### Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```

Access the dashboard at `http://localhost:3000` (configure `VITE_BACKEND_URL` to match backend host).

## Docker

```bash
cd infra
docker compose up --build
```

- Backend available at `http://localhost:8000`.
- Frontend dashboard at `http://localhost:3000`.

## Testing

```bash
source .venv/bin/activate  # or equivalent activation on Windows
pip install -r tests/requirements.txt
pytest
```

## Continuous Integration

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

- Backend unit tests under Python 3.11.
- Frontend type-checking and build using Node 20.

## Deployment Considerations

- Provide the pretrained SiamAF checkpoint in `ml/models/siamaf_pretrained.pt`.
- Configure GPU-enabled Docker runtime or set `DEVICE=cpu` in the backend environment when CUDA is unavailable.
- Harden the WebSocket endpoint with authentication before production deployment.
