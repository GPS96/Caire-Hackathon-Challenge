# Local Development Setup

This note captures everything needed to bring up the backend (FastAPI) and frontend (Vite) on Windows PowerShell without hunting for scattered environment variables.

## 1. Prerequisites
- Python 3.10+ on PATH (use a virtual environment if possible).
- Node.js 18+ (Vite requires a modern runtime).
- The arrhythmia dataset (`data/training/`) and model weights (`data/arrhythmia_project/weights/`) copied locally if you want the full simulation outputs. The repository does not ship them.

## 2. Backend environment
Create or update `backend/.env` with the values you need. Use placeholders for secrets:

```
APP_NAME=HackHealthAI Driver Monitor
APP_VERSION=0.1.0
SIMULATION_MODE=true
SIM_DATA_ROOT=data/training
SIM_WEIGHTS_DIR=data/arrhythmia_project/weights
SIM_SAMPLING_RATE=125
DEVICE=cpu

# LLM provider options – pick whichever backend you use
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<your-google-api-key>
GEMINI_MODEL=gemini-2.0-flash-exp
OPENAI_API_KEY=<your-openai-key-if-needed>
```

If you are missing the training data or weights you can still start the API by leaving `SIMULATION_MODE=false`; the ML endpoints will respond as `uninitialized` until a model is provided.

### PowerShell launcher snippet
From the repository root run the following once per terminal session before starting Uvicorn:

```powershell
Set-Location E:/MS_AI/Github_projects/HackHealthAI/.github/workspace-healthai
$env:PYTHONPATH            = (Get-Location)
$env:SIMULATION_MODE       = "true"
$env:SIM_DATA_ROOT         = "data/training"
$env:SIM_WEIGHTS_DIR       = "data/arrhythmia_project/weights"
$env:SIM_SAMPLING_RATE     = "125"
$env:DEVICE                = "cuda"  # use "cuda" only if a supported GPU is present

# Optional: point at your API keys for the navigation agent
$env:LLM_PROVIDER          = "gemini"
$env:GOOGLE_API_KEY        = "<update-me>"
$env:GEMINI_MODEL          = "gemini-2.0-flash-exp"
# $env:OPENAI_API_KEY      = "<update-me>"

# Install dependencies (once)
python -m pip install -r backend/requirements.txt
python -m pip install -r backend/requirements-nav.txt

# Start the backend
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Keep the terminal open while the backend is running. Visit `http://localhost:8000/health` to confirm you receive a JSON payload.

## 3. Frontend environment
Add a `.env.local` in `frontend/` (or update the existing one):

```
VITE_BACKEND_URL=http://localhost:8000
VITE_NAVIGATION_API_URL=http://localhost:8000/navigation/agent
```

### PowerShell launcher snippet
In a second PowerShell window:

```powershell
Set-Location E:/MS_AI/Github_projects/HackHealthAI/.github/workspace-healthai/frontend
npm install  # first time only
npm run dev
```

Vite defaults to `http://localhost:3000`; if that port is busy it will auto-select another one (check the console output). The dashboard UI expects the backend to be reachable at the URL defined in `VITE_BACKEND_URL`.

## 4. Quick verification steps
1. Open `http://localhost:8000/health` – expect `{ "status": "ok" | "degraded", ... }`.
2. Open the Vite URL (displayed in the dev console). Trigger the emergency navigation pane; the `/navigation/agent` requests should point at the backend.
3. If the navigation pane shows a 404, double-check the `VITE_NAVIGATION_API_URL` value and restart `npm run dev` after any env changes.

Having these snippets in one place lets you copy/paste them when you need fresh shells. Adjust the paths if you cloned the repository elsewhere.
