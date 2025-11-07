# rPPG to Clean PPG (Standalone)

This folder contains a small, self-contained toolkit to turn a noisy rPPG trace into a cleaner, PPG-like waveform and a simple fallback extractor based on CHROM. It is kept separate from the Flask server.

## Recommendation: pyVHR first
If you want the most accurate and reliable rPPG extraction with minimal steps, start with [pyVHR](https://github.com/phuselab/pyVHR):
- Implements strong, training-free algorithms (CHROM, POS, etc.)
- End-to-end pipeline (face detection → skin selection → rPPG → HR)
- Good defaults for demos

You can use pyVHR to extract an rPPG/BVP signal from video, then feed it to our cleaner for an easy win. The fallback CHROM extractor included here is for quick demos only.

## Install (Windows PowerShell)
Create/activate a virtual environment (optional) and install minimal deps:

```powershell
# from repo root
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r .\src\rppg\requirements.txt

# Optional (preferred extractor)
# pip install pyVHR
```

## Option A: Clean an existing rPPG signal (CSV)
Provide a CSV with either:
- one column: rPPG samples (use --fs to specify sampling rate), or
- two columns: time_s, rppg

```powershell
python -m rppg.cli --input-csv .\data\rppg.csv --fs 30 --output .\data\clean_ppg.csv --plot
```

## Option B: Quick CHROM extraction (fallback) + cleaning
Use your webcam or a video file. This does not require pyVHR.

```powershell
# Webcam
python -m rppg.cli --webcam --output .\data\clean_ppg.csv --plot

# Video file
python -m rppg.cli --video .\data\face_clip.mp4 --output .\data\clean_ppg.csv --plot
```

Notes:
- Ensure good lighting and a stable face in frame.
- The first seconds may be unstable as the filter buffers fill.

## How the cleaning works
We apply a conservative signal-processing pipeline commonly used for PPG:
- Linear detrending (remove slow drifts)
- Butterworth band-pass 0.7–4.0 Hz (≈42–240 bpm)
- Z-score normalization
- Optional short moving-average smoothing

You can tune these via CLI flags: `--low`, `--high`, `--order`, `--smooth`.

## Swapping in pyVHR
If you prefer pyVHR for extraction, use it to generate an rPPG/BVP array/CSV. Then run Option A above to clean it here. This separation keeps your experiment fast and modular.

## Outputs
- `clean_ppg.csv`: two columns: time_s, ppg
- Optional live plot comparing raw rPPG vs cleaned PPG

## Troubleshooting
- Webcam access denied: grant permission to your terminal/IDE or run as admin.
- No face detected: improve lighting or move closer to camera.
- Empty/NaN output: try a recorded video and verify face visibility.

## License
This submodule inherits the repository license.


Perfect! Now run the same command again:

What you'll see now:

Terminal instructions will print telling you what to do

A camera preview window will pop up showing:

Your live webcam feed
A green box around your face when detected
A yellow box showing the region being analyzed (forehead/cheek area)
Frame count at the bottom
"Press 'q' to stop" instruction
How to use it:

Keep your face in the green box
Stay relatively still
Record for 15-30 seconds (that's about 450-900 frames at 30fps)
Press 'q' key in the camera window to stop (recommended)
Or press Ctrl+C in the terminal
After you stop:

The camera window will close
Processing will complete
You'll see a plot comparing raw vs cleaned PPG signal
clean_ppg.csv file will be saved
Tip: The longer you record (up to 30-60 seconds), the better the heart rate signal will be! Try to have good lighting and stay still.
