"""
Simple CHROM-based rPPG extractor using OpenCV for face detection.
This is a lightweight fallback when pyVHR is not available.

References:
- de Haan & Jeanne (2013) Robust pulse rate from chrominance-based rPPG.

Note: For best accuracy and robustness, we recommend pyVHR. This module is
intended for quick demos and small experiments.
"""
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ExtractOptions:
    max_frames: Optional[int] = None  # limit for quick runs
    scale_factor: float = 1.2
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (80, 80)
    roi_ratio: Tuple[float, float, float, float] = (0.55, 0.85, 0.2, 0.8)  # y1,y2,x1,x2 within face box


def _mean_rgb_in_roi(frame: np.ndarray, x: int, y: int, w: int, h: int, roi_ratio) -> np.ndarray:
    H, W = frame.shape[:2]
    y1 = int(y + roi_ratio[0] * h)
    y2 = int(y + roi_ratio[1] * h)
    x1 = int(x + roi_ratio[2] * w)
    x2 = int(x + roi_ratio[3] * w)
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    # OpenCV is BGR; convert to RGB mean
    b, g, r = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    return np.array([np.mean(r), np.mean(g), np.mean(b)], dtype=float)


def extract_rppg_from_video(
    source: str | int = 0,
    *,
    options: Optional[ExtractOptions] = None,
    show_preview: bool = True,
) -> Tuple[np.ndarray, float]:
    """Extract rPPG via CHROM from a webcam (int) or video file (path).

    Returns: (rppg_signal, fs)
    """
    if options is None:
        options = ExtractOptions()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fs = cap.get(cv2.CAP_PROP_FPS)
    if not fs or fs <= 0:
        fs = 30.0  # fallback

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection")

    rgb_means = []
    frames = 0
    
    if show_preview:
        print("\n" + "="*60)
        print("CAMERA PREVIEW ACTIVE")
        print("="*60)
        print("Instructions:")
        print("  • Look at the camera preview window")
        print("  • Keep your face in the green box")
        print("  • Stay still with good lighting")
        print("  • Press 'q' to stop recording (recommended after 15-30 sec)")
        print("  • Or press Ctrl+C in the terminal")
        print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if options.max_frames and frames > options.max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=options.scale_factor,
            minNeighbors=options.min_neighbors,
            minSize=options.min_size,
        )
        
        # Show preview with face detection visualization
        if show_preview:
            display_frame = frame.copy()
            if len(faces) > 0:
                # Draw green box around detected face
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Draw ROI region
                y1 = int(y + options.roi_ratio[0] * h)
                y2 = int(y + options.roi_ratio[1] * h)
                x1 = int(x + options.roi_ratio[2] * w)
                x2 = int(x + options.roi_ratio[3] * w)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(display_frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No Face Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show frame count and instructions
            cv2.putText(display_frame, f"Frames: {frames}", (10, display_frame.shape[0]-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to stop", (10, display_frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('rPPG Extraction - Camera Feed', display_frame)
            
            # Check for 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nStopped by user. Captured {frames} frames.")
                break
        
        if len(faces) == 0:
            # keep previous value or append NaNs
            if len(rgb_means) > 0:
                rgb_means.append(rgb_means[-1])
            else:
                rgb_means.append(np.array([np.nan, np.nan, np.nan], dtype=float))
            continue

        # choose the largest detected face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        rgb_mean = _mean_rgb_in_roi(frame, x, y, w, h, options.roi_ratio)
        rgb_means.append(rgb_mean)

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    rgb = np.array(rgb_means, dtype=float)
    # interpolate NaNs if any
    for c in range(3):
        col = rgb[:, c]
        nans = np.isnan(col)
        if np.any(nans):
            idx = np.where(~nans)[0]
            if idx.size == 0:
                raise RuntimeError("No valid RGB samples extracted")
            rgb[nans, c] = np.interp(np.where(nans)[0], idx, col[idx])

    # Normalize each frame by its mean to reduce illumination variations
    mean_rgb = np.mean(rgb, axis=1, keepdims=True)
    mean_rgb[mean_rgb == 0] = 1.0
    rgb_norm = rgb / mean_rgb

    # CHROM projection
    u = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
    v = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
    std_u = np.std(u) or 1.0
    std_v = np.std(v) or 1.0
    alpha = std_u / std_v
    s = u - alpha * v

    # Remove DC component
    s = s - np.mean(s)

    return s.astype(float), float(fs)
