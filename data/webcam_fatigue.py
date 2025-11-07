"""Webcam live demo: extract rPPG (CHROM) + facial cues and compute fatigue score.

Usage:
  python3 webcam_fatigue.py

Requires OpenCV. If `mediapipe` is installed, the script computes blink_rate,
blink duration, mouth-open fraction, and head pitch std; otherwise it uses
very simple fallbacks (face box movement) to estimate drowsiness cues.

This script computes scores every `window_seconds` (default 15s) and prints
alerts. It's intentionally minimal to be easy to run on a laptop.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np
try:
    import cv2
    HAS_CV2 = True
except Exception:
    # Lightweight shim so the script can run in simulated mode when OpenCV
    # isn't available in the environment (useful for remote/CI runs).
    HAS_CV2 = False
    class _CV2Shim:
        COLOR_BGR2RGB = 1
        COLOR_GRAY2BGR = 2
        COLOR_BGR2GRAY = 3

        @staticmethod
        def cvtColor(img, flag):
            # Provide minimal conversions used by this script: GRAY->BGR,
            # BGR->GRAY and BGR->RGB. Works with numpy arrays.
            if flag == _CV2Shim.COLOR_GRAY2BGR:
                if img.ndim == 2:
                    return np.stack([img, img, img], axis=-1)
            if flag == _CV2Shim.COLOR_BGR2GRAY:
                if img.ndim == 3:
                    # approximate luminance from BGR channels
                    b = img[..., 0].astype(np.float32)
                    g = img[..., 1].astype(np.float32)
                    r = img[..., 2].astype(np.float32)
                    gray = (0.114 * b + 0.587 * g + 0.299 * r)
                    return gray.astype(img.dtype)
            if flag == _CV2Shim.COLOR_BGR2RGB:
                if img.ndim == 3:
                    return img[..., ::-1]
            return img

    cv2 = _CV2Shim()

from fatigue import combined_fatigue_score, alert_from_score
import argparse
import csv
from dataclasses import asdict

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

if HAS_MEDIAPIPE:
    # initialize a single FaceMesh instance for reuse
    _MP_FACE = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               refine_landmarks=True,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    def _to_pixel(landmark, w, h):
        return np.array([landmark.x * w, landmark.y * h])

    def mediapipe_facial_features(frames: Deque[np.ndarray], fs: float) -> Dict[str, float]:
        """Compute facial features using Mediapipe FaceMesh over buffered frames.

        Returns the same dict shape as `simple_facial_heuristics` but with
        more accurate blink/yawn/head-pose measures.
        """
        if len(frames) == 0:
            return {}
        # indices (FaceMesh) commonly used
        L_EYE = [33, 160, 158, 133, 153, 144]
        R_EYE = [362, 385, 387, 263, 373, 380]
        MOUTH_INNER = [13, 14]
        MOUTH_CORNERS = [78, 308]
        NOSE_TIP = 1
        CHIN = 152

        ear_list = []
        mar_list = []
        pitch_list = []

        h, w = frames[0].shape[:2]

        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            results = _MP_FACE.process(rgb)
            if not results.multi_face_landmarks:
                continue
            lm = results.multi_face_landmarks[0].landmark
            def dist(a, b):
                return np.linalg.norm(a - b)

            # eye EAR
            try:
                le = np.array([_to_pixel(lm[i], w, h) for i in L_EYE])
                re = np.array([_to_pixel(lm[i], w, h) for i in R_EYE])
                # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
                left_ear = (dist(le[1], le[5]) + dist(le[2], le[4])) / (2.0 * dist(le[0], le[3]) + 1e-6)
                right_ear = (dist(re[1], re[5]) + dist(re[2], re[4])) / (2.0 * dist(re[0], re[3]) + 1e-6)
                ear = float((left_ear + right_ear) / 2.0)
                ear_list.append(ear)
            except Exception:
                pass

            # mouth MAR
            try:
                m_in = np.array([_to_pixel(lm[i], w, h) for i in MOUTH_INNER])
                m_c = np.array([_to_pixel(lm[i], w, h) for i in MOUTH_CORNERS])
                mar = float(dist(m_in[0], m_in[1]) / (dist(m_c[0], m_c[1]) + 1e-6))
                mar_list.append(mar)
            except Exception:
                pass

            # head pitch approx: vector from nose tip to chin (pixel coords)
            try:
                nose = _to_pixel(lm[NOSE_TIP], w, h)
                chin = _to_pixel(lm[CHIN], w, h)
                vec = chin - nose
                # pitch angle (degrees) ~ atan2(dy, norm(vec))
                pitch = float(np.degrees(np.arctan2(vec[1], np.linalg.norm(vec) + 1e-6)))
                pitch_list.append(pitch)
            except Exception:
                pass

        # Now compute aggregates
        fps = float(fs) if fs > 0 else 30.0
        # smoothing helper for lists
        def _smooth_list(arr, k=3):
            arr = np.asarray(arr)
            if arr.size < k:
                return arr
            return np.convolve(arr, np.ones(k) / float(k), mode='same')

        # smooth EAR/MAR/pitch to reduce jitter
        if len(ear_list) > 0:
            ear_list = _smooth_list(ear_list, k=max(3, int(0.02 * fps)))
        if len(mar_list) > 0:
            mar_list = _smooth_list(mar_list, k=max(3, int(0.02 * fps)))
        if len(pitch_list) > 0:
            pitch_list = _smooth_list(pitch_list, k=max(3, int(0.02 * fps)))
        # blink detection from EAR: threshold & durations
        EAR_THRESH = 0.23
        MIN_BLINK_FRAMES = max(1, int(0.05 * fps))
        MAX_BLINK_FRAMES = max(1, int(0.4 * fps))

        blink_count = 0
        blink_durations = []
        if len(ear_list) > 0:
            ear_array = np.array(ear_list)
            closed = ear_array < EAR_THRESH
            # find consecutive closed runs
            i = 0
            while i < len(closed):
                if closed[i]:
                    j = i
                    while j < len(closed) and closed[j]:
                        j += 1
                    dur_frames = j - i
                    if MIN_BLINK_FRAMES <= dur_frames <= MAX_BLINK_FRAMES:
                        blink_count += 1
                        blink_durations.append(dur_frames / fps)
                    i = j
                else:
                    i += 1

        blink_rate = float(blink_count * (60.0 / (max(1.0, len(frames) / fps)))) if len(frames) > 0 else 0.0
        avg_blink_duration = float(np.mean(blink_durations)) if blink_durations else 0.0

        # mouth/yawn
        MAR_THRESH = 0.6
        YAWN_MIN_SECONDS = 0.8
        yawns = 0
        mouth_open_frames = 0
        if len(mar_list) > 0:
            mar_array = np.array(mar_list)
            open_mask = mar_array > MAR_THRESH
            mouth_open_frames = int(np.sum(open_mask))
            # detect long opens
            i = 0
            while i < len(open_mask):
                if open_mask[i]:
                    j = i
                    while j < len(open_mask) and open_mask[j]:
                        j += 1
                    dur = (j - i) / fps
                    if dur >= YAWN_MIN_SECONDS:
                        yawns += 1
                    i = j
                else:
                    i += 1

        mouth_open_duration = float(mouth_open_frames / max(1, len(frames)))

        head_pitch_std = float(np.std(pitch_list)) if len(pitch_list) > 0 else 0.0
        # count head nods as large negative pitch excursions
        nod_count = 0
        if len(pitch_list) > 1:
            pitch_arr = np.array(pitch_list)
            diffs = np.diff(pitch_arr)
            # a nod shows as a sequence where pitch decreases fast beyond threshold
            for k in range(1, len(diffs)):
                if diffs[k] < -4.0 and diffs[k-1] < -1.0:
                    nod_count += 1

        return {
            "blink_rate": blink_rate,
            "avg_blink_duration": avg_blink_duration,
            "yawns_count": float(yawns),
            "mouth_open_duration": mouth_open_duration,
            "head_nod_count": float(nod_count),
            "head_pitch_std": head_pitch_std,
        }

# Ensure local ppr_project/Caire-Hackathon-Challenge is on sys.path so `rppg` imports work
import os, sys
_HERE = os.path.dirname(__file__)
_LOCAL_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _LOCAL_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_ROOT)

try:
    from rppg.chrom_extractor import extract_rppg_from_video, ExtractOptions
    HAS_RPPG = True
except Exception:
    # If the repo's CHROM extractor (which depends on OpenCV) isn't importable
    # we provide a minimal fallback that will trigger the simulated path in
    # `run_webcam` (it raises RuntimeError when called).
    HAS_RPPG = False
    def extract_rppg_from_video(source, options=None, show_preview=True):
        raise RuntimeError("rPPG extractor not available in this environment")
    class ExtractOptions:
        def __init__(self, max_frames=None):
            self.max_frames = max_frames


@dataclass
class WebcamOpts:
    source: int = 0
    window_seconds: float = 15.0
    show_preview: bool = True
    max_frames: Optional[int] = None
    # When True, if the real webcam cannot be opened the script will simulate
    # a webcam: synthetic rPPG signal (sine + noise) and simple frames that
    # approximate brightness changes (for blink heuristics). This is useful
    # for remote environments or unit tests.
    simulate_on_fail: bool = True
    simulate_fs: int = 30
    simulate_windows: int = 3
    simulate_hr: float = 72.0


def simple_facial_heuristics(frames: Deque[np.ndarray]) -> Dict[str, float]:
    """Fallback facial cues when mediapipe is not present: use bbox motion + brightness.

    Returns a dict compatible with `facial_score_from_features` input.
    """
    # very naive heuristics: compute brightness variance as proxy for eye closure
    if len(frames) == 0:
        return {}
    imgs = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    mean_brightness = np.mean(imgs, axis=(1, 2))
    # smoothing helper (moving average)
    def _smooth(x, k=3):
        x = np.asarray(x)
        if x.size < k:
            return x
        return np.convolve(x, np.ones(k) / float(k), mode='same')

    # smooth brightness trace to reduce spurious drops
    k = max(3, int(0.05 * 30))
    mean_brightness_s = _smooth(mean_brightness, k=k)
    blink_rate = float(np.clip(np.mean((mean_brightness_s[:-1] - mean_brightness_s[1:]) > 5) * 60.0, 2, 30))
    avg_blink_duration = 0.12
    yawns_count = 0.0
    mouth_open_duration = 0.0
    head_nod_count = 0.0
    head_pitch_std = float(np.std(mean_brightness) * 0.01)
    return {
        "blink_rate": blink_rate,
        "avg_blink_duration": avg_blink_duration,
        "yawns_count": yawns_count,
        "mouth_open_duration": mouth_open_duration,
        "head_nod_count": head_nod_count,
        "head_pitch_std": head_pitch_std,
    }


def run_webcam(opts: WebcamOpts):
    # Try to extract rPPG from the requested source. If the extractor raises
    # (for example because the camera device can't be opened in a remote
    # environment), fall back to a simulated rPPG+frames when configured.
    simulated = False
    try:
        rppg, fs = extract_rppg_from_video(opts.source, options=ExtractOptions(max_frames=opts.max_frames), show_preview=opts.show_preview)
    except RuntimeError:
        if not opts.simulate_on_fail:
            raise
        simulated = True

    if simulated:
        # Create synthetic rPPG and frames
        fs = float(opts.simulate_fs)
        # determine total frames to simulate
        if opts.max_frames is not None:
            total_frames = int(opts.max_frames)
        else:
            total_frames = int(fs * opts.window_seconds * opts.simulate_windows)

        # synthetic rPPG: create beats at intervals matching simulate_hr, add small jitter,
        # then convolve with a short Gaussian kernel to produce PPG-like bumps.
        dur = total_frames / fs
        rr_interval = 60.0 / float(opts.simulate_hr)
        beats = []
        tcur = 0.0
        # start with small random phase
        tcur += np.random.uniform(0.0, rr_interval * 0.1)
        while tcur < dur:
            beats.append(tcur)
            # small beat-to-beat jitter (1% std)
            jitter = 0.01 * np.random.randn()
            tcur += rr_interval * (1.0 + jitter)

        beat_idx = (np.array(beats) * fs).astype(int)
        impulse = np.zeros(total_frames)
        beat_idx = beat_idx[beat_idx < total_frames]
        impulse[beat_idx] = 1.0

        # gaussian kernel ~ 0.3s width
        kernel_dur = 0.3
        kernel_len = max(3, int(kernel_dur * fs))
        x = np.linspace(-1.0, 1.0, kernel_len)
        sigma = 0.25
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)

        rppg = np.convolve(impulse, kernel, mode='same')
        # scale and add low-frequency baseline + noise
        rppg = 0.9 * (rppg - np.mean(rppg))
        rppg += 0.02 * np.random.randn(total_frames)
        rppg = rppg.astype(float)

        print(f"SIMULATED mode: generated {len(rppg)} rPPG samples @ {fs:.1f} Hz -> {int(total_frames // (fs * opts.window_seconds))} windows of {opts.window_seconds}s")

        # build synthetic frames: vary brightness slowly and include occasional dips as 'blinks'
        h, w = 480, 640
        simulated_frames = []
        # simple blink schedule: a blink every ~3-5 seconds
        blink_interval = int(fs * 4.0)
        for i in range(total_frames):
            base = 128 + 10 * np.sin(2.0 * np.pi * 0.1 * (i / fs))
            # occasional blink
            if (i % blink_interval) == 0:
                bright = base - 40
            else:
                bright = base + np.random.randn() * 3.0
            bright = float(np.clip(bright, 10, 245))
            gray = np.full((h, w), int(bright), dtype=np.uint8)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            simulated_frames.append(frame)

        cap = None
        frames_buffer: Deque[np.ndarray] = deque(maxlen=int(fs * opts.window_seconds))
        # preload first window frames
        preload = int(fs * opts.window_seconds)
        for f in simulated_frames[:preload]:
            frames_buffer.append(f)
        # keep simulated_frames list for later window updates
        sim_frames_list = simulated_frames
        sim_next_idx = preload
    else:
        # Real device path
        win_len = 0  # placeholder so type checkers are happy
        print("Using real device for rPPG extraction")
        cap = cv2.VideoCapture(opts.source)
        frames_buffer: Deque[np.ndarray] = deque(maxlen=int(fs * opts.window_seconds))

        # preload frames for heuristics while we process PPG windows
        frames_read = 0
        while frames_read < int(fs * opts.window_seconds) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_buffer.append(frame)
            frames_read += 1

        # chop rppg into non-overlapping windows of window_seconds
        win_len = int(fs * opts.window_seconds)
        n = len(rppg) // win_len
        print(f"Extracted {len(rppg)} rPPG samples @ {fs:.1f} Hz -> {n} windows of {opts.window_seconds}s")

    # Process each window
    # note: if simulated, compute n from synthetic signal; otherwise n already set
    if simulated:
        win_len = int(fs * opts.window_seconds)
        n = len(rppg) // win_len
    # if CSV output requested, open writer
    csv_writer = None
    csv_file = None
    fieldnames = [
        'window_idx', 'heart_rate', 'mean_rr', 'std_rr', 'fatigue_score', 'alert_level',
        'blink_rate', 'avg_blink_duration', 'yawns_count', 'mouth_open_duration', 'head_nod_count', 'head_pitch_std'
    ]
    if getattr(opts, 'out_csv', None):
        out_path = opts.out_csv
        csv_file = open(out_path, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
        csv_writer.writeheader()

    for i in range(n):
        start = i * win_len
        end = start + win_len
        window = rppg[start:end]

        # HRV extraction:
        if simulated:
            # For simulated runs, report HRV values based on the requested simulate_hr
            # with small random jitter to emulate natural variability.
            target_hr = float(opts.simulate_hr)
            hr_jitter = np.random.randn() * max(0.5, 0.01 * target_hr)
            heart_rate = float(max(30.0, target_hr + hr_jitter))
            mean_rr = 60.0 / heart_rate
            std_rr = max(0.001, 0.02 * mean_rr)
        else:
            # We'll reuse the logic from arrhythmia project's extractor in a simplified form
            # Find peaks in window (use simple maxima)
            peaks = np.where((window[1:-1] > window[:-2]) & (window[1:-1] > window[2:]))[0] + 1
            # prune peaks that are too close together (likely noise or sub-peaks)
            if peaks.size > 0:
                min_dist_samples = max(1, int(0.4 * fs))
                pruned = [int(peaks[0])]
                for p in peaks[1:]:
                    if int(p) - pruned[-1] >= min_dist_samples:
                        pruned.append(int(p))
                peaks = np.array(pruned, dtype=int)

            if peaks.size >= 2:
                rr = np.diff(peaks) / fs
                mean_rr = float(np.mean(rr))
                std_rr = float(np.std(rr)) if rr.size > 1 else 0.0
                heart_rate = 60.0 / mean_rr if mean_rr > 0 else 0.0
            else:
                mean_rr = std_rr = heart_rate = 0.0

        hrv_feats = {"heart_rate": heart_rate, "mean_rr": mean_rr, "std_rr": std_rr}

        # Facial metrics: prefer mediapipe if available
        if HAS_MEDIAPIPE:
            try:
                facial_feats = mediapipe_facial_features(frames_buffer, fs)
            except Exception:
                facial_feats = simple_facial_heuristics(frames_buffer)
        else:
            facial_feats = simple_facial_heuristics(frames_buffer)

        score = combined_fatigue_score(facial_feats, hrv_feats, facial_weight=0.6)
        alert = alert_from_score(score, threshold=70.0)
        print(f"Window {i+1}/{n}: HR {heart_rate:.1f}, mean_rr {mean_rr:.3f}, std_rr {std_rr:.3f}")
        print(f" Facial feats: {facial_feats}")
        print(f" Fatigue score: {score:.1f} -> {alert['level'].upper()}: {alert['action']}\n")

        # advance frames buffer for next window
        # write CSV row if requested
        if csv_writer is not None:
            row = {
                'window_idx': i + 1,
                'heart_rate': round(heart_rate, 3),
                'mean_rr': round(mean_rr, 6),
                'std_rr': round(std_rr, 6),
                'fatigue_score': round(score, 3),
                'alert_level': alert['level'],
                'blink_rate': facial_feats.get('blink_rate', ''),
                'avg_blink_duration': facial_feats.get('avg_blink_duration', ''),
                'yawns_count': facial_feats.get('yawns_count', ''),
                'mouth_open_duration': facial_feats.get('mouth_open_duration', ''),
                'head_nod_count': facial_feats.get('head_nod_count', ''),
                'head_pitch_std': facial_feats.get('head_pitch_std', ''),
            }
            csv_writer.writerow(row)
        if simulated:
            # fill buffer with next win_len simulated frames (or pad last frame)
            for j in range(win_len):
                if sim_next_idx < len(sim_frames_list):
                    frames_buffer.append(sim_frames_list[sim_next_idx])
                    sim_next_idx += 1
                else:
                    # pad with last frame
                    frames_buffer.append(sim_frames_list[-1])
        else:
            # read next win_len frames from real capture to refill the buffer
            read_count = 0
            while read_count < win_len and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames_buffer.append(frame)
                read_count += 1

    if cap is not None:
        cap.release()
    if csv_writer is not None:
        csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam fatigue demo (rPPG + facial heuristics)')
    parser.add_argument('--device', type=int, default=0, help='camera device index or video file path')
    parser.add_argument('--window-seconds', type=float, default=15.0)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--no-preview', dest='show_preview', action='store_false')
    parser.add_argument('--simulate', dest='simulate_on_fail', action='store_true')
    parser.add_argument('--no-simulate', dest='simulate_on_fail', action='store_false')
    parser.add_argument('--simulate-fs', type=int, default=30)
    parser.add_argument('--simulate-windows', type=int, default=3)
    parser.add_argument('--simulate-hr', type=float, default=72.0)
    parser.add_argument('--out-csv', type=str, default=None, help='write per-window features and score to CSV')
    parser.set_defaults(simulate_on_fail=True, show_preview=True)
    args = parser.parse_args()

    opts = WebcamOpts(
        source=args.device,
        window_seconds=args.window_seconds,
        show_preview=args.show_preview,
        max_frames=args.max_frames,
        simulate_on_fail=args.simulate_on_fail,
        simulate_fs=args.simulate_fs,
        simulate_windows=args.simulate_windows,
        simulate_hr=args.simulate_hr,
    )
    # attach CSV path to opts dynamically
    setattr(opts, 'out_csv', args.out_csv)

    run_webcam(opts)
