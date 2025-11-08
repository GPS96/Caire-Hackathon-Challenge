# webcam_fatigue_demo.py
import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from fatigue_score import compute_subscores_and_score, alert_from_score

# ---------------------------
# Config
# ---------------------------
CAM_INDEX = 0
FPS_SMOOTH = 0.9
WINDOW_SEC = 60.0        # analysis window
HOP_SEC = 2.0            # compute score every HOP_SEC
EAR_CLOSED = 0.21        # eyes "closed" threshold
EAR_MIN_FRAMES = 2       # frames under threshold to count as blink
MAR_YAWN = 0.65          # mouth open threshold for yawns
YAWN_MIN_SEC = 1.0       # min continuous seconds to count yawn
NOD_DOWN_DEG = 12.0      # head drop detection (nod)
NOD_MAX_INTERVAL = 2.0   # seconds within which down-up counts as nod

# ---------------------------
# MediaPipe face mesh setup
# ---------------------------
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# FaceMesh indices (Eye and Mouth landmarks)
# Using a common subset; these are from the FaceMesh topology.
# You can tweak indices if your MediaPipe version differs.
# Left eye: [33, 160, 158, 133, 153, 144]  (approx)
# Right eye: [263, 387, 385, 362, 380, 373]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
# Mouth (outer): use 13 (upper lip) and 14 (lower lip), plus horizontal 78/308
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
# Nose tip and ear points for a rough head pitch proxy
NOSE_TIP = 1
FOREHEAD = 10  # not perfect, but okay proxy for pitch changes

# ---------------------------
# Helpers
# ---------------------------
def euclid(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(landmarks, eye_idx):
    # EAR = (||p2-p6||+||p3-p5||)/(2*||p1-p4||)
    # map indices: [p1, p2, p3, p4, p5, p6] -> [0,1,2,3,4,5]
    pts = [landmarks[i] for i in eye_idx]
    p1,p2,p3,p4,p5,p6 = pts
    vert = euclid(p2, p6) + euclid(p3, p5)
    horiz = 2.0 * euclid(p1, p4)
    if horiz <= 1e-6:
        return 0.0
    return vert / horiz

def mouth_aspect_ratio(landmarks):
    # MAR = ||top-bottom|| / ||left-right||
    top = landmarks[MOUTH_TOP]
    bot = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    num = euclid(top, bot)
    den = euclid(left, right)
    if den <= 1e-6:
        return 0.0
    return num / den

def head_pitch_proxy(landmarks):
    # Rough pitch proxy: vertical delta between "forehead" and nose tip
    # Positive when head tilts down (nose closer to bottom)
    nose = landmarks[NOSE_TIP]
    fore = landmarks[FOREHEAD]
    return (nose[1] - fore[1]) * 180.0  # scaled proxy, not true degrees

# ---------------------------
# Rolling buffers
# ---------------------------
class Rolling:
    def __init__(self, seconds):
        self.seconds = seconds
        self.buf = deque()  # (t, dict)
    def add(self, t, **vals):
        self.buf.append((t, vals))
        self.trim(t)
    def trim(self, now):
        while self.buf and (now - self.buf[0][0] > self.seconds):
            self.buf.popleft()
    def window(self):
        return list(self.buf)

# Blink state machine
class BlinkMeter:
    def __init__(self, closed_thresh=EAR_CLOSED, min_frames=EAR_MIN_FRAMES):
        self.closed_thresh = closed_thresh
        self.min_frames = min_frames
        self.run = 0
        self.t0 = None
        self.blinks = []  # list of (start_t, dur)
    def update(self, t, ear, frame_dt):
        if ear < self.closed_thresh:
            if self.run == 0:
                self.t0 = t
            self.run += 1
        else:
            if self.run >= self.min_frames and self.t0 is not None:
                self.blinks.append((self.t0, t - self.t0))
            self.run = 0
            self.t0 = None
    def stats_in_window(self, t_now, window_s):
        # drop old blinks
        self.blinks = [(ts, dur) for (ts, dur) in self.blinks if (t_now - ts) <= window_s]
        total_time = max(1e-6, min(window_s, t_now - (t_now - window_s)))
        count = len(self.blinks)
        rate_pm = (count / total_time) * 60.0
        mean_dur = float(np.mean([d for _, d in self.blinks])) if self.blinks else 0.0
        return rate_pm, mean_dur

# Yawn meter
class YawnMeter:
    def __init__(self, mar_thresh=MAR_YAWN, min_sec=YAWN_MIN_SEC):
        self.mar_thresh = mar_thresh
        self.min_sec = min_sec
        self.up = False
        self.t_start = 0.0
        self.yawns = []  # (start_t, dur)
        self.mouth_open_time = 0.0
        self.last_t = None
    def update(self, t, mar):
        if self.last_t is not None and mar > self.mar_thresh:
            self.mouth_open_time += (t - self.last_t)
        self.last_t = t

        if mar > self.mar_thresh and not self.up:
            self.up = True
            self.t_start = t
        if self.up and mar <= self.mar_thresh:
            dur = t - self.t_start
            if dur >= self.min_sec:
                self.yawns.append((self.t_start, dur))
            self.up = False
    def stats_in_window(self, t_now, window_s):
        self.yawns = [(ts, d) for (ts, d) in self.yawns if (t_now - ts) <= window_s]
        mouth_open_recent = min(self.mouth_open_time, window_s)
        mouth_frac = mouth_open_recent / max(window_s, 1e-6)
        return len(self.yawns), mouth_frac

# Head nod meter (pitch down then up within interval)
class NodMeter:
    def __init__(self, down_deg=NOD_DOWN_DEG, max_interval=NOD_MAX_INTERVAL):
        self.down_deg = down_deg
        self.max_interval = max_interval
        self.state = "UP"
        self.last_down_t = None
        self.nods = []
        self.pitch_samples = deque()  # (t, pitch_proxy)
    def update(self, t, pitch):
        # store pitch for std
        self.pitch_samples.append((t, pitch))
        while self.pitch_samples and (t - self.pitch_samples[0][0]) > WINDOW_SEC:
            self.pitch_samples.popleft()

        # simple nod detection: look for quick increase (down) then back
        if self.state == "UP":
            # detect a sharp positive delta
            if len(self.pitch_samples) >= 2:
                dt = self.pitch_samples[-1][0] - self.pitch_samples[-2][0]
                dp = self.pitch_samples[-1][1] - self.pitch_samples[-2][1]
                if dp > self.down_deg * (dt / 0.5):  # scaled by dt
                    self.state = "DOWN"
                    self.last_down_t = t
        elif self.state == "DOWN":
            # return to up within time
            if t - (self.last_down_t or t) <= self.max_interval:
                if len(self.pitch_samples) >= 2:
                    dp = self.pitch_samples[-2][1] - self.pitch_samples[-1][1]
                    if dp > self.down_deg * 0.5:
                        self.nods.append(t)
                        self.state = "UP"
            else:
                self.state = "UP"
    def stats_in_window(self, t_now, window_s):
        self.nods = [ts for ts in self.nods if (t_now - ts) <= window_s]
        # pitch std over window
        ps = [p for (_, p) in self.pitch_samples if (t_now - _) <= window_s]
        pitch_std = float(np.std(ps)) if ps else 0.0
        return len(self.nods), pitch_std

def draw_overlay(frame, score, level, subs):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w-10, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Fatigue Score: {score:.1f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if score < 70 else (0, 0, 255), 2)
    cv2.putText(frame, f"Level: {level.upper()}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # small bar
    bar_x, bar_y, bar_w, bar_h = 20, 100, w-40, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80, 80, 80), 2)
    fill_w = int(bar_w * (score/100.0))
    color = (0, 200, 0) if score < 60 else ((0, 200, 255) if score < 75 else (0, 0, 255))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), color, -1)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    roll = Rolling(WINDOW_SEC)
    blink_meter = BlinkMeter()
    yawn_meter = YawnMeter()
    nod_meter = NodMeter()

    last_t = time.time()
    last_hop = last_t
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time()
        dt = t - last_t
        last_t = t
        fps = FPS_SMOOTH*fps + (1-FPS_SMOOTH)*(1.0/max(dt,1e-6))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        ear_val, mar_val, pitch_val = None, None, None

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]
            ih, iw = frame.shape[0], frame.shape[1]
            # Collect 2D points
            pts = [(lm.x*iw, lm.y*ih) for lm in face.landmark]

            # EAR and MAR
            ear_left = eye_aspect_ratio(pts, LEFT_EYE)
            ear_right = eye_aspect_ratio(pts, RIGHT_EYE)
            ear_val = (ear_left + ear_right) / 2.0
            mar_val = mouth_aspect_ratio(pts)
            pitch_val = head_pitch_proxy(pts)

            # Update meters
            blink_meter.update(t, ear_val, dt)
            yawn_meter.update(t, mar_val)
            nod_meter.update(t, pitch_val)

            # For visualization, draw simple contours (optional)
            for i in LEFT_EYE + RIGHT_EYE + [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]:
                x, y = int(pts[i][0]), int(pts[i][1])
                cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)

        # Every frame, append minimal to rolling buffer (we only use time for windowing here)
        roll.add(t, dummy=1)

        # Compute score every hop
        if (t - last_hop) >= HOP_SEC:
            last_hop = t

            # Extract window features
            blink_rate, avg_blink_dur = blink_meter.stats_in_window(t, WINDOW_SEC)
            yawns, mouth_frac = yawn_meter.stats_in_window(t, WINDOW_SEC)
            nods, pitch_std = nod_meter.stats_in_window(t, WINDOW_SEC)

            facial_feats = {
                "blink_rate": float(blink_rate),
                "avg_blink_duration": float(avg_blink_dur),
                "yawns_count": float(yawns),
                "mouth_open_duration": float(mouth_frac),
                "head_nod_count": float(nods),
                "head_pitch_std": float(pitch_std),
            }

            # HRV placeholder (if you have Caire.ai HRV, replace this dict)
            # Example: hrv_feats = {"heart_rate": hr_bpm, "mean_rr": mean_rr_s, "std_rr": std_rr_s, "rmssd": rmssd_ms}
            hrv_feats = {"heart_rate": 0.0, "mean_rr": 0.0, "std_rr": 0.0}

            score, subs = compute_subscores_and_score(
                facial_feats=facial_feats,
                hrv_feats=hrv_feats,
                facial_weight=0.8  # rely mostly on facial features when HRV is empty
            )
            decision = alert_from_score(score)

            # Stash for overlay
            frame._fatigue_score = score
            frame._fatigue_level = decision["level"]
            frame._subs = subs

        # Draw overlay if available
        score = getattr(frame, "_fatigue_score", None)
        level = getattr(frame, "_fatigue_level", None)
        if score is not None and level is not None:
            draw_overlay(frame, score, level, getattr(frame, "_subs", {}))

        # FPS
        cv2.putText(frame, f"FPS: {fps:4.1f}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Driver Fatigue (demo)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
