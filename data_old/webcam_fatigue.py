"""
Facial-only webcam fatigue demo using fatigue.py.
Install: pip install opencv-python mediapipe numpy
Run:     python webcam_fatigue.py
Press 'q' to quit.
"""
import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from fatigue import compute_subscores_and_score, alert_from_score

# -------- Tunables --------
CAM_INDEX = 0
WINDOW_SEC = 60.0
HOP_SEC = 2.0

EAR_CLOSED = 0.21      # eye aspect ratio threshold (try 0.19–0.20 if blinks not detected)
EAR_MIN_FRAMES = 2
MAR_YAWN = 0.70        # mouth aspect ratio threshold
YAWN_MIN_SEC = 1.0
NOD_DROP = 14.0        # larger => fewer nods
NOD_WINDOW = 1.5       # seconds

# -------- MediaPipe setup --------
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [263,387,385,362,380,373]
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308
NOSE_TIP, FOREHEAD = 1, 10


def _euclid(p1, p2): return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _ear(pts, idx):
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    v = _euclid(p2, p6) + _euclid(p3, p5)
    h = 2.0 * _euclid(p1, p4)
    return (v / h) if h > 1e-6 else 0.0


def _mar(pts):
    top, bot = pts[MOUTH_TOP], pts[MOUTH_BOTTOM]
    left, right = pts[MOUTH_LEFT], pts[MOUTH_RIGHT]
    d = _euclid(left, right)
    return (_euclid(top, bot) / d) if d > 1e-6 else 0.0


def _pitch_proxy(pts):
    # simple head-pitch proxy (nose vs forehead)
    return (pts[NOSE_TIP][1] - pts[FOREHEAD][1]) * 180.0


class BlinkMeter:
    def __init__(self, th, minf):
        self.th, self.minf = th, minf
        self.run, self.t0 = 0, None
        self.blinks = []
    def update(self, t, ear):
        if ear < self.th:
            if self.run == 0:
                self.t0 = t
            self.run += 1
        else:
            if self.run >= self.minf and self.t0 is not None:
                self.blinks.append((self.t0, t - self.t0))
            self.run, self.t0 = 0, None
    def stats(self, now, win):
        self.blinks = [(ts, d) for (ts, d) in self.blinks if (now - ts) <= win]
        durs = [d for _, d in self.blinks]
        rate = (len(self.blinks) / max(win, 1e-6)) * 60.0
        return rate, (float(np.mean(durs)) if durs else 0.0)


class YawnMeter:
    def __init__(self, th, minsec):
        self.th, self.minsec = th, minsec
        self.up, self.t0, self.yawns = False, 0.0, []
        self.mouth_open, self.last = 0.0, None
    def update(self, t, mar):
        if self.last is not None and mar > self.th:
            self.mouth_open += (t - self.last)
        self.last = t
        if mar > self.th and not self.up:
            self.up, self.t0 = True, t
        if self.up and mar <= self.th:
            dur = t - self.t0
            if dur >= self.minsec:
                self.yawns.append((self.t0, dur))
            self.up = False
    def stats(self, now, win):
        self.yawns = [(ts, d) for (ts, d) in self.yawns if (now - ts) <= win]
        mouth_frac = min(self.mouth_open, win) / max(win, 1e-6)
        return len(self.yawns), mouth_frac


class NodMeter:
    def __init__(self, drop, interval, win):
        self.drop, self.interval, self.win = drop, interval, win
        self.samples, self.nods = deque(), []
    def update(self, t, pitch):
        self.samples.append((t, pitch))
        while self.samples and (t - self.samples[0][0]) > self.win:
            self.samples.popleft()
        if len(self.samples) >= 3:
            t1, p1 = self.samples[-3]
            t2, p2 = self.samples[-1]
            if (p2 - p1) > self.drop and (t2 - t1) <= self.interval:
                self.nods.append(t2)
    def stats(self, now, win):
        self.nods = [ts for ts in self.nods if (now - ts) <= win]
        ps = [p for (ts, p) in self.samples if (now - ts) <= win]
        return len(self.nods), (float(np.std(ps)) if ps else 0.0)


def _draw_overlay(frame, score, level):
    h, w = frame.shape[:2]
    # header box
    cv2.rectangle(frame, (10, 10), (w - 10, 110), (0, 0, 0), -1)
    color = (0, 200, 0) if score < 60 else ((0, 200, 255) if score < 75 else (0, 0, 255))
    cv2.putText(frame, f"Fatigue: {score:5.1f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Level: {level.upper()}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # progress bar
    bar_x, bar_y, bar_w, bar_h = 260, 30, w - 280, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 2)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * (score / 100.0)), bar_y + bar_h),
                  color, -1)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    blink = BlinkMeter(EAR_CLOSED, EAR_MIN_FRAMES)
    yawn = YawnMeter(MAR_YAWN, YAWN_MIN_SEC)
    nod = NodMeter(NOD_DROP, NOD_WINDOW, WINDOW_SEC)

    last_compute = time.time()
    current_score = 0.0
    current_level = "normal"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0]
            ih, iw = frame.shape[0], frame.shape[1]
            pts = [(lm.x * iw, lm.y * ih) for lm in face.landmark]

            e = (_ear(pts, LEFT_EYE) + _ear(pts, RIGHT_EYE)) / 2.0
            m = _mar(pts)
            p = _pitch_proxy(pts)

            blink.update(t, e)
            yawn.update(t, m)
            nod.update(t, p)

            # simple keypoints for visualization
            for i in LEFT_EYE + RIGHT_EYE + [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]:
                cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 1, (255, 255, 0), -1)

        # compute score every HOP_SEC over WINDOW_SEC history
        if (t - last_compute) >= HOP_SEC:
            last_compute = t

            br, bd = blink.stats(t, WINDOW_SEC)
            yc, mouth_frac = yawn.stats(t, WINDOW_SEC)
            nc, pitch_std = nod.stats(t, WINDOW_SEC)

            facial_feats = {
                "blink_rate": float(br),
                "avg_blink_duration": float(bd),
                "yawns_count": float(yc),
                "mouth_open_duration": float(mouth_frac),
                "head_nod_count": float(nc),
                "head_pitch_std": float(pitch_std),
            }

            # No HRV in this webcam demo
            hrv_feats = {"heart_rate": 0.0, "mean_rr": 0.0, "std_rr": 0.0, "rmssd": -1.0}

            score, _subs = compute_subscores_and_score(
                facial_feats=facial_feats,
                hrv_feats=hrv_feats,
                facial_weight=1.0  # facial-only
            )
            current_score = score
            current_level = alert_from_score(score)["level"]

        _draw_overlay(frame, current_score, current_level)
        cv2.imshow("Driver Fatigue (facial only)", frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
