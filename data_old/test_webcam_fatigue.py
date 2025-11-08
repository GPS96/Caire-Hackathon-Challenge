import os
import numpy as np
import pytest
pytest.importorskip('cv2')
import cv2
import tempfile
from webcam_fatigue import simple_facial_heuristics, WebcamOpts, run_webcam


def make_gray_frames(n=10, h=64, w=64):
    frames = []
    for i in range(n):
        val = 128 + int(10 * np.sin(2 * np.pi * i / max(1, n)))
        img = np.full((h, w), val, dtype=np.uint8)
        frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return frames


def test_simple_facial_heuristics_returns_dict():
    frames = make_gray_frames(20)
    d = simple_facial_heuristics(frames)
    assert isinstance(d, dict)
    assert 'blink_rate' in d


def test_smoothing_effect():
    # create a noisy signal with a single big drop
    x = np.ones(50) * 100.0
    x[25] = 20.0
    # import smoothing helper via running through simple_facial_heuristics logic
    frames = []
    for val in x:
        img = np.full((16, 16), int(val), dtype=np.uint8)
        frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    d = simple_facial_heuristics(frames)
    # blink_rate should be finite and present
    assert 'blink_rate' in d
    assert d['blink_rate'] >= 0


def test_simulated_run_writes_csv(tmp_path):
    out = tmp_path / "sim_test.csv"
    opts = WebcamOpts()
    opts.simulate_on_fail = True
    opts.simulate_fs = 30
    opts.simulate_windows = 1
    opts.window_seconds = 2
    opts.simulate_hr = 60
    setattr(opts, 'out_csv', str(out))
    run_webcam(opts)
    assert out.exists()
    text = out.read_text()
    assert 'window_idx' in text
