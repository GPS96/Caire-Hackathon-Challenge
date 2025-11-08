from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import csv

from .cleaning import clean_bvp, CleanOptions, estimate_fs_from_time

try:
    from .chrom_extractor import extract_rppg_from_video
except Exception:
    extract_rppg_from_video = None


def read_csv_signal(path: Path) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    """Read a CSV with either one column (signal) or two columns (time, signal)."""
    arr = np.genfromtxt(str(path), delimiter=',', comments='#')
    if arr.ndim == 1:
        x = arr.astype(float)
        return x, None, None
    if arr.shape[1] >= 2:
        t = arr[:, 0].astype(float)
        x = arr[:, 1].astype(float)
        return x, None, t
    raise ValueError("Unsupported CSV format. Provide 1-col (signal) or 2-col (time, signal)")


def write_csv_signal(path: Path, y: np.ndarray, fs: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(y), dtype=float) / fs
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'ppg'])
        for tt, yy in zip(t, y):
            w.writerow([f"{tt:.6f}", f"{yy:.6f}"])


def main(argv=None):
    p = argparse.ArgumentParser(description="rPPG -> clean PPG utility")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input-csv', type=str, help='CSV file with rPPG signal (1-col) or time,signal (2-col)')
    src.add_argument('--video', type=str, help='Path to video file for CHROM extraction (fallback)')
    src.add_argument('--webcam', action='store_true', help='Use default webcam for CHROM extraction (fallback)')

    p.add_argument('--fs', type=float, default=None, help='Sampling rate if input CSV has no time column')
    p.add_argument('--low', type=float, default=0.7, help='Band-pass low cutoff Hz')
    p.add_argument('--high', type=float, default=4.0, help='Band-pass high cutoff Hz')
    p.add_argument('--order', type=int, default=3, help='Butterworth filter order')
    p.add_argument('--smooth', type=float, default=0.3, help='Moving average window (sec). 0 to disable.')
    p.add_argument('--output', type=str, default='clean_ppg.csv', help='Output CSV path')
    p.add_argument('--plot', action='store_true', help='Show a plot of raw vs clean')

    args = p.parse_args(argv)

    fs = args.fs
    raw = None
    time = None

    if args.input_csv:
        raw, _, time = read_csv_signal(Path(args.input_csv))
        if fs is None:
            if time is not None:
                fs = estimate_fs_from_time(time)
            else:
                raise SystemExit("--fs is required when the CSV has no time column")
    else:
        # webcam or video
        if extract_rppg_from_video is None:
            raise SystemExit("chrom_extractor is unavailable. Ensure OpenCV is installed.")
        source = 0 if args.webcam else args.video
        raw, fs = extract_rppg_from_video(source)

    options = CleanOptions(low_hz=args.low, high_hz=args.high, order=args.order, smooth_win_sec=args.smooth)
    clean, info = clean_bvp(raw, fs=fs, options=options, time=time)

    out_path = Path(args.output)
    write_csv_signal(out_path, clean, fs)
    print(f"Saved clean PPG to {out_path} (fs={info['fs']:.2f} Hz, band=[{info['low_hz']:.2f},{info['high_hz']:.2f}] Hz)")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            t_clean = np.arange(len(clean)) / info['fs']
            if time is not None:
                t_raw = time
            else:
                t_raw = np.arange(len(raw)) / info['fs']
            plt.figure(figsize=(10,5))
            plt.plot(t_raw, raw, label='Raw rPPG', alpha=0.5)
            plt.plot(t_clean, clean, label='Clean PPG', linewidth=2)
            plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.legend(); plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
