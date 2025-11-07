"""
Simple static plotter for saved CSV data.
Reads live_ppg.csv and plots rPPG and PPG signals.

Usage:
    python plot_saved_csv.py [csv_filename] [--window SECONDS]

Default: Shows last 30 seconds of data from live_ppg.csv
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Parse arguments
csv_file = Path("live_ppg.csv")
window_sec = 30.0

i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == '--window' and i + 1 < len(sys.argv):
        window_sec = float(sys.argv[i + 1])
        i += 2
    elif not arg.startswith('--'):
        csv_file = Path(arg)
        i += 1
    else:
        i += 1

print(f"[*] Reading CSV file: {csv_file.absolute()}")
print(f"[*] Window size: {window_sec} seconds")

if not csv_file.exists():
    print(f"[!] Error: {csv_file} not found!")
    sys.exit(1)

# Read CSV file manually (no pandas needed)
times = []
rppg_vals = []
ppg_vals = []

with open(csv_file, 'r') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if i == 0 or not line or line.startswith('#'):
            continue  # Skip header
        
        try:
            parts = line.split(',')
            if len(parts) == 3:
                t = float(parts[0])
                r = float(parts[1])
                p = float(parts[2])
                times.append(t)
                rppg_vals.append(r)
                ppg_vals.append(p)
        except (ValueError, IndexError):
            continue

if len(times) == 0:
    print("[!] No data found in CSV file")
    sys.exit(1)

print(f"[+] Loaded {len(times)} samples")

# Convert to numpy arrays
times = np.array(times)
rppg_vals = np.array(rppg_vals)
ppg_vals = np.array(ppg_vals)

# Filter to show only last window_sec seconds
max_time = times[-1]
cutoff_time = max_time - window_sec
mask = times >= cutoff_time

times_window = times[mask]
rppg_window = rppg_vals[mask]
ppg_window = ppg_vals[mask]

# Make time relative to start of window
times_rel = times_window - times_window[0]

print(f"[+] Plotting {len(times_window)} samples from last {window_sec}s")
print(f"    Time range: {times_window[0]:.2f}s to {times_window[-1]:.2f}s")
print(f"    rPPG range: {rppg_window.min():.3f} to {rppg_window.max():.3f}")
print(f"    PPG range: {ppg_window.min():.3f} to {ppg_window.max():.3f}")

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
fig.suptitle(f'rPPG and PPG Signals - Last {window_sec}s', fontsize=14, fontweight='bold')

# Plot rPPG
ax1.plot(times_rel, rppg_window, 'r-', linewidth=1.5, label='rPPG (raw)')
ax1.set_ylabel('rPPG Amplitude', fontsize=11)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, times_rel[-1])

# Plot PPG
ax2.plot(times_rel, ppg_window, 'b-', linewidth=1.5, label='PPG (cleaned)')
ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.set_ylabel('PPG Amplitude', fontsize=11)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, times_rel[-1])

plt.tight_layout()
print("[*] Displaying plot. Close window to exit.")
plt.show()
