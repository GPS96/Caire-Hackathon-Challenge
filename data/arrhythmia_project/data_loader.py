"""Utilities for discovering PPG waveform files and parsing labels.

The pipeline expects paired `.mat` and `.hea` records formatted similarly to PhysioNet
alarms datasets. Channel 2 in the waveform matrix corresponds to the PPG/PLETH signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.io import loadmat

LOGGER = logging.getLogger(__name__)

LABEL_ID_MAP = {
    "Normal": 0,
    "Asystole": 1,
    "Ventricular_Flutter_Fib": 2,
    "Tachycardia": 3,
    "Bradycardia": 4,
}

IGNORE_TOKENS = {"true alarm", "artifact"}

SYNONYM_MAP = {
    "normal": "Normal",
    "normal rhythm": "Normal",
    "normal sinus rhythm": "Normal",
    "sinus rhythm": "Normal",
    "sinus": "Normal",
    "nsr": "Normal",
    "baseline": "Normal",
    "vfib": "Ventricular_Flutter_Fib",
    "vf": "Ventricular_Flutter_Fib",
    "vflutter": "Ventricular_Flutter_Fib",
    "brady": "Bradycardia",
    "tach": "Tachycardia",
    "tachy": "Tachycardia",
    "asys": "Asystole",
}


@dataclass
class PPGRecord:
    """Container housing waveform and multi-class label information."""

    name: str
    signal: np.ndarray
    sampling_rate: int
    label_name: str
    label_id: int


def find_record_basenames(data_dir: Path) -> List[str]:
    """Return base filenames that contain both `.mat` and `.hea` companions."""

    mat_files = {path.stem for path in data_dir.glob("*.mat")}
    hea_files = {path.stem for path in data_dir.glob("*.hea")}
    basenames = sorted(mat_files & hea_files)
    if not basenames:
        LOGGER.warning("No paired .mat/.hea files found in %s", data_dir)
    return basenames


def _parse_label_from_header(lines: Iterable[str]) -> Optional[str]:
    """Extract a normalized arrhythmia label from header comment lines."""

    detected_label: Optional[str] = None
    saw_false_alarm = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or not stripped.startswith("#"):
            continue

        line = stripped.lower()
        if "false alarm" in line:
            saw_false_alarm = True

        if any(token in line for token in IGNORE_TOKENS):
            continue

        for canonical in LABEL_ID_MAP:
            if canonical.lower() in line:
                detected_label = canonical
                break
        else:  # only executed when canonical loop does not break
            for synonym, canonical in SYNONYM_MAP.items():
                if synonym in line:
                    detected_label = canonical
                    break
            else:
                if "alarm" in line and "normal" in line:
                    detected_label = "Normal"

        if detected_label and saw_false_alarm:
            return "Normal"

    if saw_false_alarm:
        return "Normal"

    return detected_label


def _load_ppg_from_mat(mat_path: Path) -> tuple[np.ndarray, int]:
    """Load channel 2 (index 1) from the PhysioNet-style MATLAB file."""

    payload = loadmat(mat_path)
    if "val" not in payload:
        raise KeyError(f"Expected 'val' key in MATLAB file {mat_path}")
    data = np.array(payload["val"], dtype=np.float32)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("PPG signal is expected to be the second channel in a 2D array")
    signal = data[1]  # Channel indexing starts at zero; channel 2 => index 1
    sampling_rate = int(payload.get("fs", [[125]])[0][0]) if "fs" in payload else 125
    return signal, sampling_rate


def load_records(data_dir: str | Path) -> Dict[str, PPGRecord]:
    """Read all paired records in *data_dir* and return structured metadata."""

    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(directory)

    records: Dict[str, PPGRecord] = {}
    for basename in find_record_basenames(directory):
        mat_path = directory / f"{basename}.mat"
        hea_path = directory / f"{basename}.hea"
        try:
            signal, fs = _load_ppg_from_mat(mat_path)
        except Exception as exc:  # pragma: no cover - defensive reporting
            LOGGER.error("Failed to load %s: %s", mat_path.name, exc)
            continue

        try:
            header_lines = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except FileNotFoundError:
            LOGGER.error("Missing header file for %s", basename)
            continue

        label_name = _parse_label_from_header(header_lines)
        if label_name is None:
            LOGGER.info("No arrhythmia label found in %s; skipping", hea_path.name)
            continue
        label_id = LABEL_ID_MAP[label_name]

        records[basename] = PPGRecord(
            name=basename,
            signal=signal,
            sampling_rate=fs,
            label_name=label_name,
            label_id=label_id,
        )
        LOGGER.debug("Loaded %s with label %s", basename, label_name)

    return records


__all__ = ["PPGRecord", "load_records", "LABEL_ID_MAP", "LABEL_NAME_MAP"]
