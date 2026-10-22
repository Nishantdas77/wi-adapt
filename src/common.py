"""
src/common.py
Small shared helpers used by every step:
  - load the config
  - preprocessing version tag (for provenance)
  - normalization of one window
  - reading the windows index / one window
  - logging results to results/experiments.csv
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ORDER_A = ["SIT", "STAND", "WALK", "SITDOWN", "STANDUP", "FALL"]
ORDER_B = ["EMPTY", "LYING", "SIT", "STAND", "WALK"]
SUBCARRIER_COLUMNS = [f"subcarrier_{i:03d}" for i in range(1, 257)]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "experiment.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg, key):
    return ROOT / cfg["paths"][key]


def preprocess_version(cfg):
    """Short tag that changes whenever a preprocessing setting changes."""
    part = {k: cfg[k] for k in ("room_a", "qc", "window")}
    return hashlib.sha1(json.dumps(part, sort_keys=True).encode()).hexdigest()[:8]


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- normalization
def log_amp(w, eps_rel=1e-3):
    """log(amp + eps_rel * mean). Scaling the window by c only adds log(c)."""
    return np.log(w + eps_rel * float(w.mean()) + 1e-12)


def normalize_window(w, eps_rel=1e-3):
    """Per-window normalization: scale-free log, then one mean/std for the window.
    Needs no statistics from any other data -> no leakage."""
    a = log_amp(w, eps_rel)
    return ((a - a.mean()) / (a.std() + 1e-6)).astype(np.float32)


# ---------------------------------------------------------------- windows
def read_windows_index(cfg):
    return read_csv_rows(path_of(cfg, "windows_index"))


def load_window(row):
    return np.load(ROOT / row["file"])


def load_normalized_windows(rows, eps_rel=1e-3, dtype=np.float16):
    """All windows of `rows`, each normalized on its own, as one (N, 150, 256) array.
    float16 halves the memory (~380 MB for 5000 windows); models convert to float32."""
    if not rows:
        return np.zeros((0, 150, 256), dtype=dtype)
    first = load_window(rows[0])
    X = np.empty((len(rows),) + first.shape, dtype=dtype)
    for i, r in enumerate(rows):
        X[i] = normalize_window(load_window(r), eps_rel)
    return X


# ---------------------------------------------------------------- results
RESULT_FIELDS = ["experiment", "target_room", "K", "metric", "value"]


def log_results(cfg, new_rows):
    """Add rows to results/experiments.csv. A row with the same
    (experiment, target_room, K, metric) replaces the old one."""
    path = path_of(cfg, "results") / "experiments.csv"
    old = read_csv_rows(path) if path.exists() else []
    key = lambda r: (r["experiment"], str(r["target_room"]), str(r["K"]), r["metric"])
    new_keys = {key(r) for r in new_rows}
    rows = [r for r in old if key(r) not in new_keys]
    rows += [{k: r[k] for k in RESULT_FIELDS} for r in new_rows]
    write_csv_rows(path, rows, RESULT_FIELDS)