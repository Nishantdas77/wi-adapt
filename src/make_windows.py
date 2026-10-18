"""
src/make_windows.py
STEP 3: cut the official model input: windows of 150 consecutive rows x 256 subcarriers.

Rules (configs/experiment.yaml -> window):
  validation capture  : exactly ONE window, from the middle of the capture
  development capture : up to 3 windows, evenly spaced, starting >= 50 rows apart
  capture < 150 rows  : not eligible, no window (already marked by build_intermediate.py)
  never mix rows from two captures; all windows of a capture stay in its split

Reads only data/intermediate_csv (the common format), so from here on the
code does not know whether a capture came from pcap or csv.

Outputs:
  data/windows/development/window_XXXXXX.npy   (150, 256) float32 amplitude
  data/windows/validation/window_XXXXXX.npy
  splits/windows_index.csv   window_id, capture_id, room, label, split, start_row, ...
                             (commit this file: it is the provenance trail)

Run from the repo root:
    python src/make_windows.py
"""
import shutil
from collections import Counter

import numpy as np

from common import ROOT, load_config, path_of, preprocess_version, read_csv_rows, write_csv_rows


def window_starts(n_rows, length, min_shift, max_windows, split):
    if n_rows < length:
        return []
    free = n_rows - length
    if split == "validation":
        return [free // 2]
    k = min(max_windows, 1 + free // min_shift)
    if k == 1:
        return [free // 2]
    return [int(round(x)) for x in np.linspace(0, free, k)]


def main():
    cfg = load_config()
    wcfg = cfg["window"]
    L = wcfg["length"]
    version = preprocess_version(cfg)
    qc = read_csv_rows(path_of(cfg, "reports") / "qc_log.csv")
    if any(q["preprocess_version"] != version for q in qc):
        raise SystemExit("Config changed since build_intermediate.py ran. Run it again first.")

    out = path_of(cfg, "windows")
    if out.exists():
        shutil.rmtree(out)                     # always rebuild from scratch
    index, n = [], 0
    for q in sorted(qc, key=lambda q: q["capture_id"]):
        if q["eligible"] != "1":
            continue
        amp = np.loadtxt(ROOT / q["intermediate_csv"], delimiter=",", skiprows=1,
                         dtype=np.float32, ndmin=2)
        for s in window_starts(len(amp), L, wcfg["min_shift"], wcfg["max_dev_windows"], q["split"]):
            win = np.ascontiguousarray(amp[s:s + L], dtype=np.float32)
            assert win.shape == (L, 256)
            wid = f"window_{n:06d}"
            f = out / q["split"] / f"{wid}.npy"
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, win)
            index.append({
                "window_id": wid, "capture_id": q["capture_id"], "room": q["room"],
                "label": q["label"], "split": q["split"], "start_row": s,
                "capture_rows": len(amp), "length": L,
                "file": f.relative_to(ROOT).as_posix(), "preprocess_version": version,
            })
            n += 1

    write_csv_rows(path_of(cfg, "windows_index"), index)
    c = Counter((r["room"], r["split"], r["label"]) for r in index)
    unique_caps = {(r["capture_id"], r["room"], r["split"]) for r in index}
    caps = Counter((room, split) for _, room, split in unique_caps)
    print(f"Saved {n} windows to {out.relative_to(ROOT)}/ and splits/windows_index.csv\n")
    print(f"{'room':<5}{'split':<13}{'label':<9}{'windows':>8}")
    for k, v in sorted(c.items()):
        print(f"{k[0]:<5}{k[1]:<13}{k[2]:<9}{v:>8}")
    print("\ncaptures with windows:", dict(caps))


if __name__ == "__main__":
    main()
