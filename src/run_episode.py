"""
src/run_episode.py
Live demo / reproducibility entry point.
Give it a raw Room A (.pcap) or Room B (.csv) file, or a folder of them,
and it writes room + activity predictions in the required format.

    python src/run_episode.py episodes/test1                 # all .pcap/.csv in the folder
    python src/run_episode.py episodes/test1/some_file.pcap  # one file
    options:
      --out PATH        where to write the predictions (default: <folder>/predictions.csv)
      --mode fewshot    (default) Task 3 system: shared activities from the OTHER room,
                        exclusive activities from K support captures (development data)
      --mode full       comparison: activity model trained on ALL development labels of the room
      --all-windows     one prediction per 150-row window (default: one window from the middle)
      --rebuild         rebuild the ready-made activity model

Pipeline (same code as development / validation):
  read (.pcap: decode CSI -> amplitude | .csv: skip header) -> QC -> 150 x 256 window
  -> Task 1 router: ROOM from the signal (file name and extension are NOT used as features)
  -> per-window normalization -> frozen SimCLR encoder -> 128 numbers
  -> activity model of the predicted room:  room A -> B->A model,  room B -> A->B model
  -> predictions.csv: window_id, room, label, confidence (+ extra info columns)

Files shorter than 150 valid rows are still predicted from all their rows (note = short_window).
If the folder has labels.csv (columns: file,label) the accuracy is printed as a self-check.
"""
import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np

from build_intermediate import quality_control
from common import ROOT, load_config, normalize_window, path_of, read_csv_rows, read_windows_index, \
    write_csv_rows
from csi_decode import read_room_a, read_room_b, to_frequency_order
from fewshot import EXCLUSIVE, OTHER, ROOM_CLASSES, SHARED, classify, fit_stats, normalize, prototypes

L = 150


# ------------------------------------------------------------------ ready-made activity model
def room_a_order(cfg):
    """The Room A subcarrier order chosen by build_intermediate.py (same order as training)."""
    mode = cfg["room_a"]["subcarrier_order"]
    if mode in ("fftshift", "stored"):
        return mode
    summary = path_of(cfg, "reports") / "qc_summary.txt"
    if summary.exists():
        m = re.search(r"Room A subcarrier order:\s*(fftshift|stored)", summary.read_text("utf-8"))
        if m:
            return m.group(1)
    print("WARNING: could not read the Room A order from reports/qc_summary.txt, using fftshift")
    return "fftshift"


def build_model(cfg, mode, path):
    """Built ONLY from development windows + their frozen SimCLR embeddings."""
    emb_path = path_of(cfg, "models") / "task2_simclr_embeddings.npz"
    if not emb_path.exists():
        raise SystemExit("models/task2_simclr_embeddings.npz missing: run task2_evaluate.py first.")
    d = np.load(emb_path, allow_pickle=True)
    pos = {w: i for i, w in enumerate(d["dev_ids"])}
    dev = [r for r in read_windows_index(cfg) if r["split"] == "development"]
    F = d["dev"][[pos[r["window_id"]] for r in dev]]
    room = np.array([r["room"] for r in dev])
    y = np.array([r["label"] for r in dev])
    cap = np.array([r["capture_id"] for r in dev])
    t3 = cfg["task3"]
    model = {"mode": mode, "rooms": {}}

    if mode == "fewshot":
        from task3_fewshot import make_support
        K, seed = int(cfg.get("episode", {}).get("K", 5)), int(cfg.get("episode", {}).get("seed", 0))
        support = make_support(cfg, dev)                     # same picks as Task 3
        model.update({"K": K, "seed": seed})
        for t in ("A", "B"):                                  # t = target room of the file
            s = OTHER[t]
            st_t = fit_stats(F[room == t], t3["normalization"])
            st_s = fit_stats(F[room == s], t3["normalization"])
            caps = [c for lab in EXCLUSIVE[t] for c in support[(t, seed, lab)][:K]]
            sup = (room == t) & np.isin(cap, caps)
            P = np.vstack([prototypes(normalize(F[room == s], st_s), y[room == s], SHARED),
                           prototypes(normalize(F[sup], st_t), y[sup], EXCLUSIVE[t])])
            model["rooms"][t] = {"stats": st_t, "P": P, "classes": SHARED + EXCLUSIVE[t],
                                 "support_captures": caps}
    else:
        from task2_evaluate import fit_probe
        for t in ("A", "B"):
            m = room == t
            model["rooms"][t] = {"probe": fit_probe("linear", F[m], y[m], cfg)}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"built ready-made activity model ({mode}) -> {path.relative_to(ROOT)}")
    return model


# ------------------------------------------------------------------ one file
def read_file(path, cfg, order):
    """Same readers + same QC as build_intermediate.py. Returns clean amplitude rows."""
    if path.suffix.lower() == ".pcap":
        amp = read_room_a(path, scale=cfg["room_a"]["scale"])["amp"]
        if order == "fftshift":
            amp = to_frequency_order(amp)
    else:
        amp = read_room_b(path)
    keep, _, _ = quality_control(amp, cfg["qc"])
    return amp[keep].astype(np.float32)


def windows_of(amp, all_windows):
    n = len(amp)
    if n == 0:
        return []
    if n < L:
        return [(0, amp, "short_window")]
    starts = list(range(0, n - L + 1, L)) if all_windows else [(n - L) // 2]
    return [(s, amp[s:s + L], "") for s in starts]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Predict room + activity for raw .pcap / .csv files")
    ap.add_argument("input", help="a .pcap/.csv file or a folder with such files")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["fewshot", "full"], default="fewshot")
    ap.add_argument("--all-windows", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    inp = Path(args.input).resolve()
    if inp.is_dir():
        files = sorted(p for p in inp.iterdir() if p.suffix.lower() in (".pcap", ".csv")
                       and p.name.lower() not in ("predictions.csv", "labels.csv"))
        out = Path(args.out) if args.out else inp / "predictions.csv"
    elif inp.is_file():
        files, out = [inp], Path(args.out) if args.out else inp.parent / "predictions.csv"
    else:
        raise SystemExit(f"not found: {inp}")
    if not files:
        raise SystemExit(f"no .pcap or .csv files in {inp}")

    # ---- load models ----
    mdir = path_of(cfg, "models")
    for need in ("task1_router.joblib", "task2_simclr_encoder.pt"):
        if not (mdir / need).exists():
            raise SystemExit(f"models/{need} missing: run Task 1 and Task 2 first.")
    from models import embed_windows, load_encoder
    from task1_room import features as router_features
    router = joblib.load(mdir / "task1_router.joblib")
    encoder, _ = load_encoder(mdir / "task2_simclr_encoder.pt")
    model_path = mdir / f"episode_model_{args.mode}.joblib"
    model = build_model(cfg, args.mode, model_path) if (args.rebuild or not model_path.exists()) \
        else joblib.load(model_path)
    order = room_a_order(cfg)
    eps = cfg["normalization"]["eps_rel"]
    tau = cfg["task3"]["temperature"]

    # ---- predict ----
    rows = []
    print(f"\n{len(files)} file(s) | mode {args.mode} | Room A subcarrier order {order}\n")
    for path in files:
        try:
            amp = read_file(path, cfg, order)
        except Exception as e:                                   # broken / unreadable file
            print(f"  {path.name}: could not read ({e})")
            rows.append({"window_id": f"{path.stem}_w0", "room": "", "label": "", "confidence": "",
                         "source_file": path.name, "start_row": "", "n_rows": 0,
                         "room_confidence": "", "note": f"read_error: {e}"})
            continue
        wins = windows_of(amp, args.all_windows)
        if not wins:
            rows.append({"window_id": f"{path.stem}_w0", "room": "", "label": "", "confidence": "",
                         "source_file": path.name, "start_row": "", "n_rows": 0,
                         "room_confidence": "", "note": "no_valid_rows"})
            continue
        for i, (start, w, note) in enumerate(wins):
            f = router_features(w, router["variant"], router["eps_rel"])[None]
            room = str(router["model"].predict(f)[0])
            room_conf = float(router["model"].predict_proba(f).max())
            h = embed_windows(encoder, normalize_window(w, eps)[None])
            rm = model["rooms"][room]
            if model["mode"] == "fewshot":
                lab, conf = classify(normalize(h, rm["stats"]), rm["P"], rm["classes"], tau)
                lab, conf = str(lab[0]), float(conf[0])
            else:
                proba = rm["probe"].predict_proba(h)[0]
                lab, conf = str(rm["probe"].classes_[proba.argmax()]), float(proba.max())
            rows.append({"window_id": f"{path.stem}_w{i}", "room": room, "label": lab,
                         "confidence": round(conf, 4), "source_file": path.name,
                         "start_row": start, "n_rows": len(w),
                         "room_confidence": round(room_conf, 4), "note": note})

    write_csv_rows(out, rows)

    # ---- summary ----
    print(f"{'window_id':<34}{'room':<6}{'activity':<10}{'conf':>6}  note")
    for r in rows:
        print(f"{r['window_id'][:33]:<34}{r['room']:<6}{r['label']:<10}"
              f"{str(r['confidence']):>6}  {r['note']}")

    labels_file = (inp if inp.is_dir() else inp.parent) / "labels.csv"
    if labels_file.exists():
        truth = {r["file"].strip(): r["label"].strip().upper() for r in read_csv_rows(labels_file)}
        scored = [(truth[r["source_file"]], r["label"]) for r in rows if r["source_file"] in truth]
        if scored:
            acc = np.mean([t == p for t, p in scored])
            print(f"\nself-check with labels.csv: {sum(t == p for t, p in scored)}/{len(scored)}"
                  f" windows correct ({acc:.1%})")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.exit(main())