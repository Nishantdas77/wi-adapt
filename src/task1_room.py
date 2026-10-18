"""
src/task1_room.py
TASK 1: Recorded-environment identification (Room A or Room B?)

  input   : one 150 x 256 amplitude window (nothing else: no file name, no length)
  train   : ALL development windows, both rooms, answer = room
  test    : ALL validation windows (1 per validation capture)
  model   : StandardScaler + logistic regression
  features: mean and std over time of each subcarrier -> 512 numbers

  Two feature versions (both reported):
    with_scale : log(1 + amplitude)            -> keeps the overall signal level
    scale_free : per-window normalized window  -> only the SHAPE of the signal
  The scale_free model is saved as the ROUTER for Task 3 (configs: task1.router_variant).

Results the assignment asks for:
  results/tables/task1_room_accuracy.csv          accuracy per window length + per room
  results/experiments.csv                         experiment, target_room, K, metric, value
  results/figures/task1/task1_accuracy_vs_window_length.png   "environment fingerprint" figure
  results/figures/task1/task1_confusion_L150.png
  results/predictions/task1_validation_predictions.csv  window_id, room, label, confidence
  models/task1_router.joblib

Run from the repo root (after make_windows.py):
    python src/task1_room.py
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (load_config, load_window, log_results, normalize_window, path_of,
                    read_windows_index, write_csv_rows)

VARIANTS = ("with_scale", "scale_free")
DPI = 200


def center_crop(w, length):
    s = (w.shape[0] - length) // 2
    return w[s:s + length]


def features(w, variant, eps_rel):
    a = np.log1p(w) if variant == "with_scale" else normalize_window(w, eps_rel)
    return np.concatenate([a.mean(axis=0), a.std(axis=0)]).astype(np.float32)


def make_model(C, seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                            random_state=seed))


def main():
    cfg = load_config()
    tcfg, eps = cfg["task1"], cfg["normalization"]["eps_rel"]
    lengths = sorted(tcfg["window_lengths"])
    full = cfg["window"]["length"]
    index = read_windows_index(cfg)
    dev = [r for r in index if r["split"] == "development"]
    val = [r for r in index if r["split"] == "validation"]
    print(f"development windows: {len(dev)}   validation windows: {len(val)}")

    # ---- features for every window, every variant, every length (load each window once) ----
    def build(rows):
        F = {(v, L): [] for v in VARIANTS for L in lengths}
        for r in rows:
            w = load_window(r)
            for L in lengths:
                c = center_crop(w, L)
                for v in VARIANTS:
                    F[(v, L)].append(features(c, v, eps))
        return {k: np.stack(x) for k, x in F.items()}

    print("computing features ...")
    F_dev, F_val = build(dev), build(val)
    y_dev = np.array([r["room"] for r in dev])
    y_val = np.array([r["room"] for r in val])

    # ---- train / evaluate ----
    table, exp_rows, models = [], [], {}
    for v in VARIANTS:
        for L in lengths:
            m = make_model(tcfg["C"], cfg["seed"]).fit(F_dev[(v, L)], y_dev)
            pred = m.predict(F_val[(v, L)])
            acc = accuracy_score(y_val, pred)
            acc_a = accuracy_score(y_val[y_val == "A"], pred[y_val == "A"])
            acc_b = accuracy_score(y_val[y_val == "B"], pred[y_val == "B"])
            table.append({"variant": v, "window_length": L, "accuracy": round(acc, 4),
                          "accuracy_roomA": round(acc_a, 4), "accuracy_roomB": round(acc_b, 4),
                          "n_validation": len(y_val)})
            exp_rows.append({"experiment": f"task1_room_id_{v}_L{L}", "target_room": "both",
                             "K": "", "metric": "accuracy", "value": round(acc, 4)})
            if L == full:
                models[v] = (m, pred)
            print(f"  {v:<11} L={L:>3}  accuracy {acc:.4f}  (A {acc_a:.4f}, B {acc_b:.4f})")

    res = path_of(cfg, "results")
    write_csv_rows(res / "tables" / "task1_room_accuracy.csv", table)
    for v in VARIANTS:  # headline numbers (official 150-row windows)
        acc = next(t["accuracy"] for t in table if t["variant"] == v and t["window_length"] == full)
        exp_rows.append({"experiment": f"task1_room_id_{v}", "target_room": "both",
                         "K": "", "metric": "accuracy", "value": acc})
    log_results(cfg, exp_rows)

    # ---- figure: accuracy vs window length ----
    figdir = res / "figures" / "task1"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, mk in zip(VARIANTS, ("o-", "s--")):
        acc = [t["accuracy"] for t in table if t["variant"] == v]
        ax.plot(lengths, acc, mk, label=v)
    ax.set_xlabel("window length (packets, centre crop)")
    ax.set_ylabel("room accuracy (validation)")
    ax.set_title("Task 1: environment fingerprint vs window length")
    ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "task1_accuracy_vs_window_length.png", dpi=DPI)
    plt.close(fig)

    # ---- figure: confusion matrices at L = 150 ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, v in zip(axes, VARIANTS):
        cm = confusion_matrix(y_val, models[v][1], labels=["A", "B"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["A", "B"]); ax.set_yticks([0, 1], ["A", "B"])
        ax.set_xlabel("predicted room"); ax.set_ylabel("true room")
        ax.set_title(f"{v} (L={full})")
    fig.tight_layout()
    fig.savefig(figdir / "task1_confusion_L150.png", dpi=DPI)
    plt.close(fig)

    # ---- predictions file + router ----
    rv = tcfg["router_variant"]
    m, pred = models[rv]
    conf = m.predict_proba(F_val[(rv, full)]).max(axis=1)
    write_csv_rows(res / "predictions" / "task1_validation_predictions.csv",
                   [{"window_id": r["window_id"], "room": p, "label": "",
                     "confidence": round(float(c), 4), "true_room": r["room"]}
                    for r, p, c in zip(val, pred, conf)])
    mdir = path_of(cfg, "models")
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "variant": rv, "length": full, "eps_rel": eps},
                mdir / "task1_router.joblib")
    print(f"\nSaved tables, figures, predictions and models/task1_router.joblib ({rv})")


if __name__ == "__main__":
    main()
