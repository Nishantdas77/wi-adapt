"""
src/task2_evaluate.py
TASK 2, part 2: evaluate a FROZEN encoder with a linear probe.

  1. embed every window (development + validation) with the frozen encoder (no augmentation)
  2. train logistic regression on DEVELOPMENT embeddings + activity labels
  3. score on VALIDATION: macro-F1 (headline), balanced accuracy, accuracy

  Feature sets compared with exactly the same probe:
     <name>       the trained encoder (e.g. simclr)
     random       same architecture, untrained weights (seed from config)
     handcrafted  per subcarrier: mean, std, range, mean |row-to-row change|  (1024 numbers)

  Probes:   linear  (official, logistic regression)   |   knn (extra, k-NN with cosine distance)
  Setups:   joint   (8 activities, both rooms)         |   A (6 activities)   |   B (5 activities)

Run from the repo root (after task2_simclr_train.py):
    python src/task2_evaluate.py
    python src/task2_evaluate.py --encoder models/task2_simclr_encoder_quick.pt --name simclr_quick

Saves:
    results/tables/task2_probe_results.csv            every feature set x probe x setup
    results/tables/task2_<name>_per_class.csv         precision / recall / F1 per activity
    results/experiments.csv                           experiment, target_room, K, metric, value
    results/figures/task2/<name>_umap_by_room.png     (required)
    results/figures/task2/<name>_umap_by_activity.png (required)
    results/figures/task2/<name>_confusion_joint.png
    results/predictions/task2_<name>_validation_predictions.csv
    models/task2_<name>_embeddings.npz                (reused by Task 3)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (load_config, load_normalized_windows, log_results, path_of, read_csv_rows,
                    read_windows_index, write_csv_rows)

ALL_LABELS = ["SIT", "STAND", "WALK", "SITDOWN", "STANDUP", "FALL", "EMPTY", "LYING"]
SETUPS = {"joint": None, "A": "A", "B": "B"}
DPI = 200


# ------------------------------------------------------------------ features
def handcrafted_features(X, chunk=512):
    """Per subcarrier over time: mean, std, max-min, mean |x[t+1]-x[t]| -> (N, 1024)."""
    out = []
    for i in range(0, len(X), chunk):
        x = np.asarray(X[i:i + chunk], dtype=np.float32)
        out.append(np.concatenate([
            x.mean(axis=1), x.std(axis=1), x.max(axis=1) - x.min(axis=1),
            np.abs(np.diff(x, axis=1)).mean(axis=1)], axis=1))
    return np.concatenate(out)


# ------------------------------------------------------------------ probes
def fit_probe(kind, F_tr, y_tr, cfg):
    t2 = cfg["task2"]
    if kind == "linear":
        clf = LogisticRegression(C=t2["probe_C"], max_iter=5000, class_weight="balanced",
                                 random_state=cfg["seed"])
    else:
        clf = KNeighborsClassifier(n_neighbors=t2["knn_k"], metric="cosine", weights="distance")
    return make_pipeline(StandardScaler(), clf).fit(F_tr, y_tr)


def scores(y_true, y_pred):
    labels = sorted(set(y_true))
    return {"macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)}


# ------------------------------------------------------------------ figures
def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 2, 0.9 * len(labels) + 1.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def project_2d(F, seed):
    F = StandardScaler().fit_transform(F)
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(F), "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        perp = max(5, min(30, (len(F) - 1) // 3))
        return TSNE(n_components=2, perplexity=perp, init="pca",
                    random_state=seed).fit_transform(F), "t-SNE"


def plot_2d(P, groups, order, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    cmap = plt.get_cmap("tab10")
    for k, g in enumerate([g for g in order if g in set(groups)]):
        m = np.asarray(groups) == g
        ax.scatter(P[m, 0], P[m, 1], s=14, alpha=0.8, color=cmap(k % 10), label=f"{g} ({m.sum()})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title); ax.legend(fontsize=8, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=str(path_of(cfg, "models") / "task2_simclr_encoder.pt"))
    ap.add_argument("--name", default="simclr")
    args = ap.parse_args()
    name, seed = args.name, cfg["seed"]

    from models import embed_windows, get_device, load_encoder, random_encoder

    index = read_windows_index(cfg)
    dev = [r for r in index if r["split"] == "development"]
    val = [r for r in index if r["split"] == "validation"]
    print(f"development windows {len(dev)} | validation windows {len(val)}")
    eps = cfg["normalization"]["eps_rel"]
    X_dev, X_val = load_normalized_windows(dev, eps), load_normalized_windows(val, eps)

    # ---- features ----
    device = get_device()
    enc, ck = load_encoder(Path(args.encoder), device)
    print(f"encoder: {args.encoder} (method {ck.get('method')}, epoch {ck.get('epoch')})")
    rand = random_encoder(ck["arch"], seed, device)
    feats = {
        name: (embed_windows(enc, X_dev), embed_windows(enc, X_val)),
        "random": (embed_windows(rand, X_dev), embed_windows(rand, X_val)),
        "handcrafted": (handcrafted_features(X_dev), handcrafted_features(X_val)),
    }

    y_dev = np.array([r["label"] for r in dev]); y_val = np.array([r["label"] for r in val])
    room_dev = np.array([r["room"] for r in dev]); room_val = np.array([r["room"] for r in val])

    # ---- probes ----
    table, exp_rows, keep = [], [], {}
    for fname, (F_dev, F_val) in feats.items():
        for probe in ("linear", "knn"):
            for setup, room in SETUPS.items():
                mtr = np.ones(len(dev), bool) if room is None else room_dev == room
                mte = np.ones(len(val), bool) if room is None else room_val == room
                if mtr.sum() == 0 or mte.sum() == 0:
                    continue
                model = fit_probe(probe, F_dev[mtr], y_dev[mtr], cfg)
                pred = model.predict(F_val[mte])
                s = scores(y_val[mte], pred)
                table.append({"features": fname, "probe": probe, "setup": setup,
                              **{k: round(v, 4) for k, v in s.items()},
                              "n_train": int(mtr.sum()), "n_validation": int(mte.sum())})
                for metric in ("macro_f1", "balanced_accuracy"):
                    exp_rows.append({"experiment": f"task2_{probe}_{fname}_{setup}",
                                     "target_room": "both" if room is None else room,
                                     "K": "", "metric": metric, "value": round(s[metric], 4)})
                if fname == name and probe == "linear" and setup == "joint":
                    keep = {"model": model, "pred": pred}
                print(f"  {fname:<12} {probe:<6} {setup:<5}  macro-F1 {s['macro_f1']:.4f}  "
                      f"bal-acc {s['balanced_accuracy']:.4f}")

    res = path_of(cfg, "results")
    # keep rows of encoders evaluated earlier (e.g. simclr) and replace only what we just ran
    table_path = res / "tables" / "task2_probe_results.csv"
    old = read_csv_rows(table_path) if table_path.exists() else []
    table = [r for r in old if r["features"] not in feats] + table
    write_csv_rows(table_path, table)
    log_results(cfg, exp_rows)

    # ---- per-class report + confusion (trained encoder, linear, joint) ----
    labels = [l for l in ALL_LABELS if l in set(y_val)]
    p, r, f, sup = precision_recall_fscore_support(y_val, keep["pred"], labels=labels,
                                                   zero_division=0)
    write_csv_rows(res / "tables" / f"task2_{name}_per_class.csv",
                   [{"activity": l, "precision": round(a, 4), "recall": round(b, 4),
                     "f1": round(c, 4), "support": int(d)} for l, a, b, c, d in zip(labels, p, r, f, sup)])
    figdir = res / "figures" / "task2"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_val, keep["pred"], labels,
                   f"Task 2: {name} + linear probe (validation)",
                   figdir / f"{name}_confusion_joint.png")

    # ---- predictions file ----
    proba = keep["model"].predict_proba(feats[name][1]).max(axis=1)
    write_csv_rows(res / "predictions" / f"task2_{name}_validation_predictions.csv",
                   [{"window_id": rw["window_id"], "room": "", "label": pr,
                     "confidence": round(float(c), 4), "true_label": rw["label"]}
                    for rw, pr, c in zip(val, keep["pred"], proba)])

    # ---- 2-D embedding plots of the validation set (qualitative) ----
    P, method = project_2d(feats[name][1], seed)
    plot_2d(P, room_val, ["A", "B"], f"{method} of {name} embeddings (validation) - by room",
            figdir / f"{name}_umap_by_room.png")
    plot_2d(P, y_val, ALL_LABELS, f"{method} of {name} embeddings (validation) - by activity",
            figdir / f"{name}_umap_by_activity.png")

    # ---- save embeddings for Task 3 ----
    np.savez(path_of(cfg, "models") / f"task2_{name}_embeddings.npz",
             dev_ids=np.array([r["window_id"] for r in dev]), dev=feats[name][0],
             val_ids=np.array([r["window_id"] for r in val]), val=feats[name][1])
    print(f"\nSaved tables, figures ({method}), predictions and embeddings for '{name}'.")


if __name__ == "__main__":
    main()