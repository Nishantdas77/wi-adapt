"""
src/task3_fewshot.py
TASK 3: cross-room few-label adaptation.

For each target room (the other room is the source):
  source labels : ALL development windows of the source room, all its activities
  support       : K distinct DEVELOPMENT captures per TARGET-EXCLUSIVE activity
                    target A -> SITDOWN, STANDUP, FALL      target B -> EMPTY, LYING
                  (target labels of SIT, STAND, WALK are never read)
  test          : ALL validation windows of the target room
  K             : 1, 5, 10, 20  x  n_seeds different support picks (nested per seed)

Method (fewshot.py): frozen features -> per-room normalization -> prototypes -> closest prototype.
  routing  predicted : the Task 1 router chooses the room of each validation window
           oracle    : the true room is used
           none      : one global normalization, prototypes of all 8 activities
  self-training (extra, "_st"): prototypes refined with confident pseudo-labels on the
           UNLABELLED target development windows (never validation)
  features: simclr (frozen Task 2 encoder), random (same network, untrained), handcrafted

Run from the repo root (after task1_room.py and task2_evaluate.py):
    python src/task3_fewshot.py

Saves:
    splits/support/task3_support.csv               every support capture (provenance)
    results/tables/task3_all_runs.csv              every run (all seeds)
    results/tables/task3_summary.csv               mean +- std over seeds
    results/tables/task3_per_class_target{A,B}_K5.csv
    results/experiments.csv                        experiment, target_room, K, metric, value
    results/figures/task3/label_efficiency.png     (required)
    results/figures/task3/routing_comparison.png   (required)
    results/figures/task3/confusion_target{A,B}_K5.png  (required)
    results/figures/task3/ssl_contribution.png
    results/predictions/task3_<main>_K5_seed0_validation_predictions.csv
"""
import random
from collections import defaultdict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from common import (load_config, load_normalized_windows, load_window, log_results, path_of,
                    read_windows_index, write_csv_rows)
from fewshot import (EXCLUSIVE, OTHER, ROOM_CLASSES, SHARED, classify, fit_stats, normalize,
                     prototypes, self_train)

ROUTINGS = ["predicted", "oracle", "none"]
DPI = 200


# ------------------------------------------------------------------ features
def get_features(name, cfg, dev, val):
    """(F_dev, F_val) for one feature set, rows in the same order as dev / val."""
    models_dir = path_of(cfg, "models")
    if name == "handcrafted":
        from task2_evaluate import handcrafted_features
        eps = cfg["normalization"]["eps_rel"]
        return (handcrafted_features(load_normalized_windows(dev, eps)),
                handcrafted_features(load_normalized_windows(val, eps)))
    if name == "random":
        from models import embed_windows, load_encoder, random_encoder
        _, ck = load_encoder(models_dir / "task2_simclr_encoder.pt")
        enc = random_encoder(ck["arch"], cfg["seed"])
        eps = cfg["normalization"]["eps_rel"]
        return (embed_windows(enc, load_normalized_windows(dev, eps)),
                embed_windows(enc, load_normalized_windows(val, eps)))
    path = models_dir / f"task2_{name}_embeddings.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found: run task2_evaluate.py for '{name}' first.")
    d = np.load(path, allow_pickle=True)
    pos = {w: i for i, w in enumerate(d["dev_ids"])}
    pos_v = {w: i for i, w in enumerate(d["val_ids"])}
    return (d["dev"][[pos[r["window_id"]] for r in dev]],
            d["val"][[pos_v[r["window_id"]] for r in val]])


def route_validation(cfg, val):
    """Task 1 router: predicted room of every validation window."""
    from task1_room import center_crop, features
    r = joblib.load(path_of(cfg, "models") / "task1_router.joblib")
    X = np.stack([features(center_crop(load_window(v), r["length"]), r["variant"], r["eps_rel"])
                  for v in val])
    return r["model"].predict(X)


# ------------------------------------------------------------------ support sets
def make_support(cfg, dev):
    """Per target room, seed and exclusive activity: a fixed random ORDER of the
    development captures. The first K captures are the K-shot support (nested)."""
    caps = defaultdict(set)
    for r in dev:
        caps[(r["room"], r["label"])].add(r["capture_id"])
    t3 = cfg["task3"]
    kmax = max(t3["K_values"])
    rows, order = [], {}
    for target in ("A", "B"):
        for seed in range(t3["n_seeds"]):
            for lab in EXCLUSIVE[target]:
                pool = sorted(caps[(target, lab)])
                random.Random(f"{cfg['seed']}|task3|{target}|{seed}|{lab}").shuffle(pool)
                if len(pool) < kmax:
                    print(f"  note: only {len(pool)} dev captures for {target}/{lab} (K max {kmax})")
                order[(target, seed, lab)] = pool[:kmax]
                rows += [{"target_room": target, "seed": seed, "label": lab, "rank": i + 1,
                          "capture_id": c} for i, c in enumerate(pool[:kmax])]
    write_csv_rows(path_of(cfg, "splits").parent / "support" / "task3_support.csv", rows)
    return order


# ------------------------------------------------------------------ metrics
def metrics(y_true, y_pred, target):
    classes, excl = ROOM_CLASSES[target], EXCLUSIVE[target]
    f = lambda labels: f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in classes if (y_true == c).any()]
    return {"primary_macro_f1": f(classes), "exclusive_macro_f1": f(excl),
            "shared_macro_f1": f(SHARED), "balanced_accuracy": float(np.mean(recalls))}


# ------------------------------------------------------------------ one run
def run_once(F_dev, F_val, dev, val, room_hat, target, support_caps, routing, st, cfg):
    t3 = cfg["task3"]
    src, tau = OTHER[target], t3["temperature"]
    room_dev = np.array([r["room"] for r in dev])
    y_dev = np.array([r["label"] for r in dev])
    cap_dev = np.array([r["capture_id"] for r in dev])
    vmask = np.array([r["room"] == target for r in val])
    Fv = F_val[vmask]

    src_m = room_dev == src
    sup_m = (room_dev == target) & np.isin(cap_dev, list(support_caps))
    excl = EXCLUSIVE[target]

    if routing == "none":
        stats = fit_stats(F_dev, t3["normalization"])                    # both rooms together
        P_src = prototypes(normalize(F_dev[src_m], stats), y_dev[src_m], ROOM_CLASSES[src])
        P_exc = prototypes(normalize(F_dev[sup_m], stats), y_dev[sup_m], excl)
        classes = ROOM_CLASSES[src] + excl                               # all 8 activities
        pred, conf = classify(normalize(Fv, stats), np.vstack([P_src, P_exc]), classes, tau)
        return pred, conf, np.array([""] * len(Fv))

    st_src = fit_stats(F_dev[src_m], t3["normalization"])
    st_tgt = fit_stats(F_dev[room_dev == target], t3["normalization"])
    Fn_src = normalize(F_dev[src_m], st_src)
    P_src_all = prototypes(Fn_src, y_dev[src_m], ROOM_CLASSES[src])      # model of the source room
    P_shared = prototypes(Fn_src, y_dev[src_m], SHARED)
    P_exc = prototypes(normalize(F_dev[sup_m], st_tgt), y_dev[sup_m], excl)
    tgt_classes = SHARED + excl
    P_tgt = np.vstack([P_shared, P_exc])
    if st:
        U = normalize(F_dev[room_dev == target], st_tgt)                 # unlabelled target dev
        P_tgt = self_train(P_tgt, U, t3["self_training"]["rounds"],
                           t3["self_training"]["top_fraction"])

    rooms = room_hat[vmask] if routing == "predicted" else np.array([target] * len(Fv))
    pred = np.empty(len(Fv), dtype=object)
    conf = np.zeros(len(Fv))
    to_t = rooms == target
    if to_t.any():
        pred[to_t], conf[to_t] = classify(normalize(Fv[to_t], st_tgt), P_tgt, tgt_classes, tau)
    if (~to_t).any():                     # router says "source room": use the source room's model
        pred[~to_t], conf[~to_t] = classify(normalize(Fv[~to_t], st_src), P_src_all,
                                            ROOM_CLASSES[src], tau)
    return pred, conf, rooms


# ------------------------------------------------------------------ figures
def band(ax, summ, key, target, metric, label, style, color):
    """Mean line + std band of one metric vs K. key = (features, routing, self_training)."""
    xs, m, s = [], [], []
    for (f, rt, st, tg, K), v in sorted(summ.items(), key=lambda kv: kv[0][4]):
        if (f, rt, st, tg) == key + (target,) and metric in v:
            xs.append(K); m.append(v[metric][0]); s.append(v[metric][1])
    if xs:
        m, s = np.array(m), np.array(s)
        ax.plot(xs, m, style, color=color, label=label)
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15)


def plot_confusion(y_true, y_pred, target, K, path):
    rows = ROOM_CLASSES[target]
    cols = rows + sorted(set(y_pred) - set(rows))
    cm = confusion_matrix(y_true, y_pred, labels=cols)[:len(rows)]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2, 0.9 * len(rows) + 1.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Task 3: target Room {target}, K={K} (all seeds pooled)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    cfg = load_config()
    t3 = cfg["task3"]
    index = read_windows_index(cfg)
    dev = [r for r in index if r["split"] == "development"]
    val = [r for r in index if r["split"] == "validation"]
    y_val = np.array([r["label"] for r in val])
    room_val = np.array([r["room"] for r in val])

    print("routing validation windows with the Task 1 router ...")
    room_hat = route_validation(cfg, val)
    for t in ("A", "B"):
        m = room_val == t
        print(f"  router accuracy on Room {t} validation: {np.mean(room_hat[m] == t):.4f}")

    support = make_support(cfg, dev)
    main_f = t3["main_features"]
    k_show = 5 if 5 in t3["K_values"] else t3["K_values"][len(t3["K_values"]) // 2]
    modes = [(rt, False) for rt in ROUTINGS] + [("predicted", True), ("oracle", True)]

    all_runs, pooled, pred_rows = [], defaultdict(lambda: ([], [])), []
    for fname in t3["feature_sets"]:
        print(f"features: {fname}")
        F_dev, F_val = get_features(fname, cfg, dev, val)
        for target in ("A", "B"):
            vmask = room_val == target
            for K in t3["K_values"]:
                for seed in range(t3["n_seeds"]):
                    caps = [c for lab in EXCLUSIVE[target]
                            for c in support[(target, seed, lab)][:K]]
                    for routing, st in modes:
                        pred, conf, rooms = run_once(F_dev, F_val, dev, val, room_hat, target,
                                                     caps, routing, st, cfg)
                        m = metrics(y_val[vmask], pred, target)
                        mode = routing + ("_st" if st else "")
                        all_runs.append({"features": fname, "routing": mode, "target_room": target,
                                         "K": K, "seed": seed,
                                         **{k: round(v, 4) for k, v in m.items()}})
                        if fname == main_f and mode == "predicted" and K == k_show:
                            pooled[target][0].extend(y_val[vmask])
                            pooled[target][1].extend(pred)
                            if seed == 0:
                                pred_rows += [{"window_id": v["window_id"], "room": rm,
                                               "label": p, "confidence": round(float(c), 4),
                                               "target_room": target, "true_label": v["label"]}
                                              for v, p, c, rm in zip(
                                                  [v for v, mm in zip(val, vmask) if mm],
                                                  pred, conf, rooms)]
            print(f"  target {target}: done")

    res = path_of(cfg, "results")
    tables, figdir = res / "tables", res / "figures" / "task3"
    figdir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(tables / "task3_all_runs.csv", all_runs)

    # ---- summary: mean +- std over seeds ----
    metric_names = ["primary_macro_f1", "exclusive_macro_f1", "shared_macro_f1", "balanced_accuracy"]
    groups = defaultdict(list)
    for r in all_runs:
        groups[(r["features"], r["routing"], r["target_room"], r["K"])].append(r)
    summary, exp_rows, summ = [], [], {}
    for (f, rt, tg, K), rs in sorted(groups.items()):
        row = {"features": f, "routing": rt, "target_room": tg,
               "direction": f"{OTHER[tg]}->{tg}", "K": K, "n_seeds": len(rs)}
        st_flag = rt.endswith("_st")
        key = (f, rt.replace("_st", ""), st_flag, tg, K)
        summ[key] = {}
        for mname in metric_names:
            vals = np.array([r[mname] for r in rs])
            row[f"{mname}_mean"] = round(vals.mean(), 4)
            row[f"{mname}_std"] = round(vals.std(), 4)
            summ[key][mname] = (vals.mean(), vals.std())
            exp_rows.append({"experiment": f"task3_{f}_{rt}", "target_room": tg, "K": K,
                             "metric": mname, "value": round(vals.mean(), 4)})
        summary.append(row)
    write_csv_rows(tables / "task3_summary.csv", summary)
    log_results(cfg, exp_rows)

    # ---- console table: main configuration ----
    print(f"\nMAIN: {main_f} + predicted routing (mean over {t3['n_seeds']} seeds)")
    print(f"{'direction':<10}{'K':>4}{'primary F1':>12}{'exclusive F1':>14}{'shared F1':>11}")
    for r in summary:
        if r["features"] == main_f and r["routing"] == "predicted":
            print(f"{r['direction']:<10}{r['K']:>4}{r['primary_macro_f1_mean']:>12.4f}"
                  f"{r['exclusive_macro_f1_mean']:>14.4f}{r['shared_macro_f1_mean']:>11.4f}")

    Ks = t3["K_values"]
    # ---- figure: label efficiency ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tg in zip(axes, ("A", "B")):
        band(ax, summ, (main_f, "predicted", False), tg, "exclusive_macro_f1",
             "target-labelled activities (K-shot)", "o-", "#C44E52")
        band(ax, summ, (main_f, "predicted", False), tg, "shared_macro_f1",
             "shared activities (from other room)", "s-", "#4C72B0")
        band(ax, summ, (main_f, "predicted", False), tg, "primary_macro_f1",
             "all activities (primary)", "^-", "#55A868")
        band(ax, summ, (main_f, "predicted", True), tg, "primary_macro_f1",
             "all activities + self-training", "^--", "#8172B2")
        ax.set_title(f"{OTHER[tg]} -> {tg} (target Room {tg})")
        ax.set_xlabel("K (labelled captures per target-exclusive activity)")
        ax.set_xticks(Ks); ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 (validation)"); axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Task 3: label efficiency ({main_f}, predicted routing, mean +- std)")
    fig.tight_layout(); fig.savefig(figdir / "label_efficiency.png", dpi=DPI); plt.close(fig)

    # ---- figure: routing comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    w = 0.25
    for ax, tg in zip(axes, ("A", "B")):
        for i, (rt, c) in enumerate(zip(ROUTINGS, ("#4C72B0", "#55A868", "#C44E52"))):
            ms = [summ[(main_f, rt, False, tg, K)]["primary_macro_f1"] for K in Ks]
            ax.bar(np.arange(len(Ks)) + (i - 1) * w, [m for m, _ in ms], w,
                   yerr=[s for _, s in ms], capsize=2, color=c,
                   label={"predicted": "predicted-room routing", "oracle": "oracle room",
                          "none": "no routing"}[rt])
        ax.set_xticks(range(len(Ks)), [f"K={k}" for k in Ks])
        ax.set_title(f"target Room {tg}"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("primary macro-F1 (validation)"); axes[1].legend(fontsize=8)
    fig.suptitle(f"Task 3: routing contribution ({main_f})")
    fig.tight_layout(); fig.savefig(figdir / "routing_comparison.png", dpi=DPI); plt.close(fig)

    # ---- figure: SSL contribution ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {"simclr": "#4C72B0", "random": "#8C8C8C", "handcrafted": "#DD8452"}
    for ax, tg in zip(axes, ("A", "B")):
        for f in t3["feature_sets"]:
            band(ax, summ, (f, "predicted", False), tg, "primary_macro_f1", f, "o-",
                 colors.get(f, None))
        ax.set_title(f"target Room {tg}"); ax.set_xticks(Ks); ax.grid(alpha=0.3)
        ax.set_xlabel("K")
    axes[0].set_ylabel("primary macro-F1 (validation)"); axes[1].legend(fontsize=8)
    fig.suptitle("Task 3: contribution of self-supervised pretraining (predicted routing)")
    fig.tight_layout(); fig.savefig(figdir / "ssl_contribution.png", dpi=DPI); plt.close(fig)

    # ---- confusion matrices + per-class at K = 5 (required) ----
    for tg in ("A", "B"):
        yt, yp = np.array(pooled[tg][0]), np.array(pooled[tg][1], dtype=object)
        if len(yt) == 0:
            continue
        plot_confusion(yt, yp, tg, k_show, figdir / f"confusion_target{tg}_K{k_show}.png")
        labs = ROOM_CLASSES[tg]
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labs, zero_division=0)
        write_csv_rows(tables / f"task3_per_class_target{tg}_K{k_show}.csv",
                       [{"activity": l, "precision": round(a, 4), "recall": round(b, 4),
                         "f1": round(c, 4), "support_pooled_over_seeds": int(d)}
                        for l, a, b, c, d in zip(labs, p, r, f, s)])

    write_csv_rows(res / "predictions" / f"task3_{main_f}_K{k_show}_seed0_validation_predictions.csv",
                   pred_rows)
    print(f"\nSaved tables in {tables}, figures in {figdir}, predictions in {res / 'predictions'}")


if __name__ == "__main__":
    main()