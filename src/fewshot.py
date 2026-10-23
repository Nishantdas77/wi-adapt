"""
src/fewshot.py
The Task 3 method (helper file, do not run). Also reused by run_episode.py.

  1. room normalization : standardize features with the statistics of ONE room
                          (fitted on that room's development windows, no labels),
                          then L2-normalize -> both rooms share the same centre
  2. prototypes         : one average vector per activity
  3. classify           : cosine similarity to every prototype, closest wins,
                          softmax(similarity / temperature) = confidence
  4. self-training      : (extra) refine prototypes with the most confident
                          pseudo-labelled UNLABELLED target development windows
"""
import numpy as np

SHARED = ["SIT", "STAND", "WALK"]
ROOM_CLASSES = {"A": ["SIT", "STAND", "WALK", "SITDOWN", "STANDUP", "FALL"],
                "B": ["EMPTY", "LYING", "SIT", "STAND", "WALK"]}
EXCLUSIVE = {"A": ["SITDOWN", "STANDUP", "FALL"], "B": ["EMPTY", "LYING"]}
OTHER = {"A": "B", "B": "A"}


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def fit_stats(F, mode="zscore"):
    """Statistics of one group of windows (no labels needed)."""
    mu = F.mean(axis=0)
    sd = F.std(axis=0) + 1e-6 if mode == "zscore" else np.ones(F.shape[1])
    if mode == "none":
        mu = np.zeros(F.shape[1])
    return mu, sd


def normalize(F, stats):
    mu, sd = stats
    return l2norm((F - mu) / sd)


def prototypes(Fn, y, classes):
    """(n_classes, dim) average of the normalized windows of each class, L2-normalized."""
    y = np.asarray(y)
    P = []
    for c in classes:
        m = y == c
        if not m.any():
            raise ValueError(f"no examples for class {c}")
        P.append(Fn[m].mean(axis=0))
    return l2norm(np.stack(P))


def classify(Fn, P, classes, temperature=0.1):
    """Closest prototype by cosine similarity. Returns (labels, confidence)."""
    if len(Fn) == 0:
        return np.array([], dtype=object), np.array([])
    sims = Fn @ P.T
    z = sims / temperature
    z -= z.max(axis=1, keepdims=True)
    prob = np.exp(z)
    prob /= prob.sum(axis=1, keepdims=True)
    best = sims.argmax(axis=1)
    return np.asarray(classes, dtype=object)[best], prob[np.arange(len(best)), best]


def self_train(P, U, rounds=3, top_fraction=0.3):
    """Refine prototypes P with unlabelled windows U (already normalized).
    Each round: pseudo-label U, keep the most confident `top_fraction` per class,
    new prototype = original prototype + mean of the kept windows (equal weight)."""
    P0 = P.copy()
    for _ in range(rounds):
        sims = U @ P.T
        lab, conf = sims.argmax(axis=1), sims.max(axis=1)
        new = P0.copy()
        for c in range(len(P)):
            idx = np.flatnonzero(lab == c)
            if len(idx) == 0:
                continue
            keep = idx[np.argsort(-conf[idx])[:max(1, int(len(idx) * top_fraction))]]
            new[c] = P0[c] + U[keep].mean(axis=0)
        P = l2norm(new)
    return P