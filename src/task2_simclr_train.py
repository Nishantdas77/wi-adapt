"""
src/task2_simclr_train.py
TASK 2, part 1: train the CSI encoder with SimCLR (self-supervised, NO labels).

  data     : all eligible DEVELOPMENT captures of both rooms (activity/room columns never read)
  sampling : every epoch takes `samples_per_capture` random 150-row segments from EVERY
             capture, so every capture counts equally (Room A is not drowned out by Room B)
  pairs    : the two views come from two segments of the SAME capture that start at most
             `neighbor_max_gap` rows apart (temporal consistency: same activity, a moment later)
  views    : time warp (random speed) + 120-row crop, time mask, subcarrier mask,
             small scaling, small noise. No time reversal (SITDOWN reversed = STANDUP).
  loss     : NT-Xent: each view must find its partner among all other views in the batch
  output   : the encoder (the projection head is dropped after training)

Segments are normalized exactly like the official windows (per-window scale-free log z-score).
Settings: configs/experiment.yaml -> task2

Run from the repo root (after make_windows.py):
    python src/task2_simclr_train.py --quick     # 2 epochs on 300 captures: checks it runs
    python src/task2_simclr_train.py             # full training

Saves:
    data/cache/dev_captures_<version>.npz   (first run only: all dev captures in one file)
    models/task2_simclr_encoder.pt          (quick mode: models/task2_simclr_encoder_quick.pt)
    results/tables/task2_simclr_loss.csv
    results/figures/task2/simclr_training.png
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import (ROOT, load_config, load_normalized_windows, path_of, preprocess_version,
                    read_csv_rows, read_windows_index, write_csv_rows)
from models import (CSIEncoder, ProjectionHead, count_params, encoder_arch, get_device,
                    save_encoder)

WINDOW = 150


# ------------------------------------------------------------------ data
def load_dev_captures(cfg):
    """All eligible development captures as one big row array + offsets (cached)."""
    cache = ROOT / "data" / "cache" / f"dev_captures_{preprocess_version(cfg)}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["rows"], d["offsets"], d["lengths"]
    qc = [q for q in read_csv_rows(path_of(cfg, "reports") / "qc_log.csv")
          if q["split"] == "development" and q["eligible"] == "1"]
    qc.sort(key=lambda q: q["capture_id"])
    print(f"building capture cache from {len(qc)} development captures (first run only) ...")
    parts, lengths = [], []
    for i, q in enumerate(qc, 1):
        a = np.loadtxt(ROOT / q["intermediate_csv"], delimiter=",", skiprows=1,
                       dtype=np.float32, ndmin=2)
        parts.append(a.astype(np.float16))
        lengths.append(len(a))
        if i % 300 == 0:
            print(f"    {i}/{len(qc)}")
    rows = np.concatenate(parts)
    lengths = np.array(lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, rows=rows, offsets=offsets, lengths=lengths)
    return rows, offsets, lengths


def sample_pairs(R, offsets, lengths, caps, gap):
    """Two 150-row raw segments per capture, starting at most `gap` rows apart."""
    n, off = lengths[caps], offsets[caps]
    B = len(caps)
    s1 = (torch.rand(B) * (n - WINDOW + 1).float()).long()
    if gap > 0:
        s2 = (s1 + torch.randint(-gap, gap + 1, (B,))).clamp(min=0)
        s2 = torch.minimum(s2, n - WINDOW)
    else:
        s2 = s1
    t = torch.arange(WINDOW)
    return R[(off + s1)[:, None] + t], R[(off + s2)[:, None] + t]     # (B, 150, 256) each


def normalize_batch(x, eps_rel):
    """Same as common.normalize_window, for a batch: scale-free log, then one z-score per window."""
    m = x.mean(dim=(1, 2), keepdim=True)
    a = torch.log(x + eps_rel * m + 1e-12)
    mu = a.mean(dim=(1, 2), keepdim=True)
    sd = a.std(dim=(1, 2), keepdim=True, correction=0)
    return (a - mu) / (sd + 1e-6)


# ------------------------------------------------------------------ augmentation
def augment(x, a, warp):
    """x: (B, T, S) normalized segments -> one view (B, crop_len, S)."""
    B, T, S = x.shape
    dev = x.device
    L = min(a["crop_len"], T)

    # time warp + crop in one step: read L rows at a random speed f
    f = torch.empty(B, 1, device=dev).uniform_(warp[0], warp[1])
    f = torch.clamp(f, max=(T - 1) / max(L - 1, 1))
    o = torch.rand(B, 1, device=dev) * (T - 1 - (L - 1) * f)
    p = o + torch.arange(L, device=dev)[None, :] * f                    # (B, L) float positions
    i0 = p.floor().long().clamp(0, T - 1)
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (p - i0.float())[:, :, None]
    x = (torch.gather(x, 1, i0[:, :, None].expand(B, L, S)) * (1 - w)
         + torch.gather(x, 1, i1[:, :, None].expand(B, L, S)) * w)

    if a["time_mask_max"] > 0:                                          # blank a block of rows
        wd = torch.randint(0, a["time_mask_max"] + 1, (B, 1), device=dev)
        s = torch.randint(0, L, (B, 1), device=dev)
        t = torch.arange(L, device=dev)[None, :]
        x = x.masked_fill(((t >= s) & (t < s + wd))[:, :, None], 0.0)
    if a["sub_mask_max"] > 0:                                           # blank a subcarrier band
        wd = torch.randint(0, a["sub_mask_max"] + 1, (B, 1), device=dev)
        s = torch.randint(0, S, (B, 1), device=dev)
        fr = torch.arange(S, device=dev)[None, :]
        x = x.masked_fill(((fr >= s) & (fr < s + wd))[:, None, :], 0.0)
    lo, hi = a["scale_range"]
    x = x * torch.empty(B, 1, 1, device=dev).uniform_(lo, hi)
    return x + a["noise_std"] * torch.randn_like(x)


# ------------------------------------------------------------------ loss + check
def nt_xent(z1, z2, tau):
    """SimCLR loss. Row i of z1 and row i of z2 are partners; all other rows are negatives."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = (z @ z.t()) / tau
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.no_grad()
def embedding_spread(encoder, x, device):
    """Collapse check: mean std of L2-normalised embeddings over fixed windows (~0 = collapse)."""
    encoder.eval()
    h = F.normalize(encoder(x.to(device).float()), dim=1)
    return float(h.std(dim=0).mean())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 epochs on 300 captures")
    args = ap.parse_args()

    cfg = load_config()
    t2 = cfg["task2"]
    seed, eps = cfg["seed"], cfg["normalization"]["eps_rel"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if t2.get("num_threads"):
        torch.set_num_threads(int(t2["num_threads"]))
    device = get_device()

    rows, offsets, lengths = load_dev_captures(cfg)
    if args.quick:
        keep = np.random.default_rng(seed).choice(len(lengths), min(300, len(lengths)), replace=False)
        offsets, lengths = offsets[keep], lengths[keep]
    R = torch.from_numpy(rows)                      # float16, all rows of all dev captures
    offsets, lengths = torch.from_numpy(offsets), torch.from_numpy(lengths)
    n_caps = len(lengths)

    epochs = 2 if args.quick else int(t2["epochs"])
    bs = min(int(t2["batch_size"]), n_caps)
    reps = int(t2["samples_per_capture"])
    steps = (n_caps * reps) // bs

    arch = encoder_arch(t2)
    enc = CSIEncoder(**arch).to(device)
    head = ProjectionHead(t2["embed_dim"], t2["proj_dim"]).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=t2["lr"], weight_decay=t2["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"device {device} | {n_caps} dev captures | {epochs} epochs x {steps} steps | batch {bs}"
          f" | encoder {count_params(enc):,} params")

    # fixed official dev windows for the collapse check
    win_rows = [r for r in read_windows_index(cfg) if r["split"] == "development"]
    pick = np.random.default_rng(seed).choice(len(win_rows), min(512, len(win_rows)), replace=False)
    check_x = torch.from_numpy(load_normalized_windows([win_rows[i] for i in pick], eps))
    print(f"embedding spread before training: {embedding_spread(enc, check_x, device):.4f}")

    name = "task2_simclr_encoder_quick.pt" if args.quick else "task2_simclr_encoder.pt"
    out_path = path_of(cfg, "models") / name
    history = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        enc.train(); head.train()
        # several independent shuffles of all captures -> every capture appears equally often
        order = torch.cat([torch.randperm(n_caps) for _ in range(reps)])
        total = 0.0
        for k in range(steps):
            caps = order[k * bs:(k + 1) * bs]
            a1, a2 = sample_pairs(R, offsets, lengths, caps, int(t2["neighbor_max_gap"]))
            x1 = normalize_batch(a1.to(device).float(), eps)
            x2 = normalize_batch(a2.to(device).float(), eps)
            z1 = head(enc(augment(x1, t2["augment"], t2["time_warp"])))
            z2 = head(enc(augment(x2, t2["augment"], t2["time_warp"])))
            loss = nt_xent(z1, z2, t2["temperature"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        spread = embedding_spread(enc, check_x, device)
        history.append({"epoch": ep, "loss": round(total / max(steps, 1), 5),
                        "embedding_spread": round(spread, 5),
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {ep:>3}/{epochs}  loss {history[-1]['loss']:.4f}  spread {spread:.4f}"
              f"  ({history[-1]['seconds']}s)")
        if spread < 1e-3:
            print("WARNING: embedding spread is ~0 -> the encoder may have collapsed")
        if ep % 10 == 0 or ep == epochs:
            save_encoder(out_path, enc, arch, method="simclr", epoch=ep,
                         preprocess_version=preprocess_version(cfg), n_train_captures=n_caps,
                         task2_config=t2, seed=seed)

    res = path_of(cfg, "results")
    suffix = "_quick" if args.quick else ""
    write_csv_rows(res / "tables" / f"task2_simclr_loss{suffix}.csv", history)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ex = [h["epoch"] for h in history]
    ax.plot(ex, [h["loss"] for h in history], color="#4C72B0")
    ax.set_xlabel("epoch"); ax.set_ylabel("NT-Xent loss", color="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(ex, [h["embedding_spread"] for h in history], color="#DD8452", ls="--")
    ax2.set_ylabel("embedding spread (collapse if ~0)", color="#DD8452")
    ax.set_title("Task 2: SimCLR training on development captures (no labels)")
    fig.tight_layout()
    figdir = res / "figures" / "task2"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"simclr_training{suffix}.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved encoder to {out_path} and the training curve to {figdir}")


if __name__ == "__main__":
    main()