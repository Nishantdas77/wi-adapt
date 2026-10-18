"""
src/plot_amplitude.py
Amplitude plots for EVERY activity in EVERY room, made from the official
150 x 256 windows (development windows only).

For each (room, activity) one figure with 3 panels:
  1. heatmap of one window: time (packet) x subcarrier, colour = amplitude
  2. average amplitude of every subcarrier over all dev windows of that activity (+- 1 std)
  3. amplitude over time of 3 subcarriers in that same window
Plus per room:
  roomX_overview_heatmaps.png   one heatmap per activity, same colour scale
  roomX_overview_spectrum.png   average spectrum of all activities on one plot

Saves to results/figures/amplitude/

Run from the repo root (after make_windows.py):
    python src/plot_amplitude.py
"""
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ORDER_A, ORDER_B, load_config, load_window, path_of, read_windows_index

MAX_WINDOWS_FOR_MEAN = 200
SUBCARRIERS = [40, 100, 200]
DPI = 200
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def main():
    cfg = load_config()
    out = path_of(cfg, "results") / "figures" / "amplitude"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_windows_index(cfg) if r["split"] == "development"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["room"], r["label"])].append(r)

    for room, order in (("A", ORDER_A), ("B", ORDER_B)):
        labels = [l for l in order if groups[(room, l)]]
        examples, means, stds = {}, {}, {}
        for lab in labels:
            g = sorted(groups[(room, lab)], key=lambda r: r["window_id"])
            examples[lab] = load_window(g[0])
            pick = random.Random(cfg["seed"]).sample(g, min(MAX_WINDOWS_FOR_MEAN, len(g)))
            per_sub = np.stack([load_window(r).mean(axis=0) for r in pick])
            means[lab], stds[lab] = per_sub.mean(axis=0), per_sub.std(axis=0)

        allvals = np.concatenate([examples[l].ravel() for l in labels])
        vmin, vmax = np.percentile(allvals, [1, 99])

        # ---- one figure per activity ----
        for lab in labels:
            w = examples[lab]
            fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
            im = ax[0].imshow(w.T, aspect="auto", origin="lower", cmap="viridis",
                              vmin=vmin, vmax=vmax)
            ax[0].set_title("one window: amplitude")
            ax[0].set_xlabel("packet (time)"); ax[0].set_ylabel("subcarrier")
            fig.colorbar(im, ax=ax[0], label="amplitude")
            x = np.arange(256)
            ax[1].plot(x, means[lab], color="#4C72B0")
            ax[1].fill_between(x, means[lab] - stds[lab], means[lab] + stds[lab],
                               color="#4C72B0", alpha=0.25)
            ax[1].set_title(f"average over {min(MAX_WINDOWS_FOR_MEAN, len(groups[(room, lab)]))}"
                            " dev windows (+-1 std)")
            ax[1].set_xlabel("subcarrier"); ax[1].set_ylabel("amplitude")
            for s in SUBCARRIERS:
                ax[2].plot(w[:, s], lw=1, label=f"subcarrier {s}")
            ax[2].set_title("same window: amplitude over time")
            ax[2].set_xlabel("packet (time)"); ax[2].set_ylabel("amplitude")
            ax[2].legend(fontsize=8)
            fig.suptitle(f"Room {room} - {lab}", fontsize=13, y=1.02)
            fig.tight_layout()
            fig.savefig(out / f"room{room}_{lab}.png", dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        # ---- overview: heatmaps ----
        fig, axes = plt.subplots(1, len(labels), figsize=(2.8 * len(labels), 3.8), sharey=True)
        for ax, lab in zip(np.atleast_1d(axes), labels):
            im = ax.imshow(examples[lab].T, aspect="auto", origin="lower", cmap="viridis",
                           vmin=vmin, vmax=vmax)
            ax.set_title(lab); ax.set_xlabel("packet")
        np.atleast_1d(axes)[0].set_ylabel("subcarrier")
        fig.colorbar(im, ax=axes, label="amplitude", shrink=0.9)
        fig.suptitle(f"Room {room}: one 150 x 256 window per activity", y=1.02)
        fig.savefig(out / f"room{room}_overview_heatmaps.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        # ---- overview: average spectrum per activity ----
        fig, ax = plt.subplots(figsize=(10, 4))
        for lab in labels:
            ax.plot(means[lab], lw=1.2, label=lab)
        ax.set_title(f"Room {room}: average amplitude per subcarrier, per activity (dev)")
        ax.set_xlabel("subcarrier"); ax.set_ylabel("amplitude"); ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"room{room}_overview_spectrum.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Room {room}: saved {len(labels)} activity figures + 2 overviews")

    print(f"All figures in {out}")


if __name__ == "__main__":
    main()
