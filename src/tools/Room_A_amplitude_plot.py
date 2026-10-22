"""
src/tools/export_room_a_amplitude.py
Saves the decoded amplitude of EVERY Room A capture in a new folder, so you can
open any capture and see for yourself whether all 256 subcarriers carry signal.

Nothing is removed here (no QC): every decoded packet is kept, in time order,
subcarriers in frequency order (low -> high, the same order the pipeline uses).

Output folder:  data/room_A_amplitude_check/
    <split>/<LABEL>/<capture_id>.csv
        one row per packet:
        packet, time_s, seq, band, q1_share .. q4_share, subcarrier_001 .. subcarrier_256
          band     = full   (signal on all 4 quarters of 64 subcarriers = 80 MHz)
                     narrow (signal on 1 quarter only = 20 MHz)
                     other  (2 or 3 quarters)
          q1..q4   = share of the packet energy in subcarriers 1-64, 65-128, 129-192, 193-256
    <split>/<LABEL>/<capture_id>.png
        top   : heatmap of the whole capture (time x 256 subcarriers)
        bottom: band of every packet (full / narrow / other)
    summary.csv   one row per capture: packets, full, narrow, other

Run from the repo root:
    python src/tools/export_room_a_amplitude.py            # CSV + PNG for every capture
    python src/tools/export_room_a_amplitude.py --no-png   # CSV only (faster)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, SUBCARRIER_COLUMNS, load_config, path_of, read_csv_rows, write_csv_rows
from csi_decode import read_room_a, to_frequency_order

OUT = ROOT / "data" / "room_A_amplitude_check"
QUARTER_SHARE = 0.05
BAND_COLOR = {"full": "#2ca02c", "narrow": "#d62728", "other": "#ff7f0e"}


def band_of(amp):
    e = (amp.astype(np.float64) ** 2).reshape(len(amp), 4, 64).sum(axis=2)
    share = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)
    used = (share > QUARTER_SHARE).sum(axis=1)
    band = np.where(used == 4, "full", np.where(used == 1, "narrow", "other"))
    return band, share


def save_png(path, amp, band, title):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [5, 1]})
    vmax = np.percentile(amp, 99) if amp.size else 1
    im = ax.imshow(amp.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                   extent=[-0.5, len(amp) - 0.5, 0.5, 256.5], interpolation="nearest")
    for y in (64.5, 128.5, 192.5):
        ax.axhline(y, color="white", lw=0.6, ls="--", alpha=0.7)
    ax.set_ylabel("subcarrier (1-256)")
    ax.set_title(title)
    fig.colorbar(im, ax=[ax, bx], label="amplitude", shrink=0.8)
    x = np.arange(len(band))
    for b, c in BAND_COLOR.items():
        m = band == b
        bx.scatter(x[m], np.zeros(m.sum()), s=12, marker="|", color=c,
                   label=f"{b} ({m.sum()})")
    bx.set_yticks([]); bx.set_xlabel("packet (time order)")
    bx.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.6), frameon=False)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    make_png = "--no-png" not in sys.argv
    cfg = load_config()
    caps = sorted((r for r in read_csv_rows(path_of(cfg, "splits")) if r["room"] == "A"),
                  key=lambda r: r["capture_id"])
    header = ["packet", "time_s", "seq", "band", "q1_share", "q2_share", "q3_share",
              "q4_share"] + SUBCARRIER_COLUMNS
    summary = []
    print(f"exporting {len(caps)} Room A captures to {OUT.relative_to(ROOT)}/ ...")
    for i, r in enumerate(caps, 1):
        d = read_room_a(ROOT / r["path"], scale=cfg["room_a"]["scale"])
        amp = to_frequency_order(d["amp"])
        band, share = band_of(amp) if len(amp) else (np.array([]), np.zeros((0, 4)))
        folder = OUT / r["split"] / r["label"]
        folder.mkdir(parents=True, exist_ok=True)

        t0 = d["ts"][0] if len(d["ts"]) else 0.0
        with open(folder / f"{r['capture_id']}.csv", "w", encoding="utf-8", newline="") as f:
            f.write(",".join(header) + "\n")
            for k in range(len(amp)):
                f.write(",".join([str(k), f"{d['ts'][k] - t0:.6f}", str(int(d["seq"][k])),
                                  band[k]] + [f"{s:.4f}" for s in share[k]]
                                 + [f"{v:.6g}" for v in amp[k]]) + "\n")
        if make_png and len(amp):
            save_png(folder / f"{r['capture_id']}.png", amp, band,
                     f"{r['capture_id']}  ({r['label']}, {r['split']})")

        summary.append({"capture_id": r["capture_id"], "split": r["split"], "label": r["label"],
                        "packets": len(amp), "full": int((band == "full").sum()),
                        "narrow": int((band == "narrow").sum()),
                        "other": int((band == "other").sum()),
                        "dropped_while_decoding": sum(d["rejected"].values())})
        if i % 100 == 0:
            print(f"    {i}/{len(caps)}")

    write_csv_rows(OUT / "summary.csv", summary)
    print(f"Done. Open any file in {OUT.relative_to(ROOT)}/<split>/<LABEL>/, "
          "and see summary.csv for counts per capture.")


if __name__ == "__main__":
    main()