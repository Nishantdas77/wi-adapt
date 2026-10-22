"""
src/tools/check_room_a_bandwidth.py
Room A has two kinds of packets:
  full band  : energy on all 256 subcarriers (80 MHz frame)
  narrow band: energy only on one 64-subcarrier quarter (20 MHz frame on the primary channel)

For every Room A capture this script counts both kinds and checks how they relate to
the 0xFFFF sequence field, the frame-control byte, time order and activity.
Decision we need: can we keep ONLY full-band packets and still cut 150-row windows?

Run from the repo root:
    python src/tools/check_room_a_bandwidth.py
Saves:
    reports/room_a_bandwidth.txt
    reports/room_a_bandwidth_per_capture.csv
    results/figures/checks/room_a_bandwidth_examples.png
    results/figures/checks/room_a_full_band_count.png
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ORDER_A, ROOT, load_config, path_of, read_csv_rows, write_csv_rows
from csi_decode import SEQ_INVALID, read_room_a, to_frequency_order

QUARTER_SHARE = 0.05   # a quarter counts as "used" if it holds > 5% of the packet energy
WINDOW = 150
_lines = []


def say(msg=""):
    print(msg)
    _lines.append(str(msg))


def band_info(amp):
    """Per packet: number of used 64-subcarrier quarters and the strongest quarter."""
    e = (to_frequency_order(amp).astype(np.float64) ** 2).reshape(len(amp), 4, 64).sum(axis=2)
    share = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)
    return (share > QUARTER_SHARE).sum(axis=1), share.argmax(axis=1)


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def main():
    cfg = load_config()
    caps = [r for r in read_csv_rows(path_of(cfg, "splits")) if r["room"] == "A"]
    per_cap, examples = [], {}
    cross_seq, cross_fc, narrow_quarter = Counter(), Counter(), Counter()

    print(f"checking {len(caps)} Room A captures ...")
    for r in sorted(caps, key=lambda r: r["capture_id"]):
        d = read_room_a(ROOT / r["path"], scale=cfg["room_a"]["scale"])
        if len(d["amp"]) == 0:
            continue
        n_used, top_q = band_info(d["amp"])
        full = n_used == 4
        narrow = n_used == 1
        for f, nu, tq, s, fc in zip(full, narrow, top_q, d["seq"], d["fc"]):
            kind = "full" if f else ("narrow" if nu else "other")
            cross_seq[(kind, "0xFFFF" if s == SEQ_INVALID else "normal")] += 1
            cross_fc[(kind, hex(int(fc)))] += 1
            if nu:
                narrow_quarter[int(tq)] += 1
        first_full = int(np.argmax(full)) if full.any() else -1
        per_cap.append({
            "capture_id": r["capture_id"], "split": r["split"], "label": r["label"],
            "packets": len(full), "full_band": int(full.sum()), "narrow_band": int(narrow.sum()),
            "other": int((~full & ~narrow).sum()), "first_full_index": first_full,
            "longest_full_run": longest_run(full),
            "full_band_at_least_150": int(full.sum() >= WINDOW),
        })
        if r["split"] == "development" and r["label"] not in examples:
            examples[r["label"]] = (n_used, d["seq"] == SEQ_INVALID)

    write_csv_rows(path_of(cfg, "reports") / "room_a_bandwidth_per_capture.csv", per_cap)

    # ---------------- report ----------------
    say("ROOM A BANDWIDTH CHECK")
    say("=" * 70)
    say(f"a quarter is 'used' if it has > {QUARTER_SHARE:.0%} of the packet energy")
    say("full band = all 4 quarters used (80 MHz) | narrow band = 1 quarter (20 MHz)\n")
    say(f"{'split':<12}{'class':<9}{'caps':>5}{'full med':>9}{'full min':>9}{'full max':>9}"
        f"{'>=150':>7}{'narrow med':>11}{'1st full med':>13}")
    groups = defaultdict(list)
    for p in per_cap:
        groups[(p["split"], p["label"])].append(p)
    for split in ("development", "validation"):
        for lab in ORDER_A:
            g = groups[(split, lab)]
            if not g:
                continue
            f = [p["full_band"] for p in g]
            nb = [p["narrow_band"] for p in g]
            ff = [p["first_full_index"] for p in g if p["first_full_index"] >= 0]
            say(f"{split:<12}{lab:<9}{len(g):>5}{int(np.median(f)):>9}{min(f):>9}{max(f):>9}"
                f"{sum(x >= WINDOW for x in f):>7}{int(np.median(nb)):>11}"
                f"{int(np.median(ff)) if ff else -1:>13}")
    tot = Counter((p["split"], p["full_band_at_least_150"]) for p in per_cap)
    for split in ("development", "validation"):
        n = tot[(split, 0)] + tot[(split, 1)]
        say(f"\n{split}: {tot[(split, 1)]}/{n} captures have >= {WINDOW} full-band packets")

    say("\npacket kind x sequence field:")
    for k, v in sorted(cross_seq.items()):
        say(f"    {k[0]:<7} {k[1]:<7} {v}")
    say("\npacket kind x frame-control byte:")
    for k, v in sorted(cross_fc.items()):
        say(f"    {k[0]:<7} fc={k[1]:<6} {v}")
    say(f"\nnarrow-band packets: strongest quarter (0 = lowest frequencies): {dict(narrow_quarter)}")

    rep = path_of(cfg, "reports") / "room_a_bandwidth.txt"
    rep.write_text("\n".join(_lines), encoding="utf-8")

    # ---------------- figures ----------------
    figdir = path_of(cfg, "results") / "figures" / "checks"
    figdir.mkdir(parents=True, exist_ok=True)
    labs = [l for l in ORDER_A if l in examples]
    fig, axes = plt.subplots(len(labs), 1, figsize=(10, 1.4 * len(labs)), sharex=True)
    for ax, lab in zip(np.atleast_1d(axes), labs):
        n_used, is_ff = examples[lab]
        x = np.arange(len(n_used))
        ax.scatter(x[~is_ff], n_used[~is_ff], s=5, label="normal seq")
        ax.scatter(x[is_ff], n_used[is_ff], s=5, color="red", label="seq 0xFFFF")
        ax.set_yticks([1, 2, 3, 4]); ax.set_ylabel(lab, rotation=0, ha="right")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="center right")
    np.atleast_1d(axes)[-1].set_xlabel("packet index (time order)")
    fig.suptitle("Room A: used quarters per packet (4 = full 80 MHz, 1 = 20 MHz)")
    fig.tight_layout()
    fig.savefig(figdir / "room_a_bandwidth_examples.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for split, c in (("development", "#4C72B0"), ("validation", "#DD8452")):
        ax.hist([p["full_band"] for p in per_cap if p["split"] == split], bins=40,
                alpha=0.7, label=split, color=c)
    ax.axvline(WINDOW, color="red", ls="--", label="150")
    ax.set_xlabel("full-band packets per capture"); ax.set_ylabel("captures")
    ax.set_title("Room A: how many full 80 MHz packets per capture?"); ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "room_a_full_band_count.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved {rep.relative_to(ROOT)} and figures in {figdir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()