"""
src/build_intermediate.py
STEP 2: bring BOTH rooms into the SAME format.

  Room A .pcap -> decode CSI -> amplitude (256)  \
                                                  > quality control -> one CSV per capture
  Room B .csv  -> skip header row -> amplitude    /   data/intermediate_csv/room_A|room_B/<capture_id>.csv
                                                      columns subcarrier_001 ... subcarrier_256
                                                      one row per valid observation, in time order

Quality control (settings in configs/experiment.yaml -> qc):
  Room A only: packets with wrong UDP port / wrong length / wrong magic are dropped while decoding
  both rooms : rows with NaN/inf, all-zero rows, rows identical to the previous row
A capture with fewer than 150 rows left is marked NOT eligible (no window can be cut).

Outputs:
  data/intermediate_csv/...          (not committed, regenerate with this script)
  reports/qc_log.csv                 one row per capture: what was removed and why
  reports/qc_summary.txt             totals + subcarrier-order decision

Run from the repo root:
    python src/build_intermediate.py
"""
import random
from collections import Counter, defaultdict

import numpy as np

from common import (ROOT, SUBCARRIER_COLUMNS, load_config, path_of,
                    preprocess_version, read_csv_rows, write_csv_rows)
from csi_decode import SEQ_INVALID, read_room_a, read_room_b, to_frequency_order


def norm_rows(a):
    m = a.mean(axis=1, keepdims=True)
    return np.divide(a, m, out=np.zeros_like(a), where=m > 0)


def decide_order(caps, cfg):
    """Room A subcarrier order. 'auto' compares the average spectrum SHAPE of
    Room A (stored order and fftshift) with Room B, using development data only."""
    mode = cfg["room_a"]["subcarrier_order"]
    if mode in ("fftshift", "stored"):
        return mode, {}
    n = cfg["room_a"]["order_check_captures"]
    rng = random.Random(cfg["seed"])
    spec = {}
    for room in ("A", "B"):
        dev = sorted((r for r in caps if r["room"] == room and r["split"] == "development"),
                     key=lambda r: r["path"])
        total, count = np.zeros(256), 0
        for r in rng.sample(dev, min(n, len(dev))):
            amp = (read_room_a(ROOT / r["path"], scale=cfg["room_a"]["scale"])["amp"]
                   if room == "A" else read_room_b(ROOT / r["path"]))
            total += norm_rows(amp).sum(axis=0)
            count += len(amp)
        spec[room] = total / max(count, 1)
    c_stored = float(np.corrcoef(spec["A"], spec["B"])[0, 1])
    c_shift = float(np.corrcoef(to_frequency_order(spec["A"]), spec["B"])[0, 1])
    order = "fftshift" if c_shift > c_stored else "stored"
    return order, {"corr_stored": c_stored, "corr_fftshift": c_shift}


def quality_control(amp, qc):
    """Return (keep_mask, counts, duplicate_mask). Works on rows in time order."""
    keep = np.ones(len(amp), dtype=bool)
    dup_mask = np.zeros(len(amp), dtype=bool)
    counts = {"nonfinite": 0, "all_zero": 0, "duplicate": 0}
    if qc["drop_nonfinite_rows"]:
        bad = ~np.isfinite(amp).all(axis=1)
        counts["nonfinite"] = int(bad.sum())
        keep &= ~bad
    if qc["drop_all_zero_rows"]:
        zero = keep & np.all(amp == 0, axis=1)
        counts["all_zero"] = int(zero.sum())
        keep &= ~zero
    if qc["drop_consecutive_duplicates"]:
        idx = np.flatnonzero(keep)
        if len(idx) > 1:
            same = np.r_[False, np.all(amp[idx[1:]] == amp[idx[:-1]], axis=1)]
            counts["duplicate"] = int(same.sum())
            dup_mask[idx[same]] = True
            keep[idx[same]] = False
    return keep, counts, dup_mask


def main():
    cfg = load_config()
    caps = sorted(read_csv_rows(path_of(cfg, "splits")), key=lambda r: r["capture_id"])
    out_dir = path_of(cfg, "intermediate")
    L = cfg["window"]["length"]
    version = preprocess_version(cfg)

    print("==> deciding Room A subcarrier order (development data only) ...")
    order, order_info = decide_order(caps, cfg)
    print(f"    order = {order} {order_info}")

    qc_rows, reasons = [], Counter()
    dup_by_seq = Counter()
    print(f"==> converting {len(caps)} captures ...")
    for i, r in enumerate(caps, 1):
        if r["room"] == "A":
            d = read_room_a(ROOT / r["path"], scale=cfg["room_a"]["scale"])
            amp = to_frequency_order(d["amp"]) if order == "fftshift" else d["amp"]
            bad_packets = sum(d["rejected"].values())
            reasons.update(d["rejected"])
            seq = d["seq"]
        else:
            amp = read_room_b(ROOT / r["path"])
            bad_packets, seq = 0, None

        raw_rows = len(amp) + bad_packets
        keep, counts, dup_mask = quality_control(amp, cfg["qc"])
        if seq is not None and counts["duplicate"]:
            # were the removed duplicates the 0xFFFF packets?
            n_ff = int(np.sum(dup_mask & (seq == SEQ_INVALID)))
            dup_by_seq["0xFFFF"] += n_ff
            dup_by_seq["normal"] += counts["duplicate"] - n_ff
        clean = amp[keep].astype(np.float32)

        dst = out_dir / f"room_{r['room']}" / f"{r['capture_id']}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(dst, clean, fmt="%.6g", delimiter=",",
                   header=",".join(SUBCARRIER_COLUMNS), comments="")

        qc_rows.append({
            "capture_id": r["capture_id"], "room": r["room"], "label": r["label"],
            "split": r["split"], "raw_rows": raw_rows, "bad_packets": bad_packets,
            "nonfinite": counts["nonfinite"], "all_zero": counts["all_zero"],
            "duplicate": counts["duplicate"], "final_rows": len(clean),
            "eligible": int(len(clean) >= L),
            "intermediate_csv": dst.relative_to(ROOT).as_posix(),
            "preprocess_version": version,
        })
        if i % 200 == 0:
            print(f"    {i}/{len(caps)}")

    reports = path_of(cfg, "reports")
    write_csv_rows(reports / "qc_log.csv", qc_rows)

    # ---------------- summary ----------------
    lines = ["QC SUMMARY", "=" * 60,
             f"preprocess_version: {version}",
             f"Room A scale: {cfg['room_a']['scale']}",
             f"Room A subcarrier order: {order}  {order_info}", ""]
    agg = defaultdict(Counter)
    for q in qc_rows:
        k = (q["room"], q["split"])
        agg[k]["captures"] += 1
        for f in ("raw_rows", "bad_packets", "nonfinite", "all_zero", "duplicate", "final_rows"):
            agg[k][f] += int(q[f])
        agg[k]["not_eligible"] += 1 - int(q["eligible"])
    for (room, split), c in sorted(agg.items()):
        lines.append(f"Room {room} {split:<12} captures {c['captures']:>5} | raw rows {c['raw_rows']:>7}"
                     f" | bad packets {c['bad_packets']:>3} | nonfinite {c['nonfinite']:>3}"
                     f" | all-zero {c['all_zero']:>4} | duplicates {c['duplicate']:>6}"
                     f" | kept {c['final_rows']:>7} | NOT eligible (<{L} rows) {c['not_eligible']:>3}")
    lines.append("")
    lines.append(f"Room A packets dropped while decoding: {dict(reasons) or 'none'}")
    lines.append(f"Room A removed duplicates by seq field: {dict(dup_by_seq) or 'none'}")
    lines.append("")
    lines.append("Not eligible captures per class:")
    ne = Counter((q["room"], q["split"], q["label"]) for q in qc_rows if q["eligible"] == 0)
    for k, v in sorted(ne.items()):
        lines.append(f"    Room {k[0]} {k[1]:<12} {k[2]:<8} {v}")
    if not ne:
        lines.append("    none")

    a_dev = agg[("A", "development")]
    if a_dev["captures"] and a_dev["not_eligible"] > 0.3 * a_dev["captures"]:
        lines += ["", "WARNING: more than 30% of Room A captures are below 150 rows.",
                  "If the duplicates line above is large, the 0xFFFF packets were exact copies.",
                  "Discuss with your TA, or set qc.drop_consecutive_duplicates: false and rerun."]

    text = "\n".join(lines)
    (reports / "qc_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved {out_dir.relative_to(ROOT)}/, reports/qc_log.csv, reports/qc_summary.txt")


if __name__ == "__main__":
    main()
