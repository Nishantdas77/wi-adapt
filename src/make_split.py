"""
src/make_splits.py
Builds ONE capture-level table for both rooms and fixes the official
development / validation split.

  Room A (Experiment-2): validation = the supplied validation/ folder (25 per class)
  Room B (Experiment-3): validation = 25 captures per class, chosen with seed 42
                         after sorting the filenames. Everything else = development.

The raw data is NOT moved or copied. The split lives only in splits/captures.csv,
so the raw folder stays read-only. Commit the splits/ folder to GitHub
(it is small and it is the provenance record of your split).

Run once from the repo root:
    python src/make_splits.py
Running it again gives exactly the same result.
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP2 = ROOT / "data" / "raw" / "Experiment-2"
EXP3 = ROOT / "data" / "raw" / "Experiment-3"
OUT_DIR = ROOT / "splits"

SEED = 42
VAL_PER_CLASS = 25

# one common label name for each activity, used by every later step
LABELS_A = {"fall": "FALL", "sit": "SIT", "sitdown": "SITDOWN",
            "stand": "STAND", "standup": "STANDUP", "walk": "WALK"}
LABELS_B = {"empty": "EMPTY", "lying": "LYING", "sitting": "SIT",
            "standing": "STAND", "walking": "WALK"}

EXPECTED = {("A", "development"): 934, ("A", "validation"): 150,
            ("B", "development"): 1076, ("B", "validation"): 125}


def rel(p):
    return p.relative_to(ROOT).as_posix()


def room_a_rows():
    rows = []
    for p in sorted(EXP2.rglob("*.pcap")):
        folder = p.parent.name.lower()
        if folder not in LABELS_A:
            raise SystemExit(f"Unknown Room A class folder: {p.parent}")
        split = "validation" if "validation" in (q.lower() for q in p.parts) else "development"
        prefix = "A_val" if split == "validation" else "A_dev"
        rows.append({
            "capture_id": f"{prefix}_{folder}_{p.stem}",
            "room": "A", "label": LABELS_A[folder], "split": split,
            "source_format": "pcap", "path": rel(p),
        })
    return rows


def room_b_rows():
    by_class = {}
    for p in EXP3.rglob("*.csv"):
        cls = p.name.split("_")[0].lower()
        if cls not in LABELS_B:
            raise SystemExit(f"Unknown Room B class in filename: {p.name}")
        by_class.setdefault(cls, []).append(p)

    rows = []
    for cls in sorted(by_class):
        files = sorted(by_class[cls], key=lambda p: p.name)       # deterministic order
        chosen = set(random.Random(SEED).sample(files, VAL_PER_CLASS))
        for p in files:
            split = "validation" if p in chosen else "development"
            rows.append({
                "capture_id": f"B_{p.stem}",
                "room": "B", "label": LABELS_B[cls], "split": split,
                "source_format": "csv", "path": rel(p),
            })
    return rows


def main():
    rows = room_a_rows() + room_b_rows()

    # ---- safety checks ----
    ids = [r["capture_id"] for r in rows]
    assert len(ids) == len(set(ids)), "capture_id is not unique!"
    paths = [r["path"] for r in rows]
    assert len(paths) == len(set(paths)), "a file appears twice!"

    counts = Counter((r["room"], r["split"]) for r in rows)
    print("Captures per room and split:")
    all_ok = True
    for key, exp in EXPECTED.items():
        ok = counts[key] == exp
        all_ok &= ok
        print(f"  Room {key[0]}  {key[1]:<12} {counts[key]:>5}   "
              f"(expected {exp}) {'OK' if ok else 'MISMATCH'}")

    print("\nValidation captures per class:")
    val = Counter((r["room"], r["label"]) for r in rows if r["split"] == "validation")
    for (room, label), n in sorted(val.items()):
        print(f"  Room {room}  {label:<8} {n}")

    # ---- write ----
    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "captures.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    info = {
        "seed": SEED,
        "val_per_class": VAL_PER_CLASS,
        "room_A_rule": "validation = supplied validation/ folder",
        "room_B_rule": ("per class: sort filenames with Python sorted() by name, "
                        "then random.Random(42).sample(files, 25)"),
        "counts": {f"{k[0]}_{k[1]}": v for k, v in sorted(counts.items())},
    }
    (OUT_DIR / "split_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nSaved splits/captures.csv ({len(rows)} captures) and splits/split_info.json")
    if not all_ok:
        print("WARNING: some counts do not match the expected numbers.")


if __name__ == "__main__":
    main()