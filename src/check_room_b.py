"""
src/check_room_b.py
Room B has two kinds of files (with and without "Amp" in the name, and two
kinds of header). This script checks:
  1. how many files of each name pattern and header type per class
  2. whether any two files contain the SAME recording (duplicates)
Duplicates matter: if a copy lands in development and the other copy in
validation, that is leakage.

Run from the repo root:
    python src/check_room_b.py
Output is printed and saved to reports/room_b_check.txt
"""
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP3 = ROOT / "data" / "raw" / "Experiment-3"
REPORT = ROOT / "reports" / "room_b_check.txt"
OVERLAP_SHOW = 0.5   # report file pairs that share more than 50% of rows

_lines = []


def say(msg=""):
    print(msg)
    _lines.append(msg)


def read_rows(path):
    """Return (header_type, data_rows). Header row is removed."""
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    first = rows[0]
    if first[0].lower().startswith("subcarrier"):
        return "Subcarrier_N names", rows[1:]
    if first == [str(i) for i in range(len(first))]:
        return "index row 0..255", rows[1:]
    return "no header", rows


def main():
    files = sorted(EXP3.rglob("*.csv"))
    if not files:
        say(f"No .csv files found under {EXP3}")
        return

    pattern_count = defaultdict(Counter)    # class -> {pattern: n}
    header_count = defaultdict(Counter)     # class -> {header type: n}
    combo = Counter()                       # (pattern, header) -> n
    row_counts = Counter()
    file_hash, row_sets = {}, {}

    for p in files:
        cls = p.name.split("_")[0].capitalize()
        pattern = "with _Amp_" if "_amp_" in p.name.lower() else "without _Amp_"
        htype, rows = read_rows(p)
        pattern_count[cls][pattern] += 1
        header_count[cls][htype] += 1
        combo[(pattern, htype)] += 1
        row_counts[len(rows)] += 1
        text = "\n".join(",".join(r) for r in rows)
        file_hash[p] = hashlib.md5(text.encode()).hexdigest()
        row_sets[p] = {hashlib.md5(",".join(r).encode()).hexdigest() for r in rows}

    say("=== 1. Files per class: name pattern and header type ===")
    for cls in sorted(pattern_count):
        say(f"  {cls:<10} names: {dict(pattern_count[cls])}")
        say(f"  {'':<10} header: {dict(header_count[cls])}")
    say("\n  name pattern x header type (all classes):")
    for (pat, h), n in sorted(combo.items()):
        say(f"    {pat:<15} + {h:<20}: {n}")
    say(f"\n  data rows per file after removing header: {dict(row_counts)}")

    say("\n=== 2. Exact duplicate files (same data, different name) ===")
    groups = defaultdict(list)
    for p, h in file_hash.items():
        groups[h].append(p.name)
    dupes = [g for g in groups.values() if len(g) > 1]
    if not dupes:
        say("  none")
    for g in dupes[:30]:
        say(f"  {g}")
    if len(dupes) > 30:
        say(f"  ... and {len(dupes) - 30} more groups")
    say(f"  total duplicate groups: {len(dupes)}")

    say(f"\n=== 3. File pairs sharing more than {OVERLAP_SHOW:.0%} of their rows ===")
    index = defaultdict(list)                 # row hash -> files
    for p, s in row_sets.items():
        for h in s:
            index[h].append(p)
    shared = Counter()
    for fl in index.values():
        if 1 < len(fl) < 50:                  # skip very common rows
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    shared[(fl[i], fl[j])] += 1
    pairs = []
    for (a, b), n in shared.items():
        frac = n / min(len(row_sets[a]), len(row_sets[b]))
        if frac > OVERLAP_SHOW and file_hash[a] != file_hash[b]:
            pairs.append((frac, a.name, b.name))
    pairs.sort(reverse=True)
    if not pairs:
        say("  none")
    for frac, a, b in pairs[:30]:
        say(f"  {frac:6.1%}  {a}  <->  {b}")
    if len(pairs) > 30:
        say(f"  ... and {len(pairs) - 30} more pairs")
    say(f"  total overlapping pairs: {len(pairs)}")

    say("\n=== 4. Same number, two names (e.g. Empty_11 vs Empty_Amp_11) ===")
    by_key = defaultdict(list)
    for p in files:
        m = re.match(r"([A-Za-z]+)_(?:Amp_)?(\d+)\.csv$", p.name, re.I)
        if m:
            by_key[(m.group(1).capitalize(), int(m.group(2)))].append(p)
    both = [v for v in by_key.values() if len(v) == 2]
    same = sum(1 for a, b in both if file_hash[a] == file_hash[b])
    say(f"  numbers that exist in both name styles: {len(both)}")
    say(f"  ... of those, identical content: {same}")


if __name__ == "__main__":
    main()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\nSaved report to {REPORT.relative_to(ROOT)}")