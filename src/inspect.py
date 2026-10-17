"""
src/inspect_data.py
Step 1 (Hint 1): look inside the raw files BEFORE writing any decoder.
Uses only the Python standard library - nothing to install.

Run from the repo root:
    python src/inspect_data.py

Output is printed and also saved to reports/data_inspection.txt
"""
import csv
import statistics
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXP2 = RAW / "Experiment-2"
EXP3 = RAW / "Experiment-3"
REPORT = ROOT / "reports" / "data_inspection.txt"

NEXMON_PORT = 5500        # Nexmon sends CSI as UDP packets to this port
SAMPLES_PER_CLASS = 2     # how many files per class to show in detail
WINDOW = 150              # official window length

_lines = []


def say(msg=""):
    print(msg)
    _lines.append(msg)


# =====================================================================
# PCAP reading (plain struct, no library)
# =====================================================================
def read_pcap(path):
    """Return (linktype, [(timestamp_sec, frame_bytes), ...])."""
    data = path.read_bytes()
    if len(data) < 24:
        raise ValueError("file too small to be a pcap")
    magic = data[:4]
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1e6),  # little-endian, microseconds
        b"\xa1\xb2\xc3\xd4": (">", 1e6),  # big-endian, microseconds
        b"\x4d\x3c\xb2\xa1": ("<", 1e9),  # little-endian, nanoseconds
        b"\xa1\xb2\x3c\x4d": (">", 1e9),  # big-endian, nanoseconds
    }
    if magic == b"\x0a\x0d\x0d\x0a":
        raise ValueError("pcapng format - needs a different reader")
    if magic not in formats:
        raise ValueError(f"unknown pcap magic {magic.hex()}")
    endian, ts_div = formats[magic]
    linktype = struct.unpack(endian + "I", data[20:24])[0]

    packets, off = [], 24
    while off + 16 <= len(data):
        sec, frac, incl_len, _ = struct.unpack(endian + "IIII", data[off:off + 16])
        off += 16
        packets.append((sec + frac / ts_div, data[off:off + incl_len]))
        off += incl_len
    return linktype, packets


def udp_payload(frame, linktype):
    """Return (dst_port, payload) for an IPv4/UDP frame, else None."""
    if linktype == 1:                       # Ethernet
        if len(frame) < 14:
            return None
        off = 14
        eth_type = struct.unpack(">H", frame[12:14])[0]
        if eth_type == 0x8100:              # VLAN tag
            eth_type = struct.unpack(">H", frame[16:18])[0]
            off = 18
        if eth_type != 0x0800:
            return None
    elif linktype in (101, 228):            # raw IPv4
        off = 0
    else:
        return None
    ip = frame[off:]
    if len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 17:   # IPv4 + UDP only
        return None
    udp = ip[(ip[0] & 0x0F) * 4:]
    if len(udp) < 8:
        return None
    return struct.unpack(">H", udp[2:4])[0], udp[8:]


def nexmon_header(p):
    """Read the 18-byte Nexmon CSI header (from the public nexmon_csi docs)."""
    if len(p) < 18:
        return None
    magic, rssi, fc = struct.unpack("<HbB", p[0:4])
    seq, core_ss, chanspec, chip = struct.unpack("<HHHH", p[10:18])
    return {
        "magic": hex(magic), "rssi": rssi, "fc": hex(fc),
        "mac": p[4:10].hex(":"), "seq": seq,
        "core": core_ss & 0x7, "stream": (core_ss >> 3) & 0x7,
        "chanspec": hex(chanspec), "chip": hex(chip),
        "csi_bytes": len(p) - 18,
    }


def analyze_pcap(path):
    linktype, packets = read_pcap(path)
    info = {
        "linktype": linktype, "total": len(packets), "csi": 0,
        "ports": Counter(), "lengths": Counter(), "magic": Counter(),
        "chanspec": Counter(), "chip": Counter(), "core_stream": Counter(),
        "mac": Counter(), "seqs": [], "times": [], "rssi": [], "first_csi": None,
    }
    for ts, frame in packets:
        u = udp_payload(frame, linktype)
        if u is None:
            continue
        port, payload = u
        info["ports"][port] += 1
        if port != NEXMON_PORT:
            continue
        h = nexmon_header(payload)
        if h is None:
            continue
        info["csi"] += 1
        info["lengths"][len(payload)] += 1
        info["magic"][h["magic"]] += 1
        info["chanspec"][h["chanspec"]] += 1
        info["chip"][h["chip"]] += 1
        info["core_stream"][(h["core"], h["stream"])] += 1
        info["mac"][h["mac"]] += 1
        info["seqs"].append(h["seq"])
        info["times"].append(ts)
        info["rssi"].append(h["rssi"])
        if info["first_csi"] is None:
            info["first_csi"] = payload[18:18 + 16].hex(" ")
    return info


def describe_pcap(path, info):
    say(f"\n  File: {path.relative_to(RAW)}")
    say(f"    link type          : {info['linktype']} (1 = Ethernet)")
    say(f"    all packets        : {info['total']}")
    say(f"    UDP ports seen     : {dict(info['ports'])}")
    say(f"    CSI packets (5500) : {info['csi']}")
    if not info["csi"]:
        return
    say(f"    payload lengths    : {dict(info['lengths'])}")
    say(f"    -> CSI bytes       : {[l - 18 for l in info['lengths']]} "
        f"(/4 = {[(l - 18) / 4 for l in info['lengths']]} values)")
    say(f"    magic              : {dict(info['magic'])}")
    say(f"    chanspec           : {dict(info['chanspec'])}")
    say(f"    chip version       : {dict(info['chip'])}")
    say(f"    (core, stream)     : {dict(info['core_stream'])}")
    say(f"    source MACs        : {dict(info['mac'])}")
    s = info["seqs"]
    out_of_order = sum(1 for a, b in zip(s, s[1:]) if b < a)
    say(f"    seq first..last    : {s[0]} .. {s[-1]}   "
        f"(unique {len(set(s))}, went backwards {out_of_order} times)")
    t = info["times"]
    if len(t) > 1:
        gaps = [b - a for a, b in zip(t, t[1:])]
        say(f"    duration           : {t[-1] - t[0]:.3f} s, "
            f"median gap {statistics.median(gaps) * 1000:.2f} ms")
    say(f"    RSSI min/max       : {min(info['rssi'])} / {max(info['rssi'])}")
    say(f"    first 16 CSI bytes : {info['first_csi']}")


def inspect_room_a():
    say("=" * 72)
    say("ROOM A  (Experiment-2, raw .pcap)")
    say("=" * 72)
    pcaps = sorted(EXP2.rglob("*.pcap"))
    if not pcaps:
        say(f"  No .pcap files found under {EXP2}")
        return

    groups = {}
    for p in pcaps:
        split = "validation" if "validation" in (q.lower() for q in p.parts) else "development"
        groups.setdefault((split, p.parent.name.lower()), []).append(p)

    say(f"\n--- Detailed look at {SAMPLES_PER_CLASS} development file(s) per class ---")
    results = {}
    for (split, cls), files in sorted(groups.items()):
        for i, p in enumerate(files):
            try:
                results[p] = analyze_pcap(p)
            except Exception as e:  # noqa: BLE001
                results[p] = None
                say(f"  ERROR reading {p.name}: {e}")
                continue
            if split == "development" and i < SAMPLES_PER_CLASS:
                describe_pcap(p, results[p])

    say("\n--- Summary over ALL Room A files ---")
    say(f"  {'split':<12}{'class':<10}{'files':>6}{'csi min':>9}{'median':>8}"
        f"{'max':>7}{'<150':>6}")
    all_len, all_chan, all_chip = Counter(), Counter(), Counter()
    for (split, cls), files in sorted(groups.items()):
        counts = [results[p]["csi"] for p in files if results.get(p)]
        for p in files:
            if results.get(p):
                all_len.update(results[p]["lengths"])
                all_chan.update(results[p]["chanspec"])
                all_chip.update(results[p]["chip"])
        if counts:
            short = sum(c < WINDOW for c in counts)
            say(f"  {split:<12}{cls:<10}{len(files):>6}{min(counts):>9}"
                f"{int(statistics.median(counts)):>8}{max(counts):>7}{short:>6}")
    say(f"\n  payload lengths (all files): {dict(all_len)}")
    say(f"  chanspec (all files)       : {dict(all_chan)}")
    say(f"  chip version (all files)   : {dict(all_chip)}")


# =====================================================================
# Room B CSV
# =====================================================================
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def analyze_csv(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = [r for r in csv.reader(f) if r]
    has_header = bool(rows) and not all(is_number(c) for c in rows[0] if c.strip())
    header = rows[0] if has_header else None
    body = rows[1:] if has_header else rows
    bad = zero_rows = 0
    vmin, vmax = float("inf"), float("-inf")
    for r in body:
        vals = []
        for c in r:
            if c.strip() == "" or not is_number(c):
                bad += 1
            else:
                vals.append(float(c))
        if vals:
            vmin, vmax = min(vmin, min(vals)), max(vmax, max(vals))
            if all(v == 0 for v in vals):
                zero_rows += 1
    return {
        "header": header, "rows": len(body),
        "cols": Counter(len(r) for r in body), "bad_cells": bad,
        "zero_rows": zero_rows, "min": vmin, "max": vmax,
        "first_row": body[0][:5] if body else [],
        "dupes": len(body) - len({tuple(r) for r in body}),
    }


def inspect_room_b():
    say("\n" + "=" * 72)
    say("ROOM B  (Experiment-3, amplitude .csv)")
    say("=" * 72)
    csvs = sorted(EXP3.rglob("*.csv"))
    if not csvs:
        say(f"  No .csv files found under {EXP3}")
        return

    groups = {}
    for p in csvs:
        groups.setdefault(p.name.split("_")[0].capitalize(), []).append(p)

    say(f"\n--- Detailed look at {SAMPLES_PER_CLASS} file(s) per class ---")
    results = {}
    for cls, files in sorted(groups.items()):
        for i, p in enumerate(files):
            results[p] = analyze_csv(p)
            if i < SAMPLES_PER_CLASS:
                r = results[p]
                say(f"\n  File: {p.name}")
                say(f"    header row        : "
                    f"{(r['header'][:4] + ['...'] + r['header'][-2:]) if r['header'] else 'none'}")
                say(f"    data rows         : {r['rows']}")
                say(f"    columns per row   : {dict(r['cols'])}")
                say(f"    empty/non-numeric : {r['bad_cells']}")
                say(f"    all-zero rows     : {r['zero_rows']}")
                say(f"    duplicate rows    : {r['dupes']}")
                say(f"    value min / max   : {r['min']:.3f} / {r['max']:.3f}")
                say(f"    first row (5 vals): {r['first_row']}")

    say("\n--- Summary over ALL Room B files ---")
    say(f"  {'class':<10}{'files':>6}{'rows min':>10}{'median':>8}{'max':>7}"
        f"{'<150':>6}{'bad files':>11}")
    all_cols = Counter()
    for cls, files in sorted(groups.items()):
        rows = [results[p]["rows"] for p in files]
        bad = sum(1 for p in files if results[p]["bad_cells"] or results[p]["zero_rows"])
        for p in files:
            all_cols.update(results[p]["cols"])
        say(f"  {cls:<10}{len(files):>6}{min(rows):>10}{int(statistics.median(rows)):>8}"
            f"{max(rows):>7}{sum(r < WINDOW for r in rows):>6}{bad:>11}")
    say(f"\n  columns per row (all files): {dict(all_cols)}")


if __name__ == "__main__":
    inspect_room_a()
    inspect_room_b()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\nSaved report to {REPORT.relative_to(ROOT)}")