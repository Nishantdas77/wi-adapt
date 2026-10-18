"""
src/csi_decode.py
Our own Python readers for both rooms. Every later step uses these.

ROOM A (.pcap, Nexmon CSI, BCM4366c0, 80 MHz)
    pcap file -> Ethernet / IPv4 / UDP (port 5500) -> 18-byte Nexmon header
    -> 1024 CSI bytes = 256 little-endian 32-bit words (one per subcarrier)
    -> 256 complex numbers -> amplitude

    Packed word layout (from the public Nexmon CSI documentation):
        bit  29      sign of the real part (I)
        bits 18-28   real mantissa (11 bits)
        bit  17      sign of the imaginary part (Q)
        bits 6-16    imaginary mantissa (11 bits)
        bits 0-5     shared exponent (6-bit, signed)
        value = sign * mantissa * 2**exponent

    Two scaling choices:
        "exact"     : sign * mantissa * 2**exponent  (the true float value)
        "autoscale" : the same, then every packet is shifted by one power of
                      two so its largest value sits near 2**10..2**11
                      (the convention used in the public Nexmon tools)

ROOM B (.csv)
    first row = header (either Subcarrier_1..256 or 0..255) -> skip it
    then 500 rows x 256 amplitude columns
"""
import struct
from collections import Counter

import numpy as np

NEXMON_PORT = 5500
NEXMON_HEADER = 18
N_SUB = 256
PAYLOAD_LEN = NEXMON_HEADER + 4 * N_SUB      # 1042 bytes
NEXMON_MAGIC = 0x1111
SEQ_INVALID = 0xFFFF
AUTOSCALE_BITS = 10                          # target top bit for "autoscale"
MANTISSA_BITS = 12                           # values below 2**-12 of a mantissa -> 0


# =====================================================================
# 1. PCAP container (plain struct, no extra library)
# =====================================================================
def read_pcap(path):
    """Return (linktype, [(timestamp_sec, frame_bytes), ...])."""
    data = open(path, "rb").read()
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1e6), b"\xa1\xb2\xc3\xd4": (">", 1e6),
        b"\x4d\x3c\xb2\xa1": ("<", 1e9), b"\xa1\xb2\x3c\x4d": (">", 1e9),
    }
    if len(data) < 24 or data[:4] not in formats:
        raise ValueError(f"{path}: not a classic pcap file")
    endian, ts_div = formats[data[:4]]
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
    if linktype == 1:                        # Ethernet
        if len(frame) < 14:
            return None
        off, eth_type = 14, struct.unpack(">H", frame[12:14])[0]
        if eth_type == 0x8100:               # VLAN tag
            off, eth_type = 18, struct.unpack(">H", frame[16:18])[0]
        if eth_type != 0x0800:
            return None
    elif linktype in (101, 228):             # raw IPv4
        off = 0
    else:
        return None
    ip = frame[off:]
    if len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 17:
        return None
    udp = ip[(ip[0] & 0x0F) * 4:]
    if len(udp) < 8:
        return None
    return struct.unpack(">H", udp[2:4])[0], udp[8:]


# =====================================================================
# 2. Packed CSI -> complex numbers
# =====================================================================
def exponents(words):
    """Signed 6-bit exponent of every word."""
    e = (words.astype(np.int64) & 0x3F)
    return np.where(e >= 32, e - 64, e)


def unpack_csi(words, scale="exact"):
    """
    words : (n_packets, 256) uint32 array
    return: (n_packets, 256) complex64 array
    """
    w = words.astype(np.int64)
    exp = exponents(w)
    m_i = (w >> 18) & 0x7FF
    m_q = (w >> 6) & 0x7FF
    s_i = np.where((w >> 29) & 1, -1.0, 1.0)
    s_q = np.where((w >> 17) & 1, -1.0, 1.0)

    if scale == "exact":
        e = exp.astype(np.float64)
    elif scale == "autoscale":
        x = m_i | m_q
        top_bit = np.where(x > 0, exp + np.floor(np.log2(np.maximum(x, 1))), -np.inf)
        packet_top = top_bit.max(axis=-1, keepdims=True)
        shift = np.where(np.isfinite(packet_top), AUTOSCALE_BITS - packet_top, 0.0)
        e = exp + shift
        e = np.where(e < -MANTISSA_BITS, -np.inf, e)     # too small -> 0
    else:
        raise ValueError("scale must be 'exact' or 'autoscale'")

    factor = np.exp2(e)
    return (s_i * m_i * factor + 1j * s_q * m_q * factor).astype(np.complex64)


def amplitude(csi):
    """|I + jQ| as float32."""
    return np.abs(csi).astype(np.float32)


def to_frequency_order(x):
    """Nexmon stores bins as 0..127, -128..-1. fftshift puts them low -> high."""
    return np.fft.fftshift(x, axes=-1)


# =====================================================================
# 3. Room A: one .pcap capture -> amplitude matrix
# =====================================================================
def read_room_a(path, scale="exact", frequency_order=False):
    """
    Returns a dict:
        amp      (N, 256) float32  amplitude, rows sorted by time
        words    (N, 256) uint32   raw packed words (for checks)
        ts       (N,)  seconds
        seq      (N,)  802.11 sequence-control field from the Nexmon header
        fc       (N,)  frame-control byte
        rejected Counter  why packets were dropped
    """
    linktype, frames = read_pcap(path)
    rejected = Counter()
    keep, ts, seq, fc = [], [], [], []
    for t, frame in frames:
        u = udp_payload(frame, linktype)
        if u is None:
            rejected["not IPv4/UDP"] += 1
            continue
        port, p = u
        if port != NEXMON_PORT:
            rejected["wrong UDP port"] += 1
            continue
        if len(p) != PAYLOAD_LEN:
            rejected[f"wrong payload length ({len(p)})"] += 1
            continue
        if struct.unpack_from("<H", p, 0)[0] != NEXMON_MAGIC:
            rejected["wrong magic"] += 1
            continue
        keep.append(p[NEXMON_HEADER:])
        ts.append(t)
        fc.append(p[3])
        seq.append(struct.unpack_from("<H", p, 10)[0])

    if keep:
        words = np.frombuffer(b"".join(keep), dtype="<u4").reshape(-1, N_SUB)
    else:
        words = np.zeros((0, N_SUB), dtype="<u4")
    order = np.argsort(np.asarray(ts), kind="stable")     # keep time order
    words = words[order]
    amp = amplitude(unpack_csi(words, scale))
    if frequency_order:
        amp = to_frequency_order(amp)
    return {
        "amp": amp, "words": words,
        "ts": np.asarray(ts)[order], "seq": np.asarray(seq)[order],
        "fc": np.asarray(fc)[order], "rejected": rejected,
    }


# =====================================================================
# 4. Room B: one .csv capture -> amplitude matrix
# =====================================================================
def read_room_b(path):
    """Skip the header row, return (rows, 256) float32 amplitude."""
    amp = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    if amp.shape[1] != N_SUB:
        raise ValueError(f"{path}: expected {N_SUB} columns, got {amp.shape[1]}")
    return amp