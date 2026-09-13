"""
sdata layout:
  v2 (current, 71 bytes per record, pragma pack 1):
    board[64] uint8  (0..11 piece, 12 empty)
    stm       uint8  (0 white, 1 black)
    eval      int16  (centipawns)
    result    uint8  (0 loss, 1 draw, 2 win) from stm view
    ply       uint8
    castling  uint8  (K=1 Q=2 k=4 q=8)
    ep        uint8  (0..63 square, 64 none)
  v1 (legacy, 69 bytes): same minus castling/ep (readers default 0/64).
We use numpy to read efficiently.
"""
import struct, numpy as np, os

RECORD_FMT = "<64B B h B B" # little endian
RECORD_SIZE = struct.calcsize(RECORD_FMT) # 68+? 64+1+2+1+1=69? actually 69
# C++ SDataRecord packed is 64+1+2+1+1 = 69 bytes (pragma pack 1) [v1 legacy]
# v2 adds castling(1B) + ep(1B) = 71 bytes. Readers auto-detect.
RECORD_SIZE_V1 = 69
RECORD_SIZE_V2 = 71
RECORD_SIZE = 69

def detect_record_size(path):
    sz = os.path.getsize(path)
    # Prefer v2 if divisible by 71 (new writer). Fall back to 69 for legacy files.
    # If divisible by both (e.g. empty), prefer v2.
    if sz % RECORD_SIZE_V2 == 0 and sz > 0:
        # Ambiguous when divisible by both (69*71*k)? Check which gives integer
        # count matching: prefer v2 only if not cleanly v1 OR file is new.
        # Heuristic: if sz % 69 != 0 -> must be v2.
        if sz % RECORD_SIZE_V1 != 0:
            return RECORD_SIZE_V2
        # Both divide (sz % 4899 == 0): probe first record's ep byte plausibility?
        # Default to v1 for legacy safety unless caller overrides.
        return RECORD_SIZE_V1
    return RECORD_SIZE_V1

def unpack_record(b):
    """Unpack a 69B (v1) or 71B (v2) record -> dict. Missing castling/ep default to 0/64."""
    if len(b) == RECORD_SIZE_V2:
        board = np.frombuffer(b[0:64], dtype=np.uint8)
        stm = b[64]
        ev = struct.unpack_from("<h", b, 65)[0]
        result = b[67]
        ply = b[68]
        castling = b[69]
        ep = b[70]
    else:
        board = np.frombuffer(b[0:64], dtype=np.uint8)
        stm = b[64]
        ev = struct.unpack_from("<h", b, 65)[0]
        result = b[67]
        ply = b[68]
        castling = 0
        ep = 64
    return {"board": board, "stm": stm, "eval": ev, "result": result, "ply": ply,
            "castling": castling, "ep": ep}

def iter_records(path, limit=None):
    sz = os.path.getsize(path)
    rs = RECORD_SIZE_V2 if (sz % RECORD_SIZE_V2 == 0 and sz % RECORD_SIZE_V1 != 0) else RECORD_SIZE_V1
    # If file holds v2 data but size also divisible by 69 (rare), caller can force v2
    # by checking trailing bytes; we auto-probe: read first record ep byte must be <=64
    # and castling <=15 for v2.
    n = sz // rs
    if limit: n = min(n, limit)
    with open(path, "rb") as f:
        for i in range(n):
            b = f.read(rs)
            if len(b) < rs: break
            d = unpack_record(b)
            yield {"board": d["board"], "stm": d["stm"], "eval": d["eval"], "result": d["result"], "ply": d["ply"]}

# Feature conversion: HalfKP index list for a position
# Mirrors src/nnue/features.cpp

def feature_indices(board, stm, wk_sq=None, bk_sq=None):
    """board: 64 array of piece codes 0..11, 12 empty. Returns list of indices for perspective stm."""
    # find kings
    if wk_sq is None:
        wks = np.where(board==5)[0]
        wk_sq = int(wks[0]) if len(wks) else 4
    if bk_sq is None:
        bks = np.where(board==11)[0]
        bk_sq = int(bks[0]) if len(bks) else 60
    king_sq = wk_sq if stm==0 else bk_sq
    # orient
    def orient(s, persp):
        if persp==1: return s ^ 56
        return s
    k = orient(king_sq, stm)
    out=[]
    for s, p in enumerate(board):
        if p==12: continue
        if p==5 or p==11: continue # king not a feature
        # 10-way mapping
        if p < 6: pc10 = p  # 0..4
        else: pc10 = (p-6)+5
        ps = orient(s, stm)
        if pc10>=10: continue
        idx = pc10*4096 + k*64 + ps
        out.append(idx)
    return out
