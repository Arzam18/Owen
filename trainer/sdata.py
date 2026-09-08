"""
sdata layout (packed, 70 bytes per record):
  board[64] uint8  (0..11 piece, 12 empty)
  stm       uint8  (0 white, 1 black)
  eval      int16  (centipawns)
  result    uint8  (0 loss, 1 draw, 2 win) from stm view
  ply       uint8
Total: 68 bytes (with padding may be 70 due to pack(1) in C++).
We use numpy to read efficiently.
"""
import struct, numpy as np, os

RECORD_FMT = "<64B B h B B" # little endian
RECORD_SIZE = struct.calcsize(RECORD_FMT) # 68+? 64+1+2+1+1=69? actually 69
# C++ SDataRecord packed is 64+1+2+1+1 = 69 bytes (pragma pack 1)
RECORD_SIZE = 69

def iter_records(path, limit=None):
    sz = os.path.getsize(path)
    n = sz // RECORD_SIZE
    if limit: n = min(n, limit)
    with open(path, "rb") as f:
        for i in range(n):
            b = f.read(RECORD_SIZE)
            if len(b) < RECORD_SIZE: break
            board = np.frombuffer(b[0:64], dtype=np.uint8)
            stm = b[64]
            ev = struct.unpack_from("<h", b, 65)[0]
            result = b[67]
            ply = b[68]
            yield {"board": board, "stm": stm, "eval": ev, "result": result, "ply": ply}

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
