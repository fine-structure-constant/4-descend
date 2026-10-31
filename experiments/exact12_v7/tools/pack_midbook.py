"""Pack C++-solved positions into the midbook buffers of a model checkpoint.

Reads a solved tsv (history, seven {-1,0,1,-1000} scores) and stores the
canonical (mirror-minimal) orientation of each position keyed by (own, mask);
the lookup canonicalises the query and mirrors outcomes back as needed. The
NN weights come from --base.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support.game import new_board, play_move
from support.tactics import position
from support.runtime import load_model

SOLVED = ROOT / 'data' / 'midbook_solved_v5.tsv'
DEFAULT_BASE = ROOT / 'runs' / 'v7_train4' / 'model.pt'
OUT = ROOT / 'runs' / 'midbook_v5'


def mirror_bits(own, mask):
    mo = mm = 0
    for c in range(7):
        src = 63 << (7 * c)
        shift = 7 * (6 - 2 * c)
        if shift >= 0:
            mo |= (own & src) << shift
            mm |= (mask & src) << shift
        else:
            mo |= (own & src) >> (-shift)
            mm |= (mask & src) >> (-shift)
    return mo, mm


def encode(w):
    code = 0 if w == -1 else 1 if w == 0 else 2 if w == 1 else 3
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=str(DEFAULT_BASE))
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    entries = {}
    bad = 0
    for line in SOLVED.read_text().splitlines():
        parts = line.split('\t')
        if len(parts) != 2:
            bad += 1
            continue
        history, rest = parts
        nums = rest.split()
        if len(nums) != 7:
            bad += 1
            continue
        w = [int(x) if int(x) != -1000 else 3 for x in nums]
        board = new_board()
        player = 1
        for ch in history:
            board = play_move(board, int(ch) - 1, player)
            player = -player
        own, mask = position(board, player)
        mo, mm = mirror_bits(own, mask)
        if (mo, mm) < (own, mask):
            own, mask = mo, mm
            w = [w[6 - c] for c in range(7)]
        packed = 0
        for col, val in enumerate(w):
            packed |= encode(val) << (2 * col)
        entries[(own, mask)] = packed
    keys = sorted(entries)
    n = len(keys)
    print(f'{len(entries)} entries from {SOLVED.name} ({bad} malformed), max slots 60000')
    own_arr = np.array([k[0] for k in keys], dtype=np.int64)
    mask_arr = np.array([k[1] for k in keys], dtype=np.int64)
    wdl_arr = np.array([entries[k] for k in keys], dtype=np.int16)
    model = load_model(args.base, strict=False)
    model.midbook_count.copy_(torch.tensor([n], dtype=torch.int64))
    model.midbook_own[:n].copy_(torch.from_numpy(own_arr))
    model.midbook_mask[:n].copy_(torch.from_numpy(mask_arr))
    model.midbook_wdl[:n].copy_(torch.from_numpy(wdl_arr))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'model.pt')
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f'saved {out}/model.pt  params={total} buffers={buffers} '
          f'total={(out/"model.pt").stat().st_size/1e6:.1f}MB')


if __name__ == '__main__':
    main()
