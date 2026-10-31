"""Attach public search data and the local exact move table to CNN parameters."""
import argparse
from pathlib import Path
import numpy as np
import torch

from support.movebook import COUNT

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('--out', required=True)
    p.add_argument('--data', default='data/public/book_12ply.dat',
                   help='MIT-licensed 12-ply WDL buffer')
    p.add_argument('--movebook', default='data/public/movebook12.bin',
                   help='exact best-move table for positions with <= 12 discs')
    a = p.parse_args()
    if Path(a.out).exists():
        p.error('Choose a new output path')
    weights = torch.load(a.model, map_location='cpu', weights_only=True)

    raw = Path(a.data).read_bytes()
    if len(raw) != 6943780:
        raise ValueError('Unexpected 12-ply database size')
    weights['opening12'] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()

    table = Path(a.movebook).read_bytes()
    if table[:4] != b'MBK2':
        raise ValueError('Unexpected movebook header')
    size = int.from_bytes(table[4:8], 'little')
    if size != COUNT or len(table) != 8 + 6 * size:
        raise ValueError('Unexpected movebook size')
    base = 8 + 4 * size
    keys = np.frombuffer(table, dtype=np.uint8, count=4 * size, offset=8).copy()
    key_view = keys.view('<u4')
    if int(key_view[0]) != 0 or not np.all(key_view[1:] > key_view[:-1]):
        raise ValueError('movebook keys must be one sorted, unique array '
                         '(run tools/sort_movebook.py first)')
    moves = np.frombuffer(table, dtype=np.uint8, count=size, offset=base)
    scores = np.frombuffer(table, dtype=np.int8, count=size, offset=base + size)
    data = np.empty((size, 2), dtype=np.uint8)
    data[:, 0] = moves
    data[:, 1] = scores.view(np.uint8)
    weights['movebook_keys'] = torch.from_numpy(keys)
    weights['movebook_data'] = torch.from_numpy(data.reshape(-1))

    torch.save(weights, a.out)
    print(a.out)
