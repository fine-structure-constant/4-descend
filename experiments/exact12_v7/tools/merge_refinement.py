"""Merge new midbook label rows into the refinement training dataset.

Dedup: drop label rows whose exact state already exists in the base dataset,
and internal duplicate states within the labels (symmetric positions emit
identical rows for both orientations).
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def key2d(states):
    return states.reshape(states.shape[0], -1)


def void_view(rows):
    return np.ascontiguousarray(rows).view(
        np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=str(ROOT / 'data' / 'refinement_v8.npz'))
    ap.add_argument('--labels', default=str(ROOT / 'data' / 'midbook_labels4.npz'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'refinement_v9.npz'))
    args = ap.parse_args()
    base = np.load(args.base)
    labels = np.load(args.labels)

    base_rows = key2d(base['states'])
    lab_rows = key2d(labels['states'])
    base_unique = np.unique(base_rows, axis=0)
    lab_unique, lab_first = np.unique(lab_rows, axis=0, return_index=True)
    base_view = void_view(base_unique)
    lab_view = void_view(lab_unique)
    pos = np.searchsorted(base_view, lab_view)
    pos = np.minimum(pos, len(base_view) - 1)
    fresh = lab_view != base_view[pos]
    keep = lab_first[fresh]
    print(f'label rows: {len(lab_rows)}, unique: {len(lab_unique)}, '
          f'new (not in base): {fresh.sum()}, internal dup: {len(lab_rows) - len(lab_unique)}')

    n = len(base['states']) + len(keep)
    out = {}
    for key in base.files:
        arr = base[key]
        shape = (n,) + arr.shape[1:]
        merged = np.empty(shape, dtype=arr.dtype)
        merged[:len(arr)] = arr
        merged[len(arr):] = labels[key][keep]
        out[key] = merged
    np.savez_compressed(args.out, **out)
    train = int((~out['validation']).sum())
    print(f'merged -> {args.out}  train/val: {train} {int(out["validation"].sum())}')


if __name__ == '__main__':
    main()
