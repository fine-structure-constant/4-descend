"""Sort an MBK2 move table by canonical key into a single searchable array.

The builder emits keys grouped by disc count, and canonical keys of different
depths interleave (a 3-disc key can be smaller than a 2-disc key), so the
concatenated array must be sorted before binary search.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    raw = Path(args.table).read_bytes()
    if raw[:4] != b'MBK2':
        raise ValueError('bad magic')
    size = int.from_bytes(raw[4:8], 'little')
    if len(raw) != 8 + 6 * size:
        raise ValueError('bad size')
    keys = np.frombuffer(raw, dtype='<u4', count=size, offset=8)
    base = 8 + 4 * size
    moves = np.frombuffer(raw, dtype=np.uint8, count=size, offset=base)
    scores = np.frombuffer(raw, dtype=np.int8, count=size, offset=base + size)
    print('input strictly increasing:', bool(np.all(keys[1:] > keys[:-1])))

    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    moves = moves[order]
    scores = scores[order]
    if not np.all(keys[1:] > keys[:-1]):
        raise ValueError('duplicate keys after sorting')
    print('output strictly increasing: True (%d records)' % size)

    with open(args.out, 'wb') as file:
        file.write(b'MBK2')
        file.write(np.uint32(size).tobytes())
        file.write(keys.tobytes())
        file.write(moves.tobytes())
        file.write(scores.tobytes())
    print('wrote', args.out)


if __name__ == '__main__':
    main()
