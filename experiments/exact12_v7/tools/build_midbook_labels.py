"""Convert midbook solves into exact training rows (state, policy, value, scores).

Each solved position yields two rows (both mirror orientations). Columns with
-999 stay unknown (masked by the q-loss 'known' selector in train.py).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support.game import new_board, play_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solved', default=str(ROOT / 'data' / 'midbook_solved.tsv'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'midbook_labels.npz'))
    ap.add_argument('--min-discs', type=int, default=12)
    ap.add_argument('--max-discs', type=int, default=21)
    args = ap.parse_args()

    states, policies, values, scores, occupied, validation = [], [], [], [], [], []
    skipped = 0
    for line in Path(args.solved).read_text().splitlines():
        parts = line.split('\t')
        if len(parts) != 2:
            skipped += 1
            continue
        history, rest = parts
        nums = rest.split()
        if len(nums) != 7:
            skipped += 1
            continue
        n = len(history)
        if not args.min_discs <= n <= args.max_discs:
            continue
        w = np.array([int(x) for x in nums], dtype=np.int16)
        legal = w != -1000
        if np.any(w[legal] == -999):
            skipped += 1
            continue
        board = new_board()
        player = 1
        for ch in history:
            board = play_move(board, int(ch) - 1, player)
            player = -player
        best = int(w[legal].max())
        opt = ((w == best) & legal).astype(np.float32)
        opt /= opt.sum()
        state = np.stack([board == player, board == -player, board == 0]).astype(np.uint8)
        canonical = min(state.tobytes(), state[:, :, ::-1].tobytes())
        val = int.from_bytes(hashlib.blake2b(canonical, digest_size=8).digest(), 'little') % 10 == 0
        for flipped in (False, True):
            s = state[:, :, ::-1] if flipped else state
            p = opt[::-1] if flipped else opt
            sc = w[::-1] if flipped else w
            states.append(s)
            policies.append(p)
            values.append(float(best))
            scores.append(sc)
            occupied.append(n)
            validation.append(val)
    order = np.argsort(validation).astype(int)
    np.savez_compressed(args.out,
                        states=np.stack([states[i] for i in order]),
                        policies=np.stack([policies[i] for i in order]),
                        values=np.array([values[i] for i in order], np.float32),
                        scores=np.stack([scores[i] for i in order]),
                        occupied=np.array([occupied[i] for i in order]),
                        source=np.full(len(order), 'midbook'),
                        validation=np.array([validation[i] for i in order], bool))
    print(f'{len(states)} rows ({skipped} skipped) -> {args.out}')
    print('by discs:', {d: sum(1 for o in occupied if o == d) for d in sorted(set(occupied))})


if __name__ == '__main__':
    main()
