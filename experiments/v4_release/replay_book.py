"""Replay recorded games through the exact movebook and report early decisions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from support import movebook
from support.game import new_board, play_move, game_result
from support.tactics import position, popcount


class Holder:
    pass


def load_table(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b'MBK2':
        raise ValueError('bad magic')
    size = int.from_bytes(raw[4:8], 'little')
    base = 8 + 4 * size
    moves = np.frombuffer(raw, dtype=np.uint8, count=size, offset=base)
    scores = np.frombuffer(raw, dtype=np.int8, count=size, offset=base + size)
    data = np.empty((size, 2), dtype=np.uint8)
    data[:, 0] = moves
    data[:, 1] = scores.view(np.uint8)
    holder = Holder()
    holder.movebook_keys = torch.frombuffer(bytearray(raw[8:base]), dtype=torch.uint8)
    holder.movebook_data = torch.from_numpy(data.reshape(-1))
    return holder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--table', default='data/public/movebook12.bin')
    args = parser.parse_args()
    assert movebook.configure(load_table(args.table))
    for report_path in args.reports:
        report = json.loads(Path(report_path).read_text())
        total = optimal = missed = 0
        rows = []
        for record in report['records']:
            board, player = new_board(), 1
            history, side = record['history'], record['side']
            for step, action in enumerate(history):
                occupied = int(np.count_nonzero(board))
                if player == side and occupied <= 12:
                    entry = movebook.lookup(*position(board, player))
                    if entry is not None:
                        mask, score = entry
                        options = [c for c in range(7) if (mask >> c) & 1]
                        total += 1
                        if (mask >> action) & 1:
                            optimal += 1
                        else:
                            missed += 1
                            rows.append((record['game'], step + 1, occupied, action,
                                         options, score))
                board = play_move(board, action, player)
                if game_result(board)[0]:
                    break
                player = -player
        print('== %s: our moves with <=12 discs: %d, optimal %d, suboptimal %d'
              % (Path(report_path).name, total, optimal, missed))
        for row in rows:
            print('   game %s ply %d (%d discs): played %d, optimal %s, exact score %d'
                  % row)


if __name__ == '__main__':
    main()
