"""Validate the <=12-disc move table against the Pons solver and its book."""
import argparse
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from support import movebook, pons_book
from support.game import new_board, legal_moves, play_move, game_result
from support.tactics import position


class Holder:
    pass


def load_movebook(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b'MBK2':
        raise ValueError('bad magic')
    size = int.from_bytes(raw[4:8], 'little')
    if size != movebook.COUNT or len(raw) != 8 + 6 * size:
        raise ValueError('bad size')
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


def load_pons_book(path):
    raw = Path(path).read_bytes()
    holder = Holder()
    holder.deepbook = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    assert pons_book.configure(holder)
    return holder


def mirror(own, mask):
    mo = mm = 0
    for col in range(7):
        mo |= ((own >> (7 * col)) & 0x7f) << (7 * (6 - col))
        mm |= ((mask >> (7 * col)) & 0x7f) << (7 * (6 - col))
    return mo, mm


def solver_scores(sequence, binary):
    out = subprocess.run([binary, '-a', '-b', '/tmp/7x6.book'],
                         input=''.join(sequence) + '\n', text=True,
                         capture_output=True, timeout=120, check=True)
    line = out.stdout.strip().split()
    return [int(v) for v in line[1:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', default='runs/movebook_first12_wdl.sorted.bin')
    parser.add_argument('--games', type=int, default=12)
    parser.add_argument('--solver', default='/tmp/pons_solver')
    args = parser.parse_args()
    holder = load_movebook(args.table)
    assert movebook.configure(holder)
    load_pons_book('../cnn_endgame_v4/data/public/7x6.book')

    rng = random.Random(20261030)
    moves_checked = 0
    book_checked = 0
    solver_checked = 0
    mirror_checked = 0
    for game in range(args.games):
        board, player = new_board(), 1
        sequence = []
        while True:
            occupied = int(np.count_nonzero(board))
            if occupied < 12:
                own, mask = position(board, player)
                entry = movebook.lookup(own, mask)
                assert entry is not None, (game, occupied)
                move_mask, score = entry
                legal = legal_moves(board)
                options = [c for c in legal if (move_mask >> c) & 1]
                assert options, (game, occupied, move_mask, legal)
                mo, mm = mirror(own, mask)
                mirror_mask, mirror_score = movebook.lookup(mo, mm)
                assert mirror_score == score, (game, occupied, score, mirror_score)
                assert mirror_mask == movebook.reverse_mask(move_mask)
                mirror_checked += 1
                exact = pons_book.score(own, mask)
                if exact is not None:
                    assert int(np.sign(exact)) == score, ('score', game, occupied, exact, score)
                    book_checked += 1
                if occupied == 11:
                    scores = solver_scores(sequence, args.solver)
                    legal_scores = [s for s in scores if s > -1000]
                    assert int(np.sign(max(legal_scores))) == score, ('solver score', game, scores, score)
                    for col in options:
                        assert int(np.sign(scores[col])) == score, ('solver move', game, scores, options)
                    solver_checked += 1
                moves_checked += 1
                action = options[0]
            else:
                action = rng.choice(legal_moves(board))
            board = play_move(board, action, player)
            sequence.append(str(action + 1))
            done, winner = game_result(board)
            if done:
                break
            player = -player
    print('checked %d positions (%d book scores, %d solver analyses, %d mirrors)'
          % (moves_checked, book_checked, solver_checked, mirror_checked))


if __name__ == '__main__':
    main()
