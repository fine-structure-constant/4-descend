"""Time the fixed Python solver per column on the five loss positions.

Prints seconds to prove each column's exact WDL (or None if > limit),
plus which side each proof lands on. Informs budget/slice tuning.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support.game import new_board, play_move
from support.tactics import position, popcount, alignment, possible, COLUMNS
from support.endgame import ExactSolver

CASES = [
    ('game3-ply14', [6, 5, 1, 1, 6, 4, 3, 5, 6, 6, 5, 5, 0, 2]),
    ('game5-ply14', [3, 0, 5, 6, 0, 4, 0, 5, 3, 3, 3, 3, 1, 0]),
    ('game16-ply15', [5, 4, 3, 6, 2, 3, 2, 5, 2, 2, 5, 4, 6, 4, 4]),
    ('game30-ply17', [2, 2, 2, 4, 6, 3, 2, 4, 2, 2, 3, 4, 4, 3, 4, 0, 5]),
    ('game39-ply16', [6, 4, 0, 1, 0, 6, 2, 1, 4, 1, 1, 4, 2, 2, 0, 0]),
]

LIMIT = 20.0
ORDER = (3, 2, 4, 1, 5, 0, 6)


def build(history):
    board = new_board()
    player = 1
    for col in history:
        board = play_move(board, int(col), player)
        player = -player
    return board, player


def main():
    for label, history in CASES:
        board, player = build(history)
        own, mask = position(board, player)
        count = popcount(mask)
        print(f"{label} (disc={count}, side={player})")
        playable = possible(mask)
        results = {}
        for col in ORDER:
            bit = playable & COLUMNS[col]
            if not bit:
                continue
            if alignment(own | bit):
                results[col] = ('win-now', 0.0)
                continue
            solver = ExactSolver()
            solver.table.clear()
            solver.deadline = time.process_time() + LIMIT
            t0 = time.process_time()
            try:
                value = solver.solve(own ^ mask, mask | bit, count + 1)
                results[col] = (value, time.process_time() - t0)
            except Exception as e:
                results[col] = (f'{type(e).__name__}', time.process_time() - t0)
        print('   ', {c: (v, round(t, 2)) for c, (v, t) in results.items()})
        print()


if __name__ == '__main__':
    main()
