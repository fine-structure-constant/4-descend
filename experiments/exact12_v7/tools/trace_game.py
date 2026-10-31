"""Trace a game move-by-move: was each 13-21 disc position in the midbook,
and did the chosen move match the book's optimal column?"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support.runtime import load_model
from support.midbook import configure, lookup
from support.game import new_board, play_move, game_result
from support.tactics import position, popcount

MODEL = ROOT / 'runs' / 'midbook_base' / 'model.pt'
BOOK = '/home/stu2026/stu2400011444/homework4/experiments/cnn_endgame_v4/data/public/7x6.book'

GAMES = {
    'arena-loss1': [3, 0, 5, 6, 0, 4, 0, 5, 3, 3, 3, 3, 1, 1, 5, 4, 4, 5, 6, 0, 1, 5, 6, 1, 6, 6, 6, 4, 4, 3, 1, 4, 5, 1, 0, 2, 0, 2],
    'arena-loss2': [6, 4, 0, 1, 0, 6, 2, 1, 4, 4, 1, 0, 1, 1, 4, 2, 2, 2, 3, 2, 4, 3, 6, 6, 3, 2, 3, 6, 3, 3, 4, 1, 6, 5, 5, 0, 0, 0],
}


def cpp_wdl(history, seconds=30):
    h = ''.join(str(c + 1) for c in history)
    out = subprocess.run([str(ROOT / 'tools' / 'score_one'), BOOK, str(seconds), h],
                         capture_output=True, text=True, timeout=seconds * 4 + 60)
    line = out.stdout.strip()
    if line in ('BAD', 'TIMEOUT', ''):
        return None
    nums = line.split()
    if len(nums) != 7:
        return None
    return [None if int(x) == -999 else int(x) for x in nums]


def main():
    model = load_model(str(MODEL))
    assert configure(model)
    for label, history in GAMES.items():
        print(f'== {label} (player1 = our side)')
        board = new_board()
        player = 1
        for ply, col in enumerate(history):
            board = play_move(board, col, player)
            player = -player
            if not 13 <= len(history[:ply + 1]) <= 21:
                continue
            if game_result(board)[0]:
                break
            own, mask = position(board, player)
            mid = lookup(own, mask)
            if mid is None:
                cpp = cpp_wdl(history[:ply + 1])
                print(f'  ply{ply+1} (disc={ply+1}, side={player}) move={col} NOT IN BOOK, cpp={cpp}')
            else:
                opt = [c for c in range(7) if mid[c] == max(v for v in mid if v is not None)]
                flag = '' if col in opt else f'  <-- SUBOPTIMAL, optimal={opt}, wdl={mid}'
                print(f'  ply{ply+1} (disc={ply+1}, side={player}) move={col} book-opt={opt}{flag}')
        print()


if __name__ == '__main__':
    main()
