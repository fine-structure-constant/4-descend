"""Compare Python ExactSolver against the C++ reference solver on the same positions.

Both engines return per-column WDL {-1,0,1} for the side to move; None/unknown
columns are skipped. A definitive disagreement is a soundness bug in one of them.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from support.game import new_board, play_move, game_result, legal_moves
from support.tactics import position, popcount, alignment
from support.endgame import ExactSolver
from support.runtime import load_model
from support.pons_book import configure as configure_scores

BOOK = '/home/stu2026/stu2400011444/homework4/experiments/cnn_endgame_v4/data/public/7x6.book'
MODEL = ROOT.parent / 'exact12_v6' / 'runs' / 'refined' / 'model.pt'


def build(history):
    board = new_board()
    player = 1
    for col in history:
        board = play_move(board, int(col), player)
        player = -player
    return board, player


def cpp_scores(history, seconds, timeout):
    h = ''.join(str(int(c) + 1) for c in history) or '-'
    out = subprocess.run([str(ROOT / 'tools' / 'score_one'), BOOK, str(seconds), h],
                         capture_output=True, text=True, timeout=timeout)
    line = [l for l in out.stdout.splitlines() if l.strip()]
    if not line or line[0].startswith(('BAD', 'TIMEOUT')):
        return None, (line[0] if line else '') + ' ' + out.stderr[-200:]
    nums = line[0].split()
    if len(nums) != 7:
        return None, line[0]
    return [None if int(x) == -999 else int(x) for x in nums], ''


def py_scores(board, player, seconds, solver):
    own, mask = position(board, player)
    solver.table.clear()
    scores = solver.score_moves(own, mask, seconds=seconds)
    wdl = []
    for c in range(7):
        s = scores[c]
        if s is None:
            wdl.append(None)
        else:
            wdl.append(1 if s > 0 else -1 if s < 0 else 0)
    return wdl, own, mask


def compare(history, label, cpp_seconds, py_seconds, solver):
    board, player = build(history)
    n = len(history)
    cpp, cpp_err = cpp_scores(history, cpp_seconds, cpp_seconds * 4 + 90)
    py, own, mask = py_scores(board, player, py_seconds, solver)
    out = {'label': label, 'discs': n, 'cpp': cpp, 'py': py}
    if cpp is None:
        out['status'] = 'cpp_failed'
        out['detail'] = cpp_err
        return out
    mism = []
    for c in range(7):
        if cpp[c] is not None and py[c] is not None and cpp[c] != py[c]:
            mism.append((c, cpp[c], py[c]))
    out['mismatches'] = mism
    out['status'] = 'mismatch' if mism else 'ok'
    return out


def random_history(rng, discs):
    while True:
        board = new_board()
        player = 1
        history = []
        for _ in range(discs):
            legal = legal_moves(board)
            col = int(rng.choice(legal))
            history.append(col)
            board = play_move(board, col, player)
            player = -player
            done, winner = game_result(board)
            if done:
                break
        if len(history) == discs:
            return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=0, help='number of random positions to add')
    ap.add_argument('--min-discs', type=int, default=13)
    ap.add_argument('--max-discs', type=int, default=19)
    ap.add_argument('--cpp-seconds', type=float, default=25.0)
    ap.add_argument('--py-seconds', type=float, default=25.0)
    ap.add_argument('--out', default=str(ROOT / 'runs' / 'crosscheck.jsonl'))
    ap.add_argument('--loss-games', action='store_true', help='include the five v6 loss positions')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    model = load_model(str(MODEL))
    assert configure_scores(model)
    solver = ExactSolver()

    jobs = []
    if a.loss_games:
        games = {
            'game3-ply14': [6, 5, 1, 1, 6, 4, 3, 5, 6, 6, 5, 5, 0, 2],
            'game5-ply14': [3, 0, 5, 6, 0, 4, 0, 5, 3, 3, 3, 3, 1, 0],
            'game16-ply15': [5, 4, 3, 6, 2, 3, 2, 5, 2, 2, 5, 4, 6, 4, 4],
            'game30-ply17': [2, 2, 2, 4, 6, 3, 2, 4, 2, 2, 3, 4, 4, 3, 4, 0, 5],
            'game39-ply16': [6, 4, 0, 1, 0, 6, 2, 1, 4, 1, 1, 4, 2, 2, 0, 0],
        }
        for label, h in games.items():
            jobs.append((h, label))
    rng = np.random.default_rng(a.seed)
    for i in range(a.batch):
        discs = int(rng.integers(a.min_discs, a.max_discs + 1))
        jobs.append((random_history(rng, discs), f'random-{i}-d{discs}'))

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.time()
    with open(out_path, 'a') as f:
        for h, label in jobs:
            r = compare(h, label, a.cpp_seconds, a.py_seconds, solver)
            f.write(json.dumps(r) + '\n')
            f.flush()
            results.append(r)
            bad = [x for x in results if x['status'] in ('mismatch', 'cpp_failed')]
            print(f"{label}: {r['status']} disc={r['discs']} "
                  f"({len(bad)} bad / {len(results)} done, {time.time()-start:.0f}s)", flush=True)
    mism = [x for x in results if x['status'] == 'mismatch']
    print(f"done: {len(results)} positions, {len(mism)} mismatches")
    for m in mism:
        print(m['label'], m['mismatches'])


if __name__ == '__main__':
    main()
