"""Paired local matches over the strong protocol openings.

Each opening is played twice (candidate as player 1 and as player 2) so every
position is contested from both sides. Uses the same search for both models;
only weights/book data differ.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support.runtime import load_model, set_seed
from support.game import new_board, play_move, legal_moves, game_result


def play(job):
    candidate, baseline, side, opening, seed = job
    from mcts import search, LAST_SEARCH, KNOWN_FLOORS, KNOWN_OUTCOMES
    set_seed(seed)
    a = load_model(candidate)
    b = load_model(baseline, strict=False)
    KNOWN_FLOORS.clear()
    KNOWN_OUTCOMES.clear()
    board = new_board()
    player = 1
    history = list(opening)
    slowest = 0
    for action in opening:
        board = play_move(board, action, player)
        player = -player
    while True:
        model = a if side == player else b
        visits = search(board, player, model, simulations=128)
        action = int(np.argmax(visits))
        assert action in legal_moves(board)
        slowest = max(slowest, LAST_SEARCH.get('cpu_seconds', 0))
        assert slowest < .95
        board = play_move(board, action, player)
        history.append(action)
        done, winner = game_result(board)
        if done:
            break
        player = -player
    result = 'draw' if not winner else 'win' if winner == side else 'loss'
    return {'side': side, 'opening': opening, 'result': result,
            'history': history, 'max_cpu_seconds': slowest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate')
    ap.add_argument('baseline')
    ap.add_argument('--openings', default=str(ROOT.parent / 'exact12_v6' / 'runs' / 'strong_openings.json'))
    ap.add_argument('--output', default=str(ROOT / 'runs' / 'arena.json'))
    ap.add_argument('--workers', type=int, default=40)
    args = ap.parse_args()
    openings = json.load(open(args.openings))
    jobs = [(args.candidate, args.baseline, side, opening, 2026103190 + i)
            for i, opening in enumerate(openings) for side in [1, -1]]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(play, jobs):
            records.append(record)
    wins = sum(1 for r in records if r['result'] == 'win')
    draws = sum(1 for r in records if r['result'] == 'draw')
    losses = sum(1 for r in records if r['result'] == 'loss')
    print(f'candidate {args.candidate} vs baseline {args.baseline}: '
          f'{wins}W/{draws}D/{losses}L over {len(records)} games')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'candidate': args.candidate, 'baseline': args.baseline,
                   'wins': wins, 'draws': draws, 'losses': losses, 'games': records}, f, indent=1)


if __name__ == '__main__':
    main()
