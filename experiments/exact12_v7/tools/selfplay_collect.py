"""Self-play v6 model vs itself from the strong protocol openings, with noise.

Collects full game histories for midbook position extraction. Runs in a
process pool; each worker keeps its own model and search module state.
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
from support.game import new_board, play_move, game_result

MODEL = ROOT.parent / 'exact12_v6' / 'runs' / 'refined' / 'model.pt'
OPENINGS = ROOT.parent / 'exact12_v6' / 'runs' / 'strong_openings.json'
OUT = ROOT / 'data' / 'selfplay_games.jsonl'


def play(job):
    model_path, opening, side, seed, out_path = job
    from mcts import search, KNOWN_FLOORS, KNOWN_OUTCOMES
    set_seed(seed)
    model = load_model(model_path)
    KNOWN_FLOORS.clear()
    KNOWN_OUTCOMES.clear()
    board = new_board()
    player = 1
    history = list(opening)
    for action in opening:
        board = play_move(board, action, player)
        player = -player
    while True:
        visits = search(board, player, model, simulations=128, add_noise=True)
        action = int(np.argmax(visits))
        board = play_move(board, action, player)
        history.append(action)
        done, winner = game_result(board)
        if done:
            break
        player = -player
    with open(out_path, 'a') as f:
        f.write(json.dumps({'history': history, 'opening': list(opening),
                            'side': side, 'seed': seed, 'result': 'draw' if not winner else winner}) + '\n')
    return len(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--games-per-side', type=int, default=6)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--model', default=str(MODEL))
    ap.add_argument('--seed', type=int, default=2026103150)
    args = ap.parse_args()
    openings = json.load(open(OPENINGS))
    jobs = []
    seed = args.seed
    for opening in openings:
        for side in (1, -1):
            for _ in range(args.games_per_side):
                jobs.append((args.model, opening, side, seed, args.out))
                seed += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('')
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, length in enumerate(pool.map(play, jobs)):
            total += 1
            if (i + 1) % 20 == 0:
                print(f'{i+1}/{len(jobs)} games done', flush=True)
    print(f'collected {total} games -> {args.out}')


if __name__ == '__main__':
    main()
