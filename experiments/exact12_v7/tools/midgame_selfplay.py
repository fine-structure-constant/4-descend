"""Self-play seeded from book-covered 12-14 disc positions to densify coverage.

Each game starts from a solved book position and plays out with the current
model (both sides, with noise). Emits full histories from the empty board so
the extractor can replay them.
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

SOLVED = ROOT / 'data' / 'midbook_solved_v5.tsv'
OUT = ROOT / 'data' / 'midgame_games.jsonl'


def load_seeds():
    seeds = []
    for line in SOLVED.read_text().splitlines():
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        history, rest = parts
        if not 12 <= len(history) <= 14:
            continue
        if any(int(x) == -1000 for x in rest.split()):
            continue
        seeds.append(history)
    return seeds


def play(job):
    model_path, history, seed, out_path = job
    from mcts import search, KNOWN_FLOORS, KNOWN_OUTCOMES
    set_seed(seed)
    model = load_model(model_path)
    KNOWN_FLOORS.clear()
    KNOWN_OUTCOMES.clear()
    board = new_board()
    player = 1
    for ch in history:
        board = play_move(board, int(ch) - 1, player)
        player = -player
    full = list(history)
    while True:
        visits = search(board, player, model, simulations=128, add_noise=True)
        action = int(np.argmax(visits))
        board = play_move(board, action, player)
        full.append(str(action + 1))
        done, winner = game_result(board)
        if done:
            break
        player = -player
    with open(out_path, 'a') as f:
        f.write(json.dumps({'history': [int(ch) - 1 for ch in full], 'seed_history': history,
                            'seed': seed, 'result': 'draw' if not winner else winner}) + '\n')
    return len(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--games', type=int, default=480)
    ap.add_argument('--min-discs', type=int, default=12)
    ap.add_argument('--max-discs', type=int, default=14)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--model', default=str(ROOT / 'runs' / 'midbook_v4' / 'model.pt'))
    ap.add_argument('--seed', type=int, default=2026103407)
    args = ap.parse_args()
    pool_rows = [h for h in load_seeds()
                 if args.min_discs <= len(h) <= args.max_discs]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pool_rows)
    seeds = pool_rows[:args.games]
    print(f'{len(pool_rows)} candidate seeds, using {len(seeds)}')
    jobs = [(args.model, h, int(rng.integers(1 << 31)), args.out) for h in seeds]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, length in enumerate(pool.map(play, jobs)):
            if (i + 1) % 40 == 0:
                print(f'{i+1}/{len(jobs)} games done', flush=True)
    print(f'collected {len(jobs)} games -> {args.out}')


if __name__ == '__main__':
    main()
