"""Run the v7 search on the five v6 loss positions and report chosen moves.

Uses the v6 refined model as a stand-in network; isolates search-side changes.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from mcts import search, LAST_SEARCH
from support.game import new_board, play_move
from support.runtime import load_model, set_seed
from support.movebook import configure as configure_movebook
from support.pons_book import configure as configure_scores

MODEL = ROOT / 'runs' / 'midbook_v5' / 'model.pt'

CASES = [
    ('game3-ply14', [6, 5, 1, 1, 6, 4, 3, 5, 6, 6, 5, 5, 0, 2], 3, {2}),
    ('game5-ply14', [3, 0, 5, 6, 0, 4, 0, 5, 3, 3, 3, 3, 1, 0], 5, {1}),
    ('game16-ply15', [5, 4, 3, 6, 2, 3, 2, 5, 2, 2, 5, 4, 6, 4, 4], 1, {0}),
    ('game30-ply17', [2, 2, 2, 4, 6, 3, 2, 4, 2, 2, 3, 4, 4, 3, 4, 0, 5], 5, {6}),
    ('game39-ply16', [6, 4, 0, 1, 0, 6, 2, 1, 4, 1, 1, 4, 2, 2, 0, 0], 4, {2}),
]


def build(history):
    board = new_board()
    player = 1
    for col in history:
        board = play_move(board, int(col), player)
        player = -player
    return board, player


def main():
    set_seed(2026103107)
    model = load_model(str(MODEL))
    assert configure_movebook(model) and configure_scores(model)
    for label, history, played, optimal in CASES:
        board, player = build(history)
        counts = search(board, player, model, simulations=128)
        chosen = int(np.argmax(counts))
        detail = dict(LAST_SEARCH)
        print(f"{label}: chosen={chosen} played={played} optimal={sorted(optimal)} "
              f"counts={[int(c) for c in counts]}")
        for key in ('allowed', 'root_known', 'root_minimum', 'outcome_bounds',
                    'guidance_depth', 'cpu_seconds', 'phase', 'root_lower'):
            if key in detail:
                print(f"   {key}: {detail[key]}")
        print()


if __name__ == '__main__':
    main()
