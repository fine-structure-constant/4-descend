"""Play paired local games between two checkpoints from identical openings."""
import argparse
import random

import numpy as np
import torch

from mcts import LAST_SEARCH, search
from support.game import new_board, legal_moves, play_move, game_result, next_player
from support.runtime import load_model, set_seed


def choose(model, board, player, simulations):
    visits = np.asarray(search(board, player, model, simulations=simulations))
    legal = legal_moves(board)
    return int(legal[np.argmax(visits[legal])]), LAST_SEARCH.get('cpu_seconds', 0.0)


def play(model_a, model_b, a_player, opening, simulations):
    board, player = new_board(), 1
    for action in opening:
        board = play_move(board, action, player)
        player = next_player(player)
    moves = list(opening)
    slowest = 0.0
    while True:
        model = model_a if player == a_player else model_b
        action, cpu = choose(model, board, player, simulations)
        slowest = max(slowest, cpu)
        moves.append(action)
        board = play_move(board, action, player)
        done, winner = game_result(board)
        if done:
            return winner, moves, slowest
        player = next_player(player)


def random_opening(rng, plies):
    board, player = new_board(), 1
    opening = []
    for _ in range(plies):
        legal = legal_moves(board)
        rng.shuffle(legal)
        for col in legal:
            after = play_move(board, col, player)
            if not game_result(after)[0]:
                opening.append(col)
                board = after
                player = next_player(player)
                break
    return opening


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_a')
    parser.add_argument('model_b')
    parser.add_argument('--pairs', type=int, default=10)
    parser.add_argument('--simulations', type=int, default=32)
    parser.add_argument('--opening-plies', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_seed(args.seed)
    model_a = load_model(args.model_a, 'cpu')
    model_b = load_model(args.model_b, 'cpu')
    rng = random.Random(args.seed)
    score = {'a': 0, 'b': 0, 'draw': 0}
    slowest = 0.0
    records = []
    for pair in range(args.pairs):
        opening = random_opening(rng, args.opening_plies)
        for a_player in (1, -1):
            winner, moves, cpu = play(model_a, model_b, a_player, opening, args.simulations)
            slowest = max(slowest, cpu)
            result = 'draw' if winner == 0 else ('a' if winner == a_player else 'b')
            score[result] += 1
            records.append({'pair': pair, 'a_player': a_player, 'opening': opening,
                            'winner': winner, 'moves': moves, 'slowest_cpu': cpu})
            print('pair %d: A plays %s, result %s, %d plies, slowest %.3fs CPU'
                  % (pair, 'first' if a_player == 1 else 'second', result,
                     len(moves), cpu), flush=True)
    print('A %d  B %d  draws %d  (max move %.3fs CPU)'
          % (score['a'], score['b'], score['draw'], slowest))
    print('score A =', (score['a'] + 0.5 * score['draw']) / (2 * args.pairs))
    if args.output:
        import json
        from pathlib import Path
        Path(args.output).write_text(json.dumps({'a': args.model_a, 'b': args.model_b,
                                                 'score': score, 'max_cpu': slowest,
                                                 'records': records}, indent=2))


if __name__ == '__main__':
    main()
