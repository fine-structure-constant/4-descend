"""Self-checks for the V10 rules, encoding, search and folded deployment net."""

import numpy as np
import torch

from deploy.net import DeployNet
from game import (COLS, ROWS, encode, flip_columns, from_board, initial_position,
                  legal_columns, make_move)
from mcts import search, visit_counts
from model import Net, to_deploy

ROWS_BOARD = 6


def top_down_board(position):
    own, opp, _, player = position
    board = [[0] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            bit = 1 << (r * 7 + c)
            if own & bit:
                board[5 - r][c] = player
            elif opp & bit:
                board[5 - r][c] = -player
    return board


def numpy_winner(board):
    for player in (1, -1):
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + i * dr, c + i * dc) for i in range(4)]
                    if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                        if all(board[rr][cc] == player for rr, cc in cells):
                            return player
    if all(board[r][c] != 0 for r in range(ROWS) for c in range(COLS)):
        return 0
    return None


def check_rules(rng, games=100):
    for _ in range(games):
        position = initial_position()
        board = top_down_board(position)
        for _ in range(42):
            winner = numpy_winner(board)
            if winner is not None:
                break
            column = int(rng.choice(legal_columns(position[2])))
            position, terminal = make_move(position, column)
            board = top_down_board(position)
            expected = numpy_winner(board)
            if terminal is None and expected is not None:
                raise AssertionError("missed terminal position")
            if terminal is not None:
                if expected == 0:
                    if terminal != 0:
                        raise AssertionError("draw value disagrees")
                elif expected != -position[3] or terminal != -1:
                    raise AssertionError("terminal value disagrees")
                break
        if numpy_winner(board) is not None and terminal is None:
            raise AssertionError("game continued after terminal")


def check_encoding(rng, samples=50):
    for _ in range(samples):
        position = initial_position()
        for _ in range(int(rng.integers(0, 42))):
            legal = legal_columns(position[2])
            if not legal:
                break
            position, terminal = make_move(position, int(rng.choice(legal)))
            if terminal is not None:
                break
        recovered = from_board(top_down_board(position), position[3])
        if recovered != position:
            raise AssertionError("board round-trip failed")
        flipped = flip_columns(position)
        a = encode(position)
        b = encode(flipped)
        if not np.array_equal(a[:4], b[:4][:, :, ::-1]):
            raise AssertionError("encoding does not mirror")
        if not np.array_equal(a[4], b[4][:, ::-1]):
            raise AssertionError("favoured plane is not symmetric")
        if not np.array_equal(a[5], b[5]):
            raise AssertionError("side plane changed under mirror")


def check_search(model, rng, simulations=16):
    position = initial_position()
    root = search(model, position, simulations, torch.device("cpu"), rng=rng)
    counts = visit_counts(root)
    if counts.sum() != simulations:
        raise AssertionError("visit counts do not sum to simulations")
    legal = set(legal_columns(position[2]))
    for column in range(7):
        if column not in legal and counts[column] != 0:
            raise AssertionError("search visited an illegal column")


def check_fold():
    model = Net(64, 2).eval()
    deploy = to_deploy(model)
    x = torch.rand(16, 6, 6, 7)
    with torch.no_grad():
        policy_a, value_a = model(x)
        policy_b, value_b = deploy(x)
    if (policy_a - policy_b).abs().max() > 1e-5:
        raise AssertionError("policy differs after folding")
    if (value_a - value_b).abs().max() > 1e-5:
        raise AssertionError("value differs after folding")


def main():
    rng = np.random.default_rng(20261101)
    check_rules(rng)
    check_encoding(rng)
    model = Net(64, 2).eval()
    check_search(model, rng)
    check_fold()
    print("V10 checks passed")


if __name__ == "__main__":
    main()
