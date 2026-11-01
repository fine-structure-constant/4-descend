"""Local arena: two V10 checkpoints, paired color-swapped openings."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from game import initial_position, legal_columns, make_move
from mcts import Node, backup, evaluate_positions, expand, select, visit_counts
from model import load_checkpoint


def random_opening(rng, plies):
    position = initial_position()
    moves = []
    for _ in range(plies):
        legal = legal_columns(position[2])
        column = int(rng.choice(legal))
        child, terminal_value = make_move(position, column)
        if terminal_value is not None:
            break
        position = child
        moves.append(column)
    return moves, position


class Game:
    __slots__ = ("root", "a_black", "ply", "opening")

    def __init__(self, position, moves, a_black):
        self.root = Node(position)
        self.a_black = a_black
        self.ply = len(moves)
        self.opening = moves


def agent_for(game, player):
    return game.a_black if player == 1 else not game.a_black


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--openings-file", default=None,
                        help="optional JSON list of fixed opening move lists")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.games % 2:
        raise SystemExit("--games must be even for color-swapped pairs")

    device = torch.device(args.device)
    torch.set_num_threads(4)
    model_a, _ = load_checkpoint(args.model_a, device)
    model_b, _ = load_checkpoint(args.model_b, device)
    model_a.to(device).eval()
    model_b.to(device).eval()
    rng = np.random.default_rng(args.seed)

    openings = []
    if args.openings_file:
        for moves in json.loads(Path(args.openings_file).read_text()):
            position = initial_position()
            for move in moves:
                if move not in legal_columns(position[2]):
                    raise SystemExit(f"illegal opening move {move} in {moves}")
                position, terminal = make_move(position, move)
                if terminal is not None:
                    raise SystemExit(f"terminal opening {moves}")
            openings.append((list(moves), position))
        if not openings:
            raise SystemExit("openings file is empty")
    else:
        for _ in range(args.games // 2):
            openings.append(random_opening(rng, args.opening_plies))

    pending = []
    for index in range(args.games):
        moves, position = openings[(index // 2) % len(openings)]
        pending.append(Game(position, moves, a_black=(index % 2 == 0)))

    active = []
    while pending and len(active) < args.concurrency:
        active.append(pending.pop(0))

    results = []
    started = time.monotonic()
    while active:
        for _ in range(args.sims):
            pending_eval = []
            for game in active:
                path = select(game.root)
                leaf = path[-1]
                if leaf.terminal:
                    backup(path, leaf.terminal_value)
                else:
                    pending_eval.append((game, path, leaf))
            group_a = [item for item in pending_eval if agent_for(item[0], item[2].position[3])]
            group_b = [item for item in pending_eval if not agent_for(item[0], item[2].position[3])]
            for model, group in ((model_a, group_a), (model_b, group_b)):
                if not group:
                    continue
                priors, values = evaluate_positions(
                    model, [leaf.position for _, _, leaf in group], device)
                for (game, path, leaf), prior, value in zip(group, priors, values):
                    expand(leaf, prior)
                    backup(path, float(value))

        survivors = []
        for game in active:
            counts = visit_counts(game.root)
            best = counts.max()
            candidates = np.flatnonzero(counts == best)
            action = int(candidates[rng.integers(len(candidates))])
            child = next(c for c in game.root.children if c.move == action)
            if child.terminal:
                winner = 0 if child.terminal_value == 0 else -child.position[3]
                if winner == 0:
                    score = 0.5
                else:
                    score = 1.0 if winner == (1 if game.a_black else -1) else 0.0
                results.append({"winner": winner, "a_black": game.a_black,
                                "score_a": score, "plies": game.ply + 1,
                                "opening": game.opening})
            else:
                game.root = child
                game.ply += 1
                survivors.append(game)
        while pending and len(survivors) < args.concurrency:
            survivors.append(pending.pop(0))
        active = survivors

    wins = sum(1 for r in results if r["score_a"] == 1.0)
    draws = sum(1 for r in results if r["score_a"] == 0.5)
    losses = sum(1 for r in results if r["score_a"] == 0.0)
    score = (wins + 0.5 * draws) / len(results)
    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "games": len(results),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(score, 4),
        "sims": args.sims,
        "opening_plies": args.opening_plies,
        "seconds": round(time.monotonic() - started, 2),
        "results": results,
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(json.dumps({key: summary[key] for key in
                      ("games", "wins", "draws", "losses", "score", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
