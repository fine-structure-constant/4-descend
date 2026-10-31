"""GPU self-play workers: batched PUCT across many concurrent games."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from game import encode, initial_position, result_value
from mcts import (Node, apply_dirichlet, backup, choose_action, evaluate_positions,
                  expand, select, visit_counts)
from model import load_checkpoint


class Game:
    __slots__ = ("root", "records", "rng", "ply")

    def __init__(self, seed):
        self.root = Node(initial_position())
        self.records = []
        self.rng = np.random.default_rng(seed)
        self.ply = 0


def temperature_for(ply):
    if ply < 10:
        return 1.0
    if ply < 24:
        return 0.5
    return 0.15


def finish_game(game, winner):
    examples = []
    for position, policy, ply in game.records:
        z = result_value(winner, position[3])
        examples.append((encode(position), policy, z, ply))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(args.device or f"cuda:{local_rank}")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(args.checkpoint, device)
    model.to(device)
    model.eval()

    rng = np.random.default_rng(args.seed + 1000 * rank + 17)
    started = 0
    active = []
    examples = []
    game_lengths = []
    wins = {1: 0, -1: 0, 0: 0}
    started_at = time.monotonic()

    def start_game():
        nonlocal started
        game = Game(int(rng.integers(1 << 31)))
        started += 1
        return game

    for _ in range(min(args.concurrency, args.games)):
        active.append(start_game())

    while active:
        needs_root = []
        for game in active:
            if game.root.children:
                apply_dirichlet(game.root, game.rng)
            else:
                needs_root.append(game)
        if needs_root:
            priors, _ = evaluate_positions(
                model, [game.root.position for game in needs_root], device)
            for game, prior in zip(needs_root, priors):
                expand(game.root, prior)
                apply_dirichlet(game.root, game.rng)
        for _ in range(args.sims):
            pending = []
            for game in active:
                path = select(game.root)
                leaf = path[-1]
                if leaf.terminal:
                    backup(path, leaf.terminal_value)
                else:
                    pending.append((game, path, leaf))
            if not pending:
                continue
            priors, values = evaluate_positions(
                model, [leaf.position for _, _, leaf in pending], device)
            for (game, path, leaf), prior, value in zip(pending, priors, values):
                expand(leaf, prior)
                backup(path, float(value))

        survivors = []
        for game in active:
            root = game.root
            counts = visit_counts(root)
            total = int(counts.sum())
            if total <= 0:
                raise RuntimeError("root received no visits")
            # Training targets always use raw visit counts (temperature 1).
            game.records.append((root.position, (counts / total).astype(np.float16), game.ply))
            action = choose_action(counts, temperature_for(game.ply), game.rng)
            child = next(c for c in root.children if c.move == action)
            if child.terminal:
                winner = 0 if child.terminal_value == 0 else -child.position[3]
                wins[winner] += 1
                game_lengths.append(game.ply + 1)
                examples.extend(finish_game(game, winner))
            else:
                game.root = child
                game.ply += 1
                survivors.append(game)
        while started < args.games and len(survivors) < args.concurrency:
            survivors.append(start_game())
        active = survivors

    if examples:
        x = np.stack([item[0] for item in examples]).astype(np.uint8)
        policy = np.stack([item[1] for item in examples]).astype(np.float16)
        value = np.array([item[2] for item in examples], dtype=np.float16)
        ply = np.array([item[3] for item in examples], dtype=np.int16)
    else:
        x = np.zeros((0, 6, 6, 7), dtype=np.uint8)
        policy = np.zeros((0, 7), dtype=np.float16)
        value = np.zeros((0,), dtype=np.float16)
        ply = np.zeros((0,), dtype=np.int16)

    shard = out_dir / f"rank_{rank:02d}.npz"
    np.savez(shard, x=x, policy=policy, value=value, ply=ply)
    summary = {
        "rank": rank,
        "world_size": world,
        "device": str(device),
        "games": wins[1] + wins[-1] + wins[0],
        "black_wins": wins[1],
        "white_wins": wins[-1],
        "draws": wins[0],
        "examples": int(len(examples)),
        "checkpoint": str(args.checkpoint),
        "sims": args.sims,
        "concurrency": args.concurrency,
        "seconds": round(time.monotonic() - started_at, 2),
    }
    (out_dir / f"rank_{rank:02d}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
