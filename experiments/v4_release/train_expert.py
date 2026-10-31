"""Reproducible teacher distillation followed by MCTS self-play fine-tuning.

Official strong games are optional additional data, not an inference service.
All inference code and weights run offline. Local teacher labels are generated
from this repository's bounded alpha-beta search, not a downloaded model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from model import Network
from support.game import new_board, play_move, game_result, encode
from support.runtime import set_seed, save_model, append_log, sample_batch
from support.tactics import Teacher, position, safe_moves, COLUMNS
from train import compute_loss, self_play


def local_game(job):
    index, seed, depth = job
    rng = np.random.RandomState(seed + index)
    teacher = Teacher(depth, node_limit=100000)
    board, player = new_board(), 1
    history, examples = [], []
    for step in range(42):
        own, mask = position(board, player)
        scores = np.asarray(teacher.scores(own, mask), dtype=np.float64)
        logits = (scores - scores.max()) / 12.
        pi = np.exp(np.maximum(logits, -80))
        pi[scores < -20000] = 0
        pi /= pi.sum()
        state = encode(board, player).numpy()
        value = float(np.tanh(scores.max() / 100))
        history.append((state, pi.astype(np.float32), player, value))
        safe, _ = safe_moves(own, mask)
        legal = [col for col in range(7) if safe & COLUMNS[col]]
        # Diverse openings and occasional safe suboptimal moves expose recovery.
        if legal and step < 10 and rng.rand() < 0.25:
            action = int(rng.choice(legal))
        else:
            sampling_pi = pi ** (0.6 if step < 10 else 2.)
            sampling_pi /= sampling_pi.sum()
            action = int(rng.choice(7, p=sampling_pi))
        board = play_move(board, action, player)
        done, winner = game_result(board)
        if done:
            for state, policy, side, value in history:
                outcome = 0. if winner == 0 else 1. if winner == side else -1.
                examples.append((state, policy, 0.6 * outcome + 0.4 * value))
            break
        player = -player
    return examples


def generate(games, workers, depth, seed, path):
    states, policies, values = [], [], []
    start = time.time()
    jobs = ((i, seed, depth) for i in range(games))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, examples in enumerate(pool.map(local_game, jobs, chunksize=1), 1):
            for state, pi, z in examples:
                states.append(state)
                policies.append(pi)
                values.append(z)
            if index % 20 == 0:
                print(json.dumps({"generated_games": index, "positions": len(states),
                                  "seconds": round(time.time()-start, 1)}), flush=True)
    np.savez_compressed(path, states=np.asarray(states, dtype=np.uint8),
                        policies=np.asarray(policies, dtype=np.float32),
                        values=np.asarray(values, dtype=np.float32))


def strong_examples(path):
    examples = []
    if not Path(path).exists():
        return examples
    for line in Path(path).read_text().splitlines():
        game = json.loads(line)
        for sample in game["samples"]:
            board = np.asarray(sample["board"], dtype=np.int8)
            pi = np.zeros(7, dtype=np.float32)
            pi[sample["action"]] = 1.
            examples.append((encode(board, sample["player"]), pi, float(sample["z"])))
    return examples


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    model = Network()
    save_model(model, out / 'initial.pt')
    dataset = np.load(args.data)
    states = torch.tensor(dataset['states'], dtype=torch.float32)
    policies = torch.tensor(dataset['policies'], dtype=torch.float32)
    values = torch.tensor(dataset['values'], dtype=torch.float32)
    # The final 5% of rows are a development split, not a strength test.
    # Playing strength is evaluated on independent, newly played games.
    n = len(states)
    split = int(n * .95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    start = time.time()
    strong = strong_examples(args.strong)
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(split, (args.batch,))
        s, pi, z = states[indices].clone(), policies[indices].clone(), values[indices].clone()
        if strong and step % 2 == 0:
            ss, pp, zz = sample_batch(strong, args.batch // 4, 'cpu')
            s[:len(ss)], pi[:len(ss)], z[:len(ss)] = ss, pp, zz
        flip = torch.rand(args.batch) < .5
        s[flip] = s[flip].flip(-1)
        pi[flip] = pi[flip].flip(-1)
        loss = compute_loss(model, s, pi, z)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 500 == 0:
            with torch.no_grad():
                model.eval()
                val = compute_loss(model, states[split:], policies[split:], values[split:]).item()
            row = {"stage": "distill", "step": step, "loss": loss.item(),
                   "validation_loss": val, "strong_positions": len(strong),
                   "local_positions": n, "seconds": round(time.time()-start, 1)}
            print(json.dumps(row), flush=True)
            append_log(out / 'log.jsonl', row)
            save_model(model, out / ('distill_%d.pt' % step))
        # Pick up finished API games while the sequential collector continues.
        if step % 250 == 0:
            strong = strong_examples(args.strong)
    save_model(model, out / 'distilled.pt')
    replay = []
    for iteration in range(1, args.iterations + 1):
        model.eval()
        for game in range(args.games_per_iteration):
            replay.extend(self_play(model, args.simulations))
        replay = replay[-30000:]
        strong = strong_examples(args.strong)
        model.train()
        for update in range(100):
            batch = random.choices(replay, k=args.batch * 3 // 4)
            batch += random.choices(strong if strong else replay, k=args.batch - len(batch))
            s, pi, z = sample_batch(batch, args.batch, 'cpu')
            # Replay symmetry augmentation.
            flip = torch.rand(args.batch) < .5
            s[flip], pi[flip] = s[flip].flip(-1), pi[flip].flip(-1)
            loss = compute_loss(model, s, pi, z)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        save_model(model, out / 'model.pt')
        if iteration % 5 == 0:
            save_model(model, out / ('selfplay_%d.pt' % iteration))
        row = {"stage": "selfplay", "iteration": iteration, "positions": len(replay),
               "strong_positions": len(strong), "loss": loss.item(),
               "seconds": round(time.time()-start, 1)}
        print(json.dumps(row), flush=True)
        append_log(out / 'log.jsonl', row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['generate', 'train'])
    parser.add_argument('--data', default='data/local.npz')
    parser.add_argument('--strong', default='data/strong.jsonl')
    parser.add_argument('--out', default='runs/expert_v2')
    parser.add_argument('--games', type=int, default=800)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--games-per-iteration', type=int, default=20)
    parser.add_argument('--simulations', type=int, default=128)
    args = parser.parse_args()
    if args.mode == 'generate':
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        generate(args.games, args.workers, args.depth, args.seed, args.data)
    else:
        train(args)
