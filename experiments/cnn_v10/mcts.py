"""Batched-friendly PUCT search shared by self-play, arena and evaluation."""

import math

import numpy as np
import torch

from game import COLS, ROWS, encode_batch, legal_columns, make_move

C_PUCT = 1.5


class Node:
    __slots__ = ("position", "move", "prior", "children", "visit_count",
                 "value_sum", "terminal", "terminal_value")

    def __init__(self, position, move=-1, prior=1.0, terminal=False, terminal_value=0.0):
        self.position = position
        self.move = move
        self.prior = prior
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.terminal = terminal
        self.terminal_value = terminal_value


def evaluate_positions(model, positions, device):
    with torch.inference_mode():
        x = torch.from_numpy(encode_batch(positions)).to(device=device, dtype=torch.float32)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, values = model(x)
        else:
            logits, values = model(x)
        logits = logits.float()
        legal_mask = torch.tensor(
            [[position[2][c] < ROWS for c in range(COLS)] for position in positions],
            dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~legal_mask, -1e9)
        priors = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        return priors, values.float().cpu().numpy().astype(np.float32)


def expand(node, priors):
    for column in legal_columns(node.position[2]):
        child_position, terminal_value = make_move(node.position, column)
        node.children.append(Node(
            child_position, move=column, prior=float(priors[column]),
            terminal=terminal_value is not None,
            terminal_value=float(terminal_value if terminal_value is not None else 0.0)))


def select(root):
    path = [root]
    node = root
    while not node.terminal and node.children:
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best = None
        best_score = -math.inf
        for child in node.children:
            q = -child.value_sum / child.visit_count if child.visit_count else 0.0
            u = C_PUCT * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best = child
        node = best
        path.append(node)
    return path


def backup(path, value):
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def apply_dirichlet(node, rng, epsilon=0.25, alpha=0.3):
    if not node.children:
        return
    noise = rng.dirichlet(np.full(len(node.children), alpha))
    for child, sample in zip(node.children, noise):
        child.prior = (1.0 - epsilon) * child.prior + epsilon * float(sample)


def visit_counts(root, columns=COLS):
    counts = np.zeros(columns, dtype=np.int64)
    for child in root.children:
        counts[child.move] = child.visit_count
    return counts


def choose_action(counts, temperature, rng):
    legal = np.flatnonzero(counts > 0)
    if len(legal) == 0:
        raise ValueError("no visited legal action")
    if temperature <= 1e-3:
        best = counts[legal].max()
        candidates = legal[counts[legal] == best]
        return int(candidates[rng.integers(len(candidates))])
    weights = counts[legal].astype(np.float64) ** (1.0 / temperature)
    weights /= weights.sum()
    return int(rng.choice(legal, p=weights))


def search(model, position, simulations, device, rng=None, add_noise=False):
    root = Node(position)
    priors, _ = evaluate_positions(model, [position], device)
    expand(root, priors[0])
    if add_noise:
        apply_dirichlet(root, rng)
    for _ in range(simulations):
        path = select(root)
        leaf = path[-1]
        if leaf.terminal:
            backup(path, leaf.terminal_value)
        else:
            priors, values = evaluate_positions(model, [leaf.position], device)
            expand(leaf, priors[0])
            backup(path, float(values[0]))
    return root
