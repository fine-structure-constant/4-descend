"""Validate the <=14-disc exact book: hits, symmetry, negamax self-consistency,
and cross-check against the local C++ exact teacher at 12 discs."""
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from support.game import new_board, play_move, game_result, legal_moves
from support.tactics import position, alignment, COLUMNS, possible
from support import pons_book

BINARY = "/tmp/homework4-exact-teacher"


class _Buffer:
    def __init__(self, array):
        self.array = array

    def data_ptr(self):
        return 1

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def load():
    raw = Path("data/public/7x6.book").read_bytes()
    model = type("M", (), {})()
    model.deepbook = _Buffer(np.frombuffer(raw, dtype=np.uint8))
    assert pons_book.configure(model)


def mirror(own, mask):
    mo = mm = 0
    for col in range(7):
        mo |= ((own >> (7 * col)) & 0x7f) << (7 * (6 - col))
        mm |= ((mask >> (7 * col)) & 0x7f) << (7 * (6 - col))
    return mo, mm


def random_position(rng, depth):
    while True:
        board = new_board()
        player = 1
        ok = True
        for _ in range(depth):
            choices = [c for c in legal_moves(board)
                       if not game_result(play_move(board, c, player))[0]]
            if not choices:
                ok = False
                break
            board = play_move(board, rng.choice(choices), player)
            player = -player
        if ok:
            return board, player


def main():
    load()
    rng = random.Random(20261031)
    # symmetry
    bad = 0
    for _ in range(2000):
        board, player = random_position(rng, rng.randint(1, 14))
        own, mask = position(board, player)
        mo, mm = mirror(own, mask)
        if pons_book.key3(own, mask) != pons_book.key3(mo, mm):
            bad += 1
    print("symmetry mismatches", bad)

    hits = {}
    for depth in range(1, 15):
        total = hit = 0
        for _ in range(300):
            board, player = random_position(rng, depth)
            own, mask = position(board, player)
            total += 1
            if pons_book.score(own, mask) is not None:
                hit += 1
        hits[depth] = (hit, total)
    print("book hit rate by disc count", hits)

    # negamax consistency: score = max over non-terminal moves of -child score
    inconsistent = checked = skipped = 0
    for depth in range(1, 14):
        for _ in range(60):
            board, player = random_position(rng, depth)
            own, mask = position(board, player)
            root = pons_book.score(own, mask)
            if root is None:
                continue
            moves = possible(mask)
            if any(alignment(own | (moves & COLUMNS[c])) for c in range(7) if moves & COLUMNS[c]):
                skipped += 1
                continue
            values = []
            ok = True
            for col in range(7):
                bit = moves & COLUMNS[col]
                if not bit:
                    continue
                child = pons_book.score(own ^ mask, mask | bit)
                if child is None:
                    ok = False
                    break
                values.append(-child)
            if not ok:
                inconsistent += 1
                continue
            checked += 1
            if root != max(values):
                inconsistent += 1
                print("mismatch", depth, root, max(values), values)
    print("negamax checked", checked, "skipped(immediate win)", skipped,
          "inconsistent", inconsistent)

    # cross-check WDL with the C++ exact teacher at 12 discs
    jobs = []
    for _ in range(200):
        board, player = random_position(rng, 12)
        own, mask = position(board, player)
        jobs.append((own, mask))
    results = []
    for own, mask in jobs:
        out = subprocess.run([BINARY], input=f"{own} {mask} 9 5 -1\n",
                             text=True, capture_output=True, timeout=30, check=True)
        values = [int(v) for v in out.stdout.split()][:7]
        results.append(values)
    agree = unknown = disagree = 0
    for (own, mask), values in zip(jobs, results):
        if any(v == -2 for v in values):
            unknown += 1
            continue
        root = pons_book.score(own, mask)
        teacher = max(values)
        if root is None:
            unknown += 1
            continue
        if np.sign(root) == teacher:
            agree += 1
        else:
            disagree += 1
            print("WDL disagreement", own, mask, root, values)
    print("teacher WDL agree", agree, "disagree", disagree, "unknown", unknown)


if __name__ == "__main__":
    main()
