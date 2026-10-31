"""Per-move exact diagnosis of recorded games, ignoring the fixed opening.

The evaluation harness plays a fixed opening for both sides; the bot only
starts deciding after it.  For each of our real turns we list exact WDL of the
played move against the best move.  Sources: complete <=12-disc WDL (Tromp
<=8 + bitbully 12), partial exact <=14-disc scores (Pons book), and the local
C++ teacher with a generous budget beyond that.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from support import pons_book
from support.deep_opening import configure as configure_opening, lookup as deep_lookup, set_deadline
from support.game import new_board, play_move, game_result, legal_moves
from support.opening import lookup as opening_lookup
from support.tactics import position, alignment, COLUMNS, possible, popcount

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


def load_books():
    raw = Path("data/public/7x6.book").read_bytes()
    model = type("M", (), {})()
    model.deepbook = _Buffer(np.frombuffer(raw, dtype=np.uint8))
    pons_book.configure(model)
    block = Path("data/public/book_12ply.dat").read_bytes()
    model.opening12 = torch.frombuffer(bytearray(block), dtype=torch.uint8)
    configure_opening(model)
    set_deadline(float("inf"))


def wdl_from_openings(own, mask, occupied):
    if occupied == 12:
        return deep_lookup(own, mask, 12)
    if occupied <= 8:
        return opening_lookup(own, mask)
    if occupied < 12:
        return deep_lookup(own, mask, occupied)
    return None


def solve_deep(own, mask, seconds):
    known = pons_book.score(own, mask)
    known = 9 if known is None else (1 if known > 0 else -1 if known < 0 else 0)
    out = subprocess.run([BINARY], input=f"{own} {mask} {known} {seconds} -1\n",
                         text=True, capture_output=True, timeout=seconds + 15, check=True)
    return [int(v) for v in out.stdout.split()][:7]


def move_values(board, player, seconds):
    """Return (values, dtm, complete) with values in {1,0,-1,None}."""
    own, mask = position(board, player)
    occupied = popcount(mask)
    moves = possible(mask)
    values = {}
    dtm = {}
    if occupied <= 14:
        for col in range(7):
            bit = moves & COLUMNS[col]
            if not bit:
                continue
            if alignment(own | bit):
                values[col], dtm[col] = 1, 1
                continue
            child = pons_book.score(own ^ mask, mask | bit)
            if child is not None:
                values[col] = 1 if child > 0 else -1 if child < 0 else 0
                dtm[col] = -child
    if occupied <= 11:
        for col in range(7):
            bit = moves & COLUMNS[col]
            if not bit or col in values:
                continue
            if alignment(own | bit):
                values[col] = 1
                continue
            value = wdl_from_openings(own ^ mask, mask | bit, occupied + 1)
            if value is not None:
                values[col] = -value
    if all(values.get(col) is not None for col in legal_moves(board)):
        return values, dtm, True
    raw = solve_deep(own, mask, seconds)
    for col in legal_moves(board):
        if values.get(col) is None:
            values[col] = raw[col] if raw[col] in (-1, 0, 1) else None
    return values, dtm, False


def classify(value):
    return {1: "WIN", 0: "DRAW", -1: "LOSS", None: "?"}[value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", default="runs/final_v3_strong.json")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--all", action="store_true", help="also print wins/draws")
    args = parser.parse_args()
    load_books()
    report = json.loads(Path(args.report).read_text())
    for record in report["records"]:
        if not args.all and record["result"] == "win":
            continue
        history, side = record["history"], record["side"]
        opening = record["opening"]
        board = new_board()
        player = 1
        for action in opening:
            board = play_move(board, action, player)
            player = -player
        opening_values, _, _ = move_values(board, player, args.seconds)
        opening_value = max((v for v in opening_values.values() if v is not None), default=None)
        if opening_value is not None and player != side:
            opening_value = -opening_value
        print(f"== game {record['game']} side {side} opening {opening} "
              f"result {record['result']} | our post-opening value: "
              f"{classify(opening_value)} ==")
        for step in range(len(opening), len(history)):
            action = history[step]
            if player == side:
                values, dtm, complete = move_values(board, player, args.seconds)
                known = {c: v for c, v in values.items() if v is not None}
                best = max(known.values()) if known else None
                played = known.get(action)
                flag = ""
                if played is not None and best is not None and played < best:
                    flag = "  <-- WDL BLUNDER"
                elif played is not None and best is not None and played == best and dtm.get(action) is not None:
                    slower = [c for c, v in known.items() if v == best and dtm.get(c, -99) < dtm[action]]
                    if slower:
                        flag = "  (slower win / faster loss)"
                print(f"  ply {step + 1:2d} disc {popcount(position(board, player)[1]):2d} "
                      f"played {action} -> {classify(played)}  best {classify(best)}  "
                      f"all { {c: classify(v) for c, v in sorted(known.items())} }{flag}")
            board = play_move(board, action, player)
            if game_result(board)[0]:
                break
            player = -player
        print()


if __name__ == "__main__":
    main()
