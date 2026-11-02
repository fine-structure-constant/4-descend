"""Audit losing-move distances with independent array rules and a C++ teacher."""
import argparse
import functools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
MATE = 10000
LINES = [tuple((row + k * dr) * 7 + col + k * dc for k in range(4))
         for row in range(6) for col in range(7)
         for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
         if 0 <= row + 3 * dr < 6 and 0 <= col + 3 * dc < 7]


def won(board, player):
    return any(board[a] == player and board[b] == player
               and board[c] == player and board[d] == player for a, b, c, d in LINES)


def drop(board, col, player):
    out = list(board)
    for row in range(5, -1, -1):
        if out[7 * row + col] == 0:
            out[7 * row + col] = player
            return tuple(out)
    raise AssertionError("Full column")


@functools.lru_cache(maxsize=300000)
def exact(board, player):
    remaining = board.count(0)
    if won(board, -player):
        return -(MATE + remaining + 1)
    if not remaining:
        return 0
    children = [drop(board, c, player) for c in range(7) if board[c] == 0]
    if any(won(child, player) for child in children):
        return MATE + remaining
    return max(-exact(child, -player) for child in children)


def replay(history):
    board, player = (0,) * 42, 1
    for col in history:
        assert not won(board, -player)
        board = drop(board, col, player)
        player = -player
    return board, player


def observation(board, player):
    return {"game": "connect4", "player": player,
            "board": [list(board[r * 7:(r + 1) * 7]) for r in range(6)],
            "legal_actions": [c for c in range(7) if board[c] == 0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/resistance_validation.json")
    args = parser.parse_args()
    stage = tempfile.TemporaryDirectory(prefix="c4-resistance-")
    if args.release:
        with zipfile.ZipFile(args.release / "code.zip") as archive:
            archive.extractall(stage.name)
        sys.path.insert(0, stage.name)
    else:
        sys.path.insert(0, str(ROOT))
    from support import native
    from support.bot import Bot
    from support.tactics import position as bit_position, safe_moves, ORDER
    def position(board, player):
        return bit_position(np.asarray(board, dtype=np.int8), player)
    assert native.warmup(), native.LAST_ERROR
    report = {"source": str(args.release or ROOT), "independent_distances": [],
              "dispatch": [], "cpp_distances": [], "tiny_budgets": []}
    rng = random.Random(20261102)
    improved = 0
    # Fresh positions, generated without either bitboard solver or a book.
    for attempt in range(10000):
        board, player, history = (0,) * 42, 1, []
        target = rng.choice((28, 30, 32, 34))
        for _ in range(target):
            options = [c for c in range(7) if board[c] == 0 and not won(drop(board, c, player), player)]
            if not options:
                break
            col = rng.choice(options)
            history.append(col)
            board = drop(board, col, player)
            player = -player
        if len(history) != target:
            continue
        children = {c: drop(board, c, player) for c in range(7) if board[c] == 0}
        if any(won(child, player) for child in children.values()):
            continue
        defences = [c for c, child in children.items() if not any(
            won(drop(child, reply, -player), -player)
            for reply in range(7) if child[reply] == 0)]
        if len(defences) < 2:
            continue
        values = {}
        for c, child in children.items():
            values[c] = -exact(child, -player)
            if values[c] > -MATE:
                break
        if len(values) != len(children):
            continue
        best = max(values.values())
        if best > -MATE or len({values[c] for c in defences}) < 2:
            continue
        old = next(c for c in ORDER if c in values)
        for mirrored in (False, True):
            test = tuple(cell for row in range(6) for cell in
                         (board[row * 7:(row + 1) * 7][::-1] if mirrored
                          else board[row * 7:(row + 1) * 7]))
            us, occupied = position(observation(test, player)["board"], player)
            start = time.monotonic()
            action, info = native.resist(us, occupied, 200)
            wall = time.monotonic() - start
            expected_col = 6 - action if mirrored else action
            assert values[expected_col] == best, (history, mirrored, action, values, info)
            plies = 43 - len(history) - (abs(best) - MATE)
            assert info["resistance_complete"], (history, info)
            assert info["resistance_plies"] == plies, (history, best, info)
            assert wall < .3, wall
            report["independent_distances"].append({
                "history": history, "mirrored": mirrored, "action": action,
                "old_action": old, "scores": values, "wall_seconds": wall, **info})
        if len(report["independent_distances"]) % 16 == 0:
            print("Completed distance checks:", len(report["independent_distances"]), flush=True)
        improved += values[old] < best
        if len(report["independent_distances"]) >= 64:
            break
    assert len(report["independent_distances"]) == 64
    assert improved > 0
    report["better_than_fixed_order"] = improved
    print("Independent distance positions: 64; improved:", improved, flush=True)

    replay_data = json.loads((ROOT / "runs/match_627ee5d2_analysis/replay.json").read_text())
    history = [f["action"] for f in replay_data["frames"] if f.get("action") is not None]
    bot = Bot({}, books_path=PROJECT / "upload-v9/model.pt")
    board, player = replay(history[:15])
    obs = observation(board, player)
    own, mask = position(obs["board"], player)
    legal = obs["legal_actions"]
    wdl = [-1 if c in legal else 9 for c in range(7)]
    # Every losing fast path must reach the shared resistance selector.
    for phase in ("ponder_exact", "exact_scorebook", "native_exact", "exact_proof"):
        bot.ponder_proofs.clear()
        if phase == "ponder_exact":
            bot.ponder_proofs[own, mask] = (legal[0], -1)
        with patch("support.bot.midbook.lookup", return_value=(
                [-1 if c in legal else None for c in range(7)] if phase == "exact_scorebook" else None)), \
             patch("support.bot.native.analyze", return_value=(legal[0], -1, phase == "native_exact", wdl)), \
             patch("support.bot.native.resist", return_value=(legal[-1], {"resistance_test": True})) as select:
            action = bot.act(obs)
            assert select.call_count == 1 and action == legal[-1], (phase, bot.last_info)
            assert bot.last_info["phase"] == phase, bot.last_info
            report["dispatch"].append(phase)
    bot.ponder_proofs.clear()
    early_board, early_player = replay(history[:3])
    with patch("support.bot.native.resist", wraps=native.resist) as select:
        bot.act(observation(early_board, early_player))
        assert select.call_count == 1 and bot.last_info["phase"] == "movebook12"
        report["dispatch"].append("movebook12")
    # A double threat: every move loses next turn, but blocking one leaves
    # fewer winning replies for an imperfect opponent than playing centre.
    double, side = replay([0, 1, 6, 5, 0, 1, 6, 5, 0, 3, 6])
    forced_obs = observation(double, side)
    own, mask = position(forced_obs["board"], -1)
    assert safe_moves(own, mask)[1] == -1
    with patch("support.bot.movebook.lookup", return_value=None), \
         patch("support.bot.native.resist", wraps=native.resist) as select:
        action = bot.act(forced_obs)
        assert select.call_count == 1 and action in (0, 6), (action, bot.last_info)
        report["dispatch"].append("forced_loss")
    bot.current_bits = (own, mask)
    for outcome in (0, 1, None):
        with patch("support.bot.native.resist", side_effect=AssertionError("Unexpected resistance")):
            bot._done(0, "proof_test", time.monotonic(), True, outcome)
    report["nonlosing_paths_unchanged"] = True
    bot.known_floors.clear()
    for depth in (3, 7, 13):
        test, side = replay(history[:depth])
        own, mask = position(observation(test, side)["board"], side)
        action, info = native.resist(own, mask, 1)
        assert action in observation(test, side)["legal_actions"]
        if not info["resistance_complete"]:
            assert info["resistance_plies"] is None
        report["tiny_budgets"].append(info)

    # The existing independent C++ teacher computes full distance scores.
    teacher = Path(stage.name) / "score-distance"
    subprocess.run(["g++", "-O3", "-std=c++17", "-I/tmp/mbwork",
                    str(ROOT / "tools/score_distance.cpp"), "/tmp/mbwork/Solver.cpp",
                    "-o", str(teacher)], check=True)
    for depth in (13, 15, 17, 19):
        test, side = replay(history[:depth])
        action = bot.act(observation(test, side))
        result = subprocess.run([str(teacher), "/tmp/7x6.book", "10",
                                 "".join(str(c + 1) for c in history[:depth])],
                                capture_output=True, text=True, check=True, timeout=12)
        scores = list(map(int, result.stdout.split()))
        assert len(scores) == 7 and scores[action] == max(scores), (depth, scores, bot.last_info)
        old = history[depth]
        remaining = 42 - depth
        old_plies = 2 * (remaining // 2 + 1 + scores[old])
        best_plies = 2 * (remaining // 2 + 1 + scores[action])
        if bot.last_info["resistance_complete"]:
            assert bot.last_info["resistance_plies"] == best_plies
        else:
            assert bot.last_info["resistance_plies"] is None
        report["cpp_distances"].append({"step": depth + 1, "old_action": old,
            "new_action": action, "scores": scores, "old_plies": old_plies,
            "new_plies": best_plies, **bot.last_info})
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": True, "report": str(args.output),
                      "cpp_distances": report["cpp_distances"]}), flush=True)


if __name__ == "__main__":
    main()
