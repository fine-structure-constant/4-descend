"""Validate the exported baseline against independent rules and exact oracles."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def winner(board, player):
    for r in range(6):
        for c in range(7):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7:
                    if all(board[(r + k * dr) * 7 + c + k * dc] == player for k in range(4)):
                        return True
    return False


def drop(board, col, player):
    out = list(board)
    for row in range(5, -1, -1):
        if out[row * 7 + col] == 0:
            out[row * 7 + col] = player
            return tuple(out)
    raise AssertionError("Illegal move")


def replay(history):
    board, player = (0,) * 42, 1
    for col in history:
        assert not winner(board, -player)
        board = drop(board, col, player)
        player = -player
    return board, player


@functools.lru_cache(maxsize=200000)
def exact(board, player):
    if winner(board, -player):
        return -1
    legal = [c for c in range(7) if board[c] == 0]
    if not legal:
        return 0
    best = -1
    for c in legal:
        best = max(best, -exact(drop(board, c, player), -player))
        if best == 1:
            break
    return best


def observation(board, player):
    return {"game": "connect4", "board": [list(board[r * 7:(r + 1) * 7]) for r in range(6)],
            "player": player, "legal_actions": [c for c in range(7) if board[c] == 0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=str(PROJECT / "upload-v9"))
    parser.add_argument("--disable-native", action="store_true")
    args = parser.parse_args()
    release = Path(args.release).resolve()
    report = {"release": str(release), "learned_parameters": 0, "checks": {}}
    with tempfile.TemporaryDirectory(prefix="v9-validation-") as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(release / "code.zip") as archive:
            assert all(Path(n).suffix in {".py", ".md", ".txt"} for n in archive.namelist())
            assert "model.py" not in archive.namelist() and "mcts.py" not in archive.namelist()
            archive.extractall(stage)
        (stage / "models").mkdir()
        os.symlink(release / "model.pt", stage / "models/model.pt")
        sys.path.insert(0, str(stage))
        # Any accidental neural forward call must fail the test.
        def forbidden_forward(*args, **kwargs):
            raise AssertionError("Neural network inference is forbidden in v9")
        torch.nn.Module._call_impl = forbidden_forward
        from strategy import Bot
        from support import native, movebook
        from support.tactics import position
        if args.disable_native:
            native.warmup = lambda: False
            native.analyze = lambda *args, **kwargs: None
        started = time.monotonic()
        bot = Bot({"seed": 20261031})
        report["init_seconds"] = time.monotonic() - started
        report["initial_ponder"] = bot.initial_ponder
        assert bot.native_ready == (not args.disable_native)
        report["native_disabled_for_test"] = args.disable_native
        binary = stage / "native-inspect"
        binary.write_bytes(native._decode_payload())
        desc = subprocess.check_output(["file", str(binary)], text=True).strip()
        dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
        assert "statically linked" in desc and "no dynamic section" in dynamic
        report["native"] = {"file": desc, "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
        data = torch.load(release / "model.pt", map_location="cpu", weights_only=True)
        from support.bot import BOOK_NAMES
        assert set(data) == BOOK_NAMES
        report["book_tensors"] = sorted(data)
        arena = json.loads((PROJECT / "experiments/exact12_v7/runs/arena_v5nn2_fixed_vs_v6.json").read_text())
        games = arena["games"]
        max_wall = 0.
        # Independent array-board exhaustive minimax, with no native/book oracle.
        late = []
        seen = set()
        for game in games:
            for depth in (36, 38, 40):
                history = game["history"][:depth]
                if len(history) != depth or tuple(history) in seen:
                    continue
                board, player = replay(history)
                if winner(board, -player):
                    continue
                seen.add(tuple(history))
                obs = observation(board, player)
                action = bot.act(obs)
                expected = exact(board, player)
                actual = -exact(drop(board, action, player), -player)
                assert action in obs["legal_actions"] and actual == expected, (history, actual, expected)
                late.append({"history": history, "action": action, "value": actual, **bot.last_info})
                max_wall = max(max_wall, bot.last_info["wall_seconds"])
                if len(late) >= 32:
                    break
            if len(late) >= 32:
                break
        report["checks"]["independent_late_minimax"] = late
        print("Independent minimax positions:", len(late), flush=True)
        # Check nontrivial book/search boundary decisions with the separate C++ solver.
        oracle = PROJECT / "experiments/exact12_v7/tools/score_one"
        cross = []
        for game in games[:6]:
            for depth in (12, 13, 14, 15, 18):
                history = game["history"][:depth]
                board, player = replay(history)
                action = bot.act(observation(board, player))
                info = dict(bot.last_info)
                max_wall = max(max_wall, info["wall_seconds"])
                run = subprocess.run([str(oracle), "/tmp/7x6.book", "3", "".join(str(c + 1) for c in history)],
                                     capture_output=True, text=True, timeout=12, check=True)
                values = run.stdout.split()
                if len(values) != 7:
                    cross.append({"history": history, "oracle": run.stdout.strip(), "skipped": True})
                    continue
                values = list(map(int, values))
                assert all(v in (-1000, -1, 0, 1) for v in values), values
                best = max(values)
                assert values[action] == best, (history, info, values)
                # Every advertised proof must also agree with the independent solver.
                if info["outcome"] is not None:
                    assert values[action] == info["outcome"], (history, info, values)
                cross.append({"history": history, "action": action, "oracle_wdl": values, **info})
        report["checks"]["cpp_oracle"] = cross
        print("C++ oracle positions:", len(cross), flush=True)
        # Exercise the actual no-executable path independently of the native engine.
        original = native.analyze
        native.analyze = lambda *args, **kwargs: None
        fallback = []
        for game in games[:4]:
            history = game["history"][:22]
            board, player = replay(history)
            action = bot.act(observation(board, player))
            assert action in observation(board, player)["legal_actions"]
            fallback.append({"history": history, **bot.last_info})
            max_wall = max(max_wall, bot.last_info["wall_seconds"])
        native.analyze = original
        report["checks"]["execution_blocked_fallback"] = fallback
        selfplay = []
        for opening in ([], [0, 6], [3, 3], [6, 0]):
            board, player = replay(opening)
            history = list(opening)
            moves = []
            while not winner(board, -player) and any(board[c] == 0 for c in range(7)):
                action = bot.act(observation(board, player))
                assert action in observation(board, player)["legal_actions"]
                history.append(action)
                moves.append(dict(bot.last_info))
                max_wall = max(max_wall, bot.last_info["wall_seconds"])
                board = drop(board, action, player)
                player = -player
            selfplay.append({"opening": opening, "history": history, "moves": moves,
                             "winner": -player if winner(board, -player) else 0})
            print("Selfplay:", len(history), "plies", flush=True)
        report["checks"]["selfplay"] = selfplay
        hits = sum(m["phase"] == "ponder_exact" for g in selfplay for m in g["moves"])
        if not args.disable_native:
            assert hits > 0, "Precomputed proofs were never reused"
        report["ponder_cache_hits"] = hits
        report["max_act_wall_seconds"] = max_wall
        assert max_wall < .98, max_wall
        native.close()
    report["passed"] = True
    name = "package_validation_python.json" if args.disable_native else "package_validation.json"
    (ROOT / "runs" / name).write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": True, "max_wall": max_wall,
                      "report": str(ROOT / "runs" / name)}), flush=True)


if __name__ == "__main__":
    main()
