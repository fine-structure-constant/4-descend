"""Convert validated v7 game records into supervised warm-start shards.

Only games whose recorded result agrees with the board replay are kept.  The
result is a winner colour for self-play/midgame files and side-relative for
arena files.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from game import encode, initial_position, legal_columns, make_move

SELFPLAY_FILES = ("selfplay_games*.jsonl", "midgame_games.jsonl")
ARENA_FILES = ("arena_games.jsonl",)


def replay(history):
    position = initial_position()
    frames = []
    for column in history:
        if column not in legal_columns(position[2]):
            return None
        frames.append((position, int(column)))
        position, terminal_value = make_move(position, column)
        if terminal_value is not None:
            winner = 0 if terminal_value == 0 else -position[3]
            return frames, winner, True
    return frames, 0, False


def expected_winner(record, side_relative):
    result = record.get("result")
    if isinstance(result, str):
        if result == "draw":
            return 0
        if result not in ("win", "loss"):
            return None
        if not side_relative:
            return None
        side = record.get("side")
        if side not in (1, -1):
            return None
        return side if result == "win" else -side
    if result in (1, -1, 0):
        if side_relative:
            side = record.get("side")
            if side not in (1, -1):
                return None
            return 0 if result == 0 else (side if result == 1 else -side)
        return int(result)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../exact12_v7/data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth", type=float, default=0.05)
    parser.add_argument("--decisive-repeat", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.data_dir)
    xs, ps, vs, plys = [], [], [], []
    stats = {}
    files = []
    for pattern in SELFPLAY_FILES:
        files.extend((Path(path), False) for path in sorted(glob.glob(str(root / pattern))))
    for pattern in ARENA_FILES:
        files.extend((Path(path), True) for path in sorted(glob.glob(str(root / pattern))))

    for path, side_relative in files:
        kept = 0
        skipped = 0
        invalid = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            expected = expected_winner(record, side_relative)
            replayed = replay(record.get("history", []))
            if expected is None or replayed is None:
                invalid += 1
                continue
            frames, winner, finished = replayed
            if not finished or winner != expected:
                skipped += 1
                continue
            kept += 1
            repeat = args.decisive_repeat if winner != 0 else 1
            for position, column in frames:
                legal = legal_columns(position[2])
                policy = np.zeros(7, dtype=np.float32)
                policy[legal] = args.smooth / len(legal)
                policy[column] += 1.0 - args.smooth
                for _ in range(repeat):
                    xs.append(encode(position))
                    ps.append(policy.astype(np.float16))
                    vs.append(np.float16(1.0 if winner == position[3] else
                                         -1.0 if winner != 0 else 0.0))
                    plys.append(np.int16(sum(position[2])))
        stats[path.name] = {"kept_games": kept, "result_mismatch": skipped, "invalid": invalid}
        print(json.dumps({path.name: stats[path.name]}), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.stack(xs).astype(np.uint8)
    policy = np.stack(ps)
    value = np.array(vs, dtype=np.float16)
    ply = np.array(plys, dtype=np.int16)
    np.savez(out, x=x, policy=policy, value=value, ply=ply)
    summary = {"out": str(out), "examples": int(len(x)), "files": stats,
               "value_counts": {str(v): int((value == v).sum()) for v in (-1.0, 0.0, 1.0)}}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("examples", "value_counts")}), flush=True)


if __name__ == "__main__":
    main()
