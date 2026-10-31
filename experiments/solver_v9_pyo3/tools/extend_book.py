"""Add exact development positions and alternative replies, without training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from validate_package import replay, drop, winner, observation
from support.tactics import position
from support.midbook import _mirror


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROJECT / "upload-v9/model.pt"))
    parser.add_argument("--output", default=str(ROOT / "runs/books_extended.pt"))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Refusing to overwrite data")
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    n = int(data["midbook_count"][0])
    existing = {(int(data["midbook_own"][i]), int(data["midbook_mask"][i])):
                int(data["midbook_wdl"][i]) for i in range(n)}
    games = json.loads((PROJECT / "experiments/exact12_v7/runs/arena_v5nn2_fixed_vs_v6.json").read_text())["games"]
    histories = [g["history"] for g in games]
    for name in ("smoke_live.json", "flash_loss_live.json"):
        d = json.loads((ROOT / "runs/platform" / name).read_text())
        histories.append([f["action"] for f in d["frames"][1:]])
    candidates = {}
    for full in histories:
        for depth in range(12, min(24, len(full) - 1) + 1):
            prefix = full[:depth]
            board, player = replay(prefix)
            if winner(board, -player):
                continue
            alternatives = [prefix]
            if depth <= 15:
                alternatives += [prefix + [c] for c in range(7)
                                 if board[c] == 0 and not winner(drop(board, c, player), player)]
            for history in alternatives:
                b, p = replay(history)
                own, mask = position(np.asarray(observation(b, p)["board"]), p)
                mirrored = _mirror(own, mask)
                key = min((own, mask), mirrored)
                if key not in existing:
                    candidates[key] = (history, mirrored < (own, mask))
    print("Uncovered development positions:", len(candidates), flush=True)
    oracle = PROJECT / "experiments/exact12_v7/tools/score_one"
    def solve(item):
        key, (history, mirrored) = item
        result = subprocess.run([str(oracle), "/tmp/7x6.book", ".75", "".join(str(c + 1) for c in history)],
                                capture_output=True, text=True, timeout=8)
        fields = result.stdout.split()
        if result.returncode or len(fields) != 7:
            return key, None
        values = list(map(int, fields))
        if any(v not in (-1000, -1, 0, 1) for v in values):
            return key, None
        if mirrored:
            values.reverse()
        packed = sum((3 if v == -1000 else v + 1) << (2 * c) for c, v in enumerate(values))
        return key, packed
    additions = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(solve, item) for item in candidates.items()]
        for done, future in enumerate(as_completed(futures), 1):
            key, packed = future.result()
            if packed is not None:
                additions[key] = packed
            if done % 100 == 0:
                print("Processed", done, "exact additions", len(additions), flush=True)
    existing.update(additions)
    assert len(existing) <= len(data["midbook_own"])
    for i, ((own, mask), packed) in enumerate(sorted(existing.items())):
        data["midbook_own"][i] = own
        data["midbook_mask"][i] = mask
        data["midbook_wdl"][i] = packed
    data["midbook_count"][0] = len(existing)
    torch.save(data, output)
    report = {"input_positions": n, "candidate_positions": len(candidates),
              "added_exact_positions": len(additions), "total_positions": len(existing),
              "development_data": True, "note": "Exact offline labels; timeouts are omitted, not labelled as draws.",
              "additions": [[own, mask, value] for (own, mask), value in sorted(additions.items())]}
    (ROOT / "runs/book_extension.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "additions"}), flush=True)


if __name__ == "__main__":
    main()
