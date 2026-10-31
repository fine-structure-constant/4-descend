"""Export and validate a pure-CNN V10 upload package for the platform."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from game import encode_batch, from_board, initial_position, legal_columns, make_move
from model import load_checkpoint, to_deploy

FILES = ("strategy.py", "game.py", "mcts.py",
         "deploy/__init__.py", "deploy/net.py", "deploy/bot.py")


def board_from_position(position):
    own, opp, _, player = position
    board = [[0] * 7 for _ in range(6)]
    for r in range(6):
        for c in range(7):
            bit = 1 << (r * 7 + c)
            if own & bit:
                board[5 - r][c] = player
            elif opp & bit:
                board[5 - r][c] = -player
    return board


def make_cases(rng, count=4):
    cases = []
    targets = (4, 12, 24, 34)
    while len(cases) < count:
        position = initial_position()
        for ply in range(max(targets) + 1):
            if ply in targets:
                cases.append({
                    "board": board_from_position(position),
                    "player": position[3],
                    "legal_actions": legal_columns(position[2]),
                })
            legal = legal_columns(position[2])
            if not legal:
                break
            position, terminal = make_move(position, int(rng.choice(legal)))
            if terminal is not None:
                break
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--limit", type=float, default=0.95)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new release directory")

    model, config = load_checkpoint(args.model, "cpu")
    model.eval()
    deploy = to_deploy(model)

    rng = np.random.default_rng(20261101)
    cases = make_cases(rng)
    positions = [from_board(case["board"], case["player"]) for case in cases]
    with torch.no_grad():
        logits_a, value_a = model(torch.from_numpy(encode_batch(positions)))
        logits_b, value_b = deploy(torch.from_numpy(encode_batch(positions)))
    fold_error = max(float((logits_a - logits_b).abs().max()),
                     float((value_a - value_b).abs().max()))
    if fold_error > 1e-5:
        raise SystemExit(f"BatchNorm folding error too large: {fold_error}")

    out.mkdir(parents=True)
    torch.save({"format": "v10-deploy-fp32",
                "config": {"channels": model.channels, "blocks": model.blocks_count},
                "simulations": args.sims,
                "state_dict": deploy.state_dict()}, out / "model.pt")
    with zipfile.ZipFile(out / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            path = root / name
            if not path.is_file():
                raise SystemExit(f"missing package file: {name}")
            if path.suffix not in {".py"}:
                raise SystemExit(f"non-Python package entry: {name}")
            archive.write(path, name)

    sizes = {name: (out / name).stat().st_size for name in ("code.zip", "model.pt")}
    if sizes["code.zip"] >= 5_000_000:
        raise SystemExit(f"code.zip too large: {sizes['code.zip']}")
    if sizes["model.pt"] >= 100_000_000:
        raise SystemExit(f"model.pt too large: {sizes['model.pt']}")

    with tempfile.TemporaryDirectory(prefix="v10_pkg_") as temp:
        temp = Path(temp)
        with zipfile.ZipFile(out / "code.zip") as archive:
            archive.extractall(temp)
            names = set(archive.namelist())
        if names != set(FILES) | {"strategy.py"}:
            raise SystemExit(f"unexpected archive entries: {sorted(names)}")
        (temp / "models").mkdir()
        shutil.copy2(out / "model.pt", temp / "models" / "model.pt")
        (temp / "cases.json").write_text(json.dumps(cases))
        probe = (
            "import json, time\n"
            "from strategy import Bot\n"
            "bot = Bot({'seed': 0})\n"
            "cases = json.load(open('cases.json'))\n"
            "results = []\n"
            "for case in cases:\n"
            "    start = time.process_time()\n"
            "    action = bot.act(case)\n"
            "    results.append({'action': action, 'legal': case['legal_actions'],\n"
            "                    'cpu_seconds': time.process_time() - start})\n"
            "print(json.dumps(results))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(temp)
        proc = subprocess.run([sys.executable, "-c", probe], cwd=temp, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"package probe failed:\n{proc.stdout}\n{proc.stderr}")
        results = json.loads(proc.stdout.strip().splitlines()[-1])
        worst = max(item["cpu_seconds"] for item in results)
        for item in results:
            if item["action"] not in item["legal"]:
                raise SystemExit(f"illegal packaged action: {item}")
        if worst >= args.limit:
            raise SystemExit(f"packaged CPU time {worst:.3f}s exceeds {args.limit}s")

    manifest = {
        "output": str(out),
        "artifacts": {
            name: {"bytes": sizes[name],
                   "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
            for name in sizes
        },
        "learned_parameters": sum(p.numel() for p in model.parameters()),
        "config": {"channels": model.channels, "blocks": model.blocks_count},
        "simulations": args.sims,
        "fold_max_error": fold_error,
        "cpu_probe": {
            "worst_seconds": round(worst, 4),
            "cases": [{"action": item["action"],
                       "cpu_seconds": round(item["cpu_seconds"], 4)} for item in results],
        },
        "code_format": "Python-only package; pure CNN with no tables, solver or network access",
    }
    (root / "runs" / "package_validation.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
