"""Resumable 10-GPU pure-CNN training loop: self-play, DDP training, arena."""

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, cwd):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def read_rank_summaries(round_dir, world):
    records = []
    for rank in range(world):
        path = round_dir / f"rank_{rank:02d}.json"
        if not path.is_file():
            return None
        records.append(json.loads(path.read_text()))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/tmp/cnn_v10_data")
    parser.add_argument("--runs", default="/tmp/cnn_v10_runs")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--init", default=None)
    parser.add_argument("--games-per-worker", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=192)
    parser.add_argument("--sims", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--history-rounds", type=int, default=4)
    parser.add_argument("--seed-data", default="/tmp/cnn_v10_data/seed/historical.npz")
    parser.add_argument("--exact-data", default=None,
                        help="optional exact WDL shard (x/policy/value npz)")
    parser.add_argument("--arena-games", type=int, default=64)
    parser.add_argument("--arena-sims", type=int, default=32)
    parser.add_argument("--devices", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data = Path(args.data)
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    log_path = runs / "pipeline_log.jsonl"
    started = time.monotonic()

    if args.init:
        previous = Path(args.init).resolve()
    else:
        previous = runs / "init.pt"
        if not previous.is_file():
            run([args.python, "init_checkpoint.py", "--out", previous], root)
    if not previous.is_file():
        raise SystemExit(f"checkpoint not found: {previous}")

    for round_index in range(args.start_round, args.start_round + args.rounds):
        if (time.monotonic() - started) / 3600 >= args.max_hours:
            print(f"time budget reached before round {round_index}", flush=True)
            break
        round_name = f"round_{round_index:04d}"
        round_dir = data / "replays" / round_name
        checkpoint = runs / f"ckpt_{round_name}.pt"
        arena_out = runs / f"arena_{round_name}.json"
        summary_out = runs / f"summary_{round_name}.json"
        stage_started = time.monotonic()

        rank_summaries = read_rank_summaries(round_dir, args.devices)
        if rank_summaries is None:
            run([args.python, "-m", "torch.distributed.run", "--standalone",
                 "--nproc_per_node", args.devices,
                 "--master_port", 29500 + round_index % 200,
                 "selfplay.py",
                 "--checkpoint", previous, "--round", round_index,
                 "--out-dir", round_dir,
                 "--games", args.games_per_worker,
                 "--concurrency", args.concurrency,
                 "--sims", args.sims,
                 "--seed", 20261101 + 1000 * round_index], root)
            rank_summaries = read_rank_summaries(round_dir, args.devices)
            if rank_summaries is None:
                raise SystemExit("self-play did not produce every rank summary")

        if not checkpoint.is_file():
            patterns = []
            if args.seed_data:
                patterns.append(args.seed_data)
            if args.exact_data:
                patterns.append(args.exact_data)
            for old in range(max(1, round_index - args.history_rounds + 1),
                             round_index + 1):
                patterns.append(str(data / "replays" / f"round_{old:04d}" / "rank_*.npz"))
            existing = [pattern for pattern in patterns if glob.glob(pattern)]
            if not existing:
                raise SystemExit("no training shards found")
            run([args.python, "-m", "torch.distributed.run", "--standalone",
                 "--nproc_per_node", args.devices,
                 "--master_port", 29500 + (round_index + 50) % 200,
                 "train_ddp.py",
                 "--data", *existing,
                 "--init", previous,
                 "--out", checkpoint,
                 "--epochs", args.epochs,
                 "--batch", args.batch,
                 "--lr", args.lr,
                 "--log", runs / f"train_{round_name}.jsonl"], root)
        if not checkpoint.is_file():
            raise SystemExit(f"training did not produce {checkpoint}")

        if not arena_out.is_file():
            run([args.python, "arena.py",
                 "--model-a", checkpoint, "--model-b", previous,
                 "--games", args.arena_games, "--sims", args.arena_sims,
                 "--concurrency", min(args.arena_games, 64),
                 "--device", "cuda:0", "--seed", 20261101 + round_index,
                 "--out", arena_out], root)

        arena = json.loads(arena_out.read_text())
        selfplay_totals = {
            "games": sum(item["games"] for item in rank_summaries),
            "examples": sum(item["examples"] for item in rank_summaries),
            "seconds": max(item["seconds"] for item in rank_summaries),
        }
        train_summary_path = checkpoint.with_suffix(".summary.json")
        train_summary = (json.loads(train_summary_path.read_text())
                         if train_summary_path.is_file() else {})
        summary = {
            "round": round_index,
            "previous": str(previous),
            "checkpoint": str(checkpoint),
            "selfplay": selfplay_totals,
            "train": train_summary,
            "arena": {key: arena[key] for key in
                      ("games", "wins", "draws", "losses", "score", "seconds")},
            "stage_seconds": round(time.monotonic() - stage_started, 2),
        }
        summary_out.write_text(json.dumps(summary, indent=2))
        with log_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        previous = checkpoint

    print(json.dumps({"final_checkpoint": str(previous),
                      "hours": round((time.monotonic() - started) / 3600, 3)}), flush=True)


if __name__ == "__main__":
    main()
