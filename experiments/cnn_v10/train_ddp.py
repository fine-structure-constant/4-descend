"""Distributed training of the V10 pure CNN from self-play shards."""

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from model import Net, load_checkpoint


def load_shards(patterns):
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no shards matched {patterns}")
    xs, ps, vs = [], [], []
    total = 0
    for path in files:
        with np.load(path) as data:
            xs.append(data["x"])
            ps.append(data["policy"])
            vs.append(data["value"])
            total += len(data["x"])
    x = np.concatenate(xs)
    policy = np.concatenate(ps)
    value = np.concatenate(vs)
    if not (len(x) == len(policy) == len(value)):
        raise SystemExit("shard length mismatch")
    return files, x, policy.astype(np.float32), value.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--init", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128, help="per-rank batch")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--log", default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    files, x, policy, value = load_shards(args.data)
    if rank == 0:
        print(json.dumps({"shards": len(files), "examples": int(len(x)),
                          "world": world, "batch_total": args.batch * world}), flush=True)

    channels, blocks = args.channels, args.blocks
    if args.init:
        model, config = load_checkpoint(args.init, device)
        channels = int(config.get("channels", channels))
        blocks = int(config.get("blocks", blocks))
        model.to(device)
        model.train()
    else:
        model = Net(channels, blocks).to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                  betas=(0.9, 0.95))
    steps_per_epoch = max(1, (len(x) // world) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = min(args.warmup, max(1, total_steps // 10))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))

    log_path = Path(args.log) if args.log else Path(args.out).with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    step = 0
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        indices = rng.permutation(len(x))[rank::world]
        usable = (len(indices) // args.batch) * args.batch
        if usable == 0:
            raise SystemExit("not enough data for one batch per rank")
        for start in range(0, usable, args.batch):
            if args.max_steps and step >= args.max_steps:
                break
            batch_idx = indices[start:start + args.batch]
            xb = torch.from_numpy(x[batch_idx].astype(np.float32)).to(device, non_blocking=True)
            pb = torch.from_numpy(policy[batch_idx]).to(device, non_blocking=True)
            vb = torch.from_numpy(value[batch_idx]).to(device, non_blocking=True)
            flip = torch.rand(len(batch_idx), device=device) < 0.5
            if bool(flip.any()):
                xb[flip] = xb[flip].flip(-1)
                pb[flip] = pb[flip].flip(-1)

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, values = model(xb)
                log_probs = torch.log_softmax(logits.float(), dim=1)
                policy_loss = -(pb * log_probs).sum(dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(values.float(), vb)
                loss = policy_loss + args.value_weight * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 50 == 0:
                stats = torch.tensor([policy_loss.item(), value_loss.item(), loss.item()],
                                     device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.AVG)
                elapsed = time.monotonic() - started
                record = {
                    "step": step,
                    "epoch": epoch,
                    "lr": round(lr_at(step), 6),
                    "policy_loss": round(float(stats[0].item()), 5),
                    "value_loss": round(float(stats[1].item()), 5),
                    "loss": round(float(stats[2].item()), 5),
                    "examples_per_second": round(step * args.batch * world / max(elapsed, 1e-6), 1),
                }
                if rank == 0:
                    print(json.dumps(record), flush=True)
                    with log_path.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")

    if rank == 0:
        checkpoint = {
            "config": {"channels": channels, "blocks": blocks},
            "state_dict": model.module.state_dict(),
            "steps": step,
            "epochs": args.epochs,
            "examples": int(len(x)),
            "shards": files,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.out)
        summary = {"out": args.out, "steps": step, "examples": int(len(x)),
                   "seconds": round(time.monotonic() - started, 1)}
        print(json.dumps(summary), flush=True)
        with (Path(args.out).with_suffix(".summary.json")).open("w") as handle:
            json.dump(summary, handle, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
