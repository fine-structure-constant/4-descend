"""Measure value/policy quality of a checkpoint on held-out exact labels."""

import argparse
import json

import numpy as np
import torch

from model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="/tmp/cnn_v10_data/exact/exact_labels.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.load(args.data)
    mask = data["validation"].astype(bool)
    x = data["x"][mask].astype(np.float32)
    policy = data["policy"][mask].astype(np.float32)
    value = data["value"][mask].astype(np.float32)
    if not len(x):
        raise SystemExit("no validation positions")

    device = torch.device(args.device)
    model, config = load_checkpoint(args.model, device)
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), args.batch):
            xb = torch.from_numpy(x[start:start + args.batch]).to(device)
            logits, values = model(xb)
            predictions.append((logits.float().cpu(), values.float().cpu()))
    logits = torch.cat([item[0] for item in predictions])
    values = torch.cat([item[1] for item in predictions]).numpy()

    log_probs = torch.log_softmax(logits, dim=1).numpy()
    policy = policy / policy.sum(1, keepdims=True)
    predicted = logits.argmax(1).numpy()
    optimal = policy > 1e-6
    top1_optimal = float(np.mean([optimal[i, predicted[i]] for i in range(len(x))]))
    best_mass = float(np.mean([policy[i, optimal[i]].max() for i in range(len(x))]))

    predicted_value = np.where(values > 0.5, 1, np.where(values < -0.5, -1, 0))
    report = {
        "model": str(args.model),
        "config": config,
        "positions": int(len(x)),
        "value_mse": round(float(np.mean((values - value) ** 2)), 5),
        "value_class_accuracy": round(float(np.mean(predicted_value == value)), 4),
        "value_accuracy_by_true": {
            str(v): round(float(np.mean(predicted_value[value == v] == v)), 4)
            for v in (-1, 0, 1)},
        "value_mse_by_true": {
            str(v): round(float(np.mean((values[value == v] - v) ** 2)), 5)
            for v in (-1, 0, 1)},
        "policy_top1_optimal": round(top1_optimal, 4),
        "policy_cross_entropy": round(float(-np.mean((policy * log_probs).sum(1))), 4),
        "policy_best_mass_mean": round(best_mass, 4),
    }
    print(json.dumps(report), flush=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
