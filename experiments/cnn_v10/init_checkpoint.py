"""Create a randomly initialized V10 checkpoint for the first self-play round."""

import argparse
from pathlib import Path

import torch

from model import Net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = Net(args.channels, args.blocks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"channels": args.channels, "blocks": args.blocks},
                "state_dict": model.state_dict(), "steps": 0}, out)
    print(out)


if __name__ == "__main__":
    main()
