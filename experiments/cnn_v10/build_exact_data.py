"""Convert exact WDL solver labels into V10 training tensors.

The v7 refinement states are top-row-first with own = side to move.  V10
stores row 0 at the bottom, so the board is vertically flipped; the side is
inferred from the stone counts, and every position is re-validated with the
independent V10 rules before it is written.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from game import CELLS, ROWS, COLS, encode, is_win, legal_columns


def convert(states, policies, values, validation):
    rows = []
    skipped = {"terminal": 0, "bad_counts": 0, "bad_policy": 0, "bad_value": 0}
    for index in range(len(states)):
        own_plane = states[index, 0] > 0
        opp_plane = states[index, 1] > 0
        if np.any(own_plane & opp_plane):
            skipped["bad_counts"] += 1
            continue
        own_count = int(own_plane.sum())
        opp_count = int(opp_plane.sum())
        if own_count == opp_count:
            player = 1
        elif own_count + 1 == opp_count:
            player = -1
        else:
            skipped["bad_counts"] += 1
            continue

        own_bits = 0
        opp_bits = 0
        heights = [0] * COLS
        gravity_ok = True
        for c in range(COLS):
            filled = np.where(own_plane[:, c] | opp_plane[:, c])[0]
            if len(filled) and not np.array_equal(filled, np.arange(ROWS - len(filled), ROWS)):
                gravity_ok = False
                break
            heights[c] = len(filled)
            for r_top in filled:
                bit = 1 << ((ROWS - 1 - r_top) * 7 + c)
                if own_plane[r_top, c]:
                    own_bits |= bit
                else:
                    opp_bits |= bit
        if not gravity_ok or sum(heights) != own_count + opp_count:
            skipped["bad_counts"] += 1
            continue
        if any(is_win(own_bits, cell) for cell in range(CELLS) if (own_bits >> cell) & 1):
            skipped["terminal"] += 1
            continue
        if any(is_win(opp_bits, cell) for cell in range(CELLS) if (opp_bits >> cell) & 1):
            skipped["terminal"] += 1
            continue

        legal = legal_columns(tuple(heights))
        policy = np.asarray(policies[index], dtype=np.float32)
        if policy.shape != (COLS,) or not np.isfinite(policy).all() or policy[legal].sum() <= 0:
            skipped["bad_policy"] += 1
            continue
        illegal_mass = float(policy.sum() - policy[legal].sum())
        if illegal_mass > 1e-4:
            skipped["bad_policy"] += 1
            continue
        normalized = policy[legal] / policy[legal].sum()
        target = np.zeros(COLS, dtype=np.float32)
        target[legal] = normalized

        value = float(values[index])
        if value not in (-1.0, 0.0, 1.0):
            skipped["bad_value"] += 1
            continue
        rows.append((own_bits, opp_bits, tuple(heights), player, target, value,
                     bool(validation[index])))
    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="/home/stu2026/stu2400011444/homework4/"
                        "experiments/exact12_v7/data/refinement_v10.npz")
    parser.add_argument("--out", default="/tmp/cnn_v10_data/exact/exact_labels.npz")
    args = parser.parse_args()

    data = np.load(args.labels)
    rows, skipped = convert(data["states"], data["policies"], data["values"],
                            data["validation"])
    if not rows:
        raise SystemExit("no exact positions survived validation")
    x = np.stack([encode(row[:4]) for row in rows])
    policy = np.stack([row[4] for row in rows]).astype(np.float16)
    value = np.array([row[5] for row in rows], dtype=np.float16)
    validation = np.array([row[6] for row in rows], dtype=bool)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x, policy=policy, value=value, validation=validation)
    train = ~validation
    train_out = out.with_name(out.stem.replace("_labels", "") + "_train.npz")
    np.savez_compressed(train_out, x=x[train], policy=policy[train], value=value[train],
                        validation=validation[train])
    summary = {
        "out": str(out), "positions": len(rows), "skipped": skipped,
        "train_out": str(train_out), "train_positions": int(train.sum()),
        "value_counts": {str(v): int((value == v).sum()) for v in (-1, 0, 1)},
        "validation": int(validation.sum()),
        "bytes": out.stat().st_size, "train_bytes": train_out.stat().st_size,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
