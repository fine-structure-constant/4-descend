"""Merge old solved rows with a fresh batch solve, deduping by (own, mask)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support.game import new_board, play_move
from support.tactics import position


def replay(h):
    b = new_board()
    p = 1
    for ch in h:
        b = play_move(b, int(ch) - 1, p)
        p = -p
    return position(b, p)


def main():
    old_path = ROOT / 'data' / 'midbook_solved_v4.tsv'
    new_path = Path('/tmp/solve_new5_out.tsv')
    out_path = ROOT / 'data' / 'midbook_solved_v5.tsv'

    rows = {}
    bad = 0
    for line in old_path.read_text().splitlines():
        parts = line.split('\t')
        if len(parts) != 2 or len(parts[1].split()) != 7:
            bad += 1
            continue
        rows[replay(parts[0])] = line
    fresh = 0
    timeouts = 0
    for line in new_path.read_text().splitlines():
        parts = line.split('\t')
        if len(parts) != 2:
            bad += 1
            continue
        nums = parts[1].split()
        if len(nums) != 7 or any(int(x) == -999 for x in nums):
            timeouts += 1
            continue
        rows[replay(parts[0])] = line
        fresh += 1
    out_path.write_text('\n'.join(rows.values()) + '\n')
    print(f'{len(rows)} total rows ({fresh} fresh, {bad} malformed, {timeouts} timed-out dropped) -> {out_path}')


if __name__ == '__main__':
    main()
