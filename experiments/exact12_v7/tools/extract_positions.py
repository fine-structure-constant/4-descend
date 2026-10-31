"""Extract midbook positions (12-21 discs + 1-ply deviations) from game logs.

Output: data/midbook_positions.tsv with columns
  history(1-based chars)  discs  own  mask
one row per canonical (mirror-minimised) position. The representative
history reproduces exactly the stored (own, mask) orientation.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support.game import new_board, play_move, game_result, legal_moves
from support.tactics import position

SOURCES = [ROOT / 'data' / 'harvested_games.jsonl', ROOT / 'data' / 'selfplay_games.jsonl',
           ROOT / 'data' / 'selfplay_games2.jsonl', ROOT / 'data' / 'selfplay_games3.jsonl', ROOT / 'data' / 'selfplay_games4.jsonl',
           ROOT / 'data' / 'arena_games.jsonl', ROOT / 'data' / 'selfplay_games5.jsonl',
           ROOT / 'data' / 'midgame_games.jsonl']
OUT = ROOT / 'data' / 'midbook_positions.tsv'


def mirror_bits(own, mask):
    mo = mm = 0
    for c in range(7):
        src = 63 << (7 * c)
        dst = 63 << (7 * (6 - c))
        if c < 3:
            mo |= ((own & src) << (7 * (6 - 2 * c)))
            mm |= ((mask & src) << (7 * (6 - 2 * c)))
        else:
            mo |= ((own & src) >> (7 * (2 * c - 6)))
            mm |= ((mask & src) >> (7 * (2 * c - 6)))
    return mo, mm


def mirror_history(history):
    return [6 - c for c in history]


def main():
    positions = {}
    for src in SOURCES:
        if not src.exists():
            print(f'skip missing {src}')
            continue
        for line in src.read_text().splitlines():
            obj = json.loads(line)
            history = [int(c) for c in obj['history']]
            for n in range(12, min(21, len(history)) + 1):
                prefix = history[:n]
                board = new_board()
                player = 1
                for col in prefix:
                    board = play_move(board, col, player)
                    player = -player
                if game_result(board)[0]:
                    continue
                own, mask = position(board, player)
                mo, mm = mirror_bits(own, mask)
                if (mo, mm) < (own, mask):
                    key = (mo, mm)
                    rep = mirror_history(prefix)
                else:
                    key = (own, mask)
                    rep = prefix
                if key not in positions or len(rep) < len(positions[key][0]):
                    positions[key] = (rep, n)
                if n <= 20:
                    for col in legal_moves(board):
                        child = prefix + [col]
                        cb = play_move(board, col, player)
                        cown, cmask = position(cb, -player)
                        mo, mm = mirror_bits(cown, cmask)
                        if (mo, mm) < (cown, cmask):
                            ckey = (mo, mm)
                            crep = mirror_history(child)
                        else:
                            ckey = (cown, cmask)
                            crep = child
                        if ckey not in positions or len(crep) < len(positions[ckey][0]):
                            positions[ckey] = (crep, n + 1)
    with open(OUT, 'w') as f:
        for (own, mask), (rep, discs) in positions.items():
            h = ''.join(str(c + 1) for c in rep)
            f.write(f'{h}\t{discs}\t{own}\t{mask}\n')
    counts = {}
    for _, (_, d) in positions.items():
        counts[d] = counts.get(d, 0) + 1
    print(f'{len(positions)} unique positions -> {OUT}')
    print('by discs:', dict(sorted(counts.items())))


if __name__ == '__main__':
    main()
