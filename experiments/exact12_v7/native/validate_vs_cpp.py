"""Cross-check c4core against the AGPL C++ solver with the 7x6 book."""
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUST = ROOT / "native/c4core/target/x86_64-unknown-linux-gnu/release/c4core"
CPP = ROOT / "tools/score_one"
BOOK = Path("/tmp/7x6.book")


def winner(board, player):
    for row in range(6):
        for col in range(7):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= row + 3 * dr < 6 and 0 <= col + 3 * dc < 7:
                    if all(board[row + i * dr][col + i * dc] == player for i in range(4)):
                        return True
    return False


def random_history(target, rng):
    while True:
        board = [[0] * 7 for _ in range(6)]
        history = []
        player = 1
        valid = True
        for _ in range(target):
            legal = [col for col in range(7) if board[0][col] == 0]
            if not legal:
                valid = False
                break
            col = rng.choice(legal)
            history.append(col)
            for row in range(5, -1, -1):
                if board[row][col] == 0:
                    board[row][col] = player
                    break
            if winner(board, player):
                valid = False
                break
            player = -player
        if valid and len(history) == target:
            return history


def bits(history):
    board = [[0] * 7 for _ in range(6)]
    player = 1
    for col in history:
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                board[row][col] = player
                break
        player = -player
    own = mask = 0
    for col in range(7):
        for row in range(6):
            if board[row][col]:
                bit = 1 << (7 * col + 5 - row)
                mask |= bit
                if board[row][col] == player:
                    own |= bit
    return own, mask, history


def cpp_scores(history, seconds):
    text = "".join(str(col + 1) for col in history)
    run = subprocess.run(
        [str(CPP), str(BOOK), str(seconds), text],
        capture_output=True, text=True, timeout=seconds + 8,
    )
    out = run.stdout.strip()
    if not out or out == "TIMEOUT":
        return None
    values = [int(item) for item in out.split()]
    return values if len(values) == 7 else None


def main():
    rng = random.Random(20261031)
    total = exact = comparisons = 0
    errors = []
    for depth in (14, 16, 18, 20, 22, 24):
        for _ in range(4):
            own, mask, history = bits(random_history(depth, rng))
            expected = cpp_scores(history, 3.0)
            total += 1
            if expected is None:
                continue
            run = subprocess.run(
                [str(RUST), str(own), str(mask), "1500"],
                capture_output=True, text=True, timeout=8,
            )
            fields = run.stdout.split()
            col, score, nodes, ms, solved = fields[:5]
            col, score, solved = int(col), int(score), int(solved)
            wdl = [int(value) for value in fields[5:12]]
            legal = [c for c in range(7) if expected[c] != -1000]
            if col not in legal:
                errors.append((depth, history, "illegal", col, expected))
                continue
            for c in legal:
                if wdl[c] != 9 and wdl[c] != ((expected[c] > 0) - (expected[c] < 0)):
                    errors.append((depth, history, f"wdl col{c}={wdl[c]} expected {expected[c]}", col, expected))
            for c in range(7):
                if expected[c] == -1000 and wdl[c] != 9:
                    errors.append((depth, history, f"wdl played col{c}", col, expected))
            best = max(expected[c] for c in legal)
            if solved:
                exact += 1
                comparisons += 1
                if score != best or expected[col] != best:
                    errors.append((depth, history, f"score {score} best {best} col {expected[col]}", col, expected))
            elif best >= 0 and expected[col] < 0:
                errors.append((depth, history, f"timeout blunder best {best} col {expected[col]}", col, expected))
    print(f"positions={total} solved={exact} compared={comparisons} errors={len(errors)}")
    for err in errors[:10]:
        print("ERR", err)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
