"""Measure c4core solve coverage per time budget on random midgame positions."""
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUST = ROOT / "native/c4core/target/release/c4core"


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
        for _ in range(target):
            legal = [col for col in range(7) if board[0][col] == 0]
            if not legal:
                break
            col = rng.choice(legal)
            history.append(col)
            for row in range(5, -1, -1):
                if board[row][col] == 0:
                    board[row][col] = player
                    break
            if winner(board, player):
                break
            player = -player
        if len(history) == target:
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
    return own, mask


def main():
    rng = random.Random(424242)
    budgets = (250, 500, 750, 1000)
    stats = {b: {"solved": 0, "win": 0, "n": 0, "wall": 0.0, "worst": 0.0} for b in budgets}
    for depth in range(13, 27):
        for _ in range(4):
            own, mask = bits(random_history(depth, rng))
            for budget in budgets:
                started = time.monotonic()
                run = subprocess.run(
                    [str(RUST), str(own), str(mask), str(budget)],
                    capture_output=True, text=True, timeout=budget / 1000 + 10,
                )
                wall = time.monotonic() - started
                col, score, nodes, ms, solved = run.stdout.split()
                st = stats[budget]
                st["n"] += 1
                st["solved"] += int(solved)
                st["win"] += int(int(score) == 1)
                st["wall"] += wall
                st["worst"] = max(st["worst"], wall)
    for budget in budgets:
        st = stats[budget]
        print(
            f"budget={budget}ms n={st['n']} solved={st['solved']} "
            f"wins={st['win']} avgwall={st['wall']/st['n']*1000:.0f}ms worst={st['worst']*1000:.0f}ms"
        )


if __name__ == "__main__":
    main()
