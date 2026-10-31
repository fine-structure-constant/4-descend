"""Ask the baseline for one move; history columns are numbered 1 through 7."""
import argparse
import json

from support.bot import Bot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/model.pt")
    parser.add_argument("--history", default="")
    args = parser.parse_args()
    board = [[0] * 7 for _ in range(6)]
    player = 1
    for char in args.history:
        if char not in "1234567":
            parser.error("History must contain only columns 1..7")
        col = int(char) - 1
        rows = [r for r in range(6) if board[r][col] == 0]
        if not rows:
            parser.error("History contains a move in a full column")
        board[max(rows)][col] = player
        player = -player
    bot = Bot({"seed": 20261031}, books_path=args.model)
    action = bot.act({"game": "connect4", "board": board, "player": player,
                      "legal_actions": [c for c in range(7) if board[0][c] == 0]})
    print(json.dumps({"column_1_based": action + 1, "diagnostics": bot.last_info}, indent=2))


if __name__ == "__main__":
    main()
