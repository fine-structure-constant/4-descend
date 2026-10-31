"""Collect bounded sequential teacher games; keep credentials in environment."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from evaluate_api import request_move
from support.game import new_board, play_move, game_result
from support.tactics import position, Teacher, safe_moves, COLUMNS


def collect(games, output, seed):
    rng = random.Random(seed)
    path = Path(output)
    if path.exists():
        raise FileExistsError(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(depth=5, node_limit=10000)
    for game in range(games):
        side = 1 if game % 2 == 0 else -1
        board, history, player = new_board(), [], 1
        samples = []
        while True:
            if player == side:
                own, mask = position(board, player)
                scores = np.array(teacher.scores(own, mask))
                allowed, _ = safe_moves(own, mask)
                choices = [col for col in range(7) if allowed & COLUMNS[col]]
                if choices and len(history) < 10 and rng.random() < 0.35:
                    action = rng.choice(choices)
                else:
                    best = scores.max()
                    choices = np.flatnonzero(scores >= best - (8 if len(history) < 12 else 0)).tolist()
                    action = rng.choice(choices)
                submitted = history + [int(action)]
                ai_board = play_move(board, action, player)
            else:
                action = None
                submitted = history
                ai_board = board.copy()
            response = request_move(history, side, "strong", action)
            ai_action = response.get("ai_action")
            if ai_action is not None:
                samples.append({"board": ai_board.tolist(), "player": -side,
                                "action": ai_action, "history": submitted})
            returned = response["history"]
            for move in returned[len(history):]:
                board = play_move(board, move, player)
                player = -player
            history = returned
            if not np.array_equal(board, response["board"]):
                raise ValueError("API board mismatch")
            done, winner = game_result(board)
            if done:
                for sample in samples:
                    sample["z"] = 0 if winner == 0 else 1 if winner == sample["player"] else -1
                row = {"game": game, "seed": seed, "model_id": "strong", "human_player": side,
                       "history": history, "winner": winner, "samples": samples}
                with path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps({"game": game+1, "positions": len(samples), "winner": winner}), flush=True)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--output", default="data/strong.jsonl")
    parser.add_argument("--seed", type=int, default=81237)
    a = parser.parse_args()
    collect(a.games, a.output, a.seed)
