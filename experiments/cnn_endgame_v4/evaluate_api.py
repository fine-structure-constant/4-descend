"""Paired online evaluation. Credentials come from BOTLAB_COOKIE, never reports."""
import argparse
import hashlib
import mcts
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error

import numpy as np

from mcts import search
from support.game import game_result, new_board, play_move
from support.runtime import load_model, set_seed


def request_move(history, side, opponent, action):
    cookie = os.environ.get("BOTLAB_COOKIE", "")
    csrf = next((item.strip().split("=", 1)[1] for item in cookie.split(";")
                 if item.strip().startswith("csrftoken=")), "")
    if action is None and history:
        history, action = history[:-1], history[-1]
    payload = {"history": history, "human_player": side,
               "model_id": opponent}
    if action is not None:
        payload["action"] = action
    request = urllib.request.Request(
        "https://botlab.aiphys.cn/api/v1/games/connect4/play",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Cookie": cookie,
                 "X-CSRFToken": csrf, "Origin": "https://botlab.aiphys.cn",
                 "Referer": "https://botlab.aiphys.cn/games/connect4/play"})
    # Bounded sequential calls; stop on HTTP errors, no credential logging.
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as error:
            message = error.read().decode()[:1000]
            if error.code in (502, 503, 504) and attempt < 3:
                time.sleep(.5 * (2 ** attempt))
                continue
            raise RuntimeError("API HTTP %d: %s" % (error.code, message)) from None
    time.sleep(0.12)
    return result


def evaluate(model_path, opponent, games, simulations, seed, output, openings):
    set_seed(seed)
    model = None if os.environ.get("BOTLAB_TEACHER_DEPTH") else load_model(model_path)
    records = []
    root = Path(__file__).resolve().parent
    sources = [Path(model_path), root / "mcts.py", root / "model.py", root / "support/endgame.py",
               root / "support/tactics.py", root / "support/opening.py", root / "support/deep_opening.py"]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.exists()}
    started = time.time()
    for index in range(games):
        side = 1 if index % 2 == 0 else -1
        opening = openings[(index // 2) % len(openings)]
        history = list(opening)
        board = new_board()
        player = 1
        for action in history:
            board = play_move(board, action, player)
            player = -player
        timings = []
        cpu_timings = []
        ai_moves = []
        search_details = []
        while True:
            if player == side:
                t = time.perf_counter()
                cpu_t = time.process_time()
                if os.environ.get("BOTLAB_TEACHER_DEPTH"):
                    from support.tactics import Teacher, position
                    counts = Teacher(int(os.environ["BOTLAB_TEACHER_DEPTH"])).scores(*position(board, player))
                else:
                    counts = search(board, player, model, simulations=simulations)
                action = int(np.argmax(counts))
                search_details.append(dict(mcts.LAST_SEARCH) if hasattr(mcts, "LAST_SEARCH") else {})
                timings.append(time.perf_counter() - t)
                cpu_timings.append(time.process_time() - cpu_t)
            else:
                action = None
            response = request_move(history, side, opponent, action)
            returned = response["history"]
            if returned[:len(history)] != history or len(returned) <= len(history):
                raise ValueError("API history did not advance consistently")
            # Validate all returned moves and the server board independently.
            for move in returned[len(history):]:
                board = play_move(board, move, player)
                player = -player
            if not np.array_equal(board, response["board"]):
                raise ValueError("API board disagrees with local rules")
            history = returned
            ai_moves.append(response.get("ai_action"))
            done, winner = game_result(board)
            if done != response["terminal"]:
                raise ValueError("API terminal state disagrees with local rules")
            if done:
                if winner != response["winner"]:
                    raise ValueError("API winner disagrees with local rules")
                result = "draw" if winner == 0 else "win" if winner == side else "loss"
                row = {"game": index + 1, "side": side, "opening": opening,
                       "result": result, "winner": winner, "history": history,
                       "move_seconds": timings, "move_cpu_seconds": cpu_timings, "search_details": search_details}
                records.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                report = {"model": str(Path(model_path).resolve()), "opponent": opponent,
                          "simulations": simulations, "seed": seed,
                          "sha256": source_hashes,
                          "total": {r: sum(x["result"] == r for x in records)
                                    for r in ("win", "draw", "loss")},
                          "seconds": round(time.time() - started, 2), "records": records}
                Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--opponent", default="standard")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--output", required=True)
    parser.add_argument("--openings", default="[[],[3,3],[2,4],[4,2]]")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("Use an even number of games")
    if Path(args.output).exists():
        parser.error("Output already exists")
    evaluate(args.model, args.opponent, args.games, args.simulations,
             args.seed, args.output, json.loads(args.openings))
