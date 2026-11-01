"""Paired online evaluation of the exported V10 pure-CNN bot.

Credentials come from BOTLAB_COOKIE.  The report never contains cookies or
session identifiers.  Official play calls are read-only with respect to
competition entry and uploaded versions.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch

from deploy.net import DeployNet
from game import initial_position, legal_columns, make_move
from mcts import search, visit_counts

ROOT = Path(__file__).resolve().parent


def request_move(history, side, opponent, action):
    cookie = os.environ.get("BOTLAB_COOKIE", "")
    csrf = next((item.strip().split("=", 1)[1] for item in cookie.split(";")
                 if item.strip().startswith("csrftoken=")), "")
    # The play endpoint expects the client's latest move as `action`; when the
    # AI moves first after an opening, resend the human's last opening move.
    if action is None and history:
        history, action = history[:-1], history[-1]
    payload = {"history": history, "human_player": side, "model_id": opponent}
    if action is not None:
        payload["action"] = action
    request = urllib.request.Request(
        "https://botlab.aiphys.cn/api/v1/games/connect4/play",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Cookie": cookie,
                 "X-CSRFToken": csrf, "Origin": "https://botlab.aiphys.cn",
                 "Referer": "https://botlab.aiphys.cn/games/connect4/play"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as error:
            message = error.read().decode()[:1000]
            if error.code in (307, 308) and attempt < 3:
                target = urllib.parse.urljoin(request.full_url,
                                              error.headers.get("Location", ""))
                parts = urllib.parse.urlsplit(target)
                if parts.hostname != "botlab.aiphys.cn":
                    raise RuntimeError("Refusing cross-origin credential redirect") from None
                target = urllib.parse.urlunsplit(parts._replace(scheme="https"))
                request = urllib.request.Request(target, data=request.data,
                                                 headers=dict(request.headers))
                time.sleep(0.25)
                continue
            if error.code in (502, 503, 504) and attempt < 3:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise RuntimeError("API HTTP %d: %s" % (error.code, message)) from None
        except urllib.error.URLError as error:
            if attempt < 3:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise RuntimeError(f"API connection failed: {error.reason}") from None
    time.sleep(0.12)
    return result


def board_from_position(position):
    own, opp, _, player = position
    board = [[0] * 7 for _ in range(6)]
    for r in range(6):
        for c in range(7):
            bit = 1 << (r * 7 + c)
            if own & bit:
                board[5 - r][c] = player
            elif opp & bit:
                board[5 - r][c] = -player
    return board


def load_deploy(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("format") != "v10-deploy-fp32":
        raise SystemExit("expected an exported v10-deploy-fp32 model.pt")
    config = checkpoint["config"]
    model = DeployNet(int(config["channels"]), int(config["blocks"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, int(checkpoint.get("simulations", 64))


def source_hashes(model_path):
    paths = [Path(model_path), ROOT / "game.py", ROOT / "mcts.py",
             ROOT / "deploy" / "net.py", ROOT / "deploy" / "bot.py"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="exported models/model.pt")
    parser.add_argument("--opponent", default="strong")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--openings-file", default=str(ROOT.parent / "exact12_v6" / "runs" / "strong_openings.json"))
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("use an even number of games")
    output = Path(args.output)
    if output.exists() and not args.resume:
        parser.error("output already exists")

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, default_sims = load_deploy(args.model, device)
    simulations = args.simulations or default_sims
    openings = json.loads(Path(args.openings_file).read_text())
    hashes = source_hashes(args.model)

    records = []
    elapsed_before = 0.0
    if args.resume and output.exists():
        previous = json.loads(output.read_text())
        if previous.get("sha256") != hashes:
            raise SystemExit("cannot mix different code/weights in a resumed evaluation")
        if previous.get("opponent") != args.opponent or previous.get("seed") != args.seed:
            raise SystemExit("cannot change settings during resume")
        records = previous["records"]
        elapsed_before = float(previous.get("seconds", 0))
    started = time.time()

    for index in range(len(records), args.games):
        side = 1 if index % 2 == 0 else -1
        opening = list(openings[(index // 2) % len(openings)])
        history = list(opening)
        position = initial_position()
        for move in opening:
            if move not in legal_columns(position[2]):
                raise ValueError("opening contains an illegal move")
            position, terminal = make_move(position, move)
            if terminal is not None:
                raise ValueError("opening is already terminal")
        move_seconds = []
        move_cpu = []
        ai_moves = []
        while True:
            if position[3] == side:
                start = time.perf_counter()
                cpu_start = time.process_time()
                root = search(model, position, simulations, device)
                counts = visit_counts(root)
                legal = legal_columns(position[2])
                action = int(max(legal, key=lambda column: counts[column]))
                move_seconds.append(time.perf_counter() - start)
                move_cpu.append(time.process_time() - cpu_start)
            else:
                action = None
            response = request_move(history, side, args.opponent, action)
            returned = response["history"]
            if returned[:len(history)] != history or len(returned) <= len(history):
                raise ValueError("API history did not advance consistently")
            new_moves = returned[len(history):]
            if len(new_moves) > 2:
                raise ValueError("API returned too many new moves")
            for move in new_moves:
                if move not in legal_columns(position[2]):
                    raise ValueError("API returned an illegal move")
                position, terminal = make_move(position, move)
                if terminal is not None and move != new_moves[-1]:
                    raise ValueError("API continued after a terminal move")
            if board_from_position(position) != response["board"]:
                raise ValueError("API board disagrees with local rules")
            if (terminal is not None) != bool(response["terminal"]):
                raise ValueError("API terminal state disagrees with local rules")
            if terminal is not None:
                winner = 0 if terminal == 0 else -position[3]
                if winner != response["winner"]:
                    raise ValueError("API winner disagrees with local rules")
                result = "draw" if winner == 0 else "win" if winner == side else "loss"
                ai_moves.append(response.get("ai_action"))
                record = {
                    "game": index + 1,
                    "side": side,
                    "opponent": args.opponent,
                    "opening": opening,
                    "result": result,
                    "winner": winner,
                    "history": returned,
                    "move_seconds": move_seconds,
                    "move_cpu_seconds": move_cpu,
                    "ai_moves": ai_moves,
                }
                records.append(record)
                report = {
                    "model": str(Path(args.model).resolve()),
                    "opponent": args.opponent,
                    "simulations": simulations,
                    "device": str(device),
                    "seed": args.seed,
                    "sha256": hashes,
                    "total": {name: sum(row["result"] == name for row in records)
                              for name in ("win", "draw", "loss")},
                    "seconds": round(elapsed_before + time.time() - started, 2),
                    "records": records,
                }
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2))
                summary = {key: record[key] for key in
                           ("game", "side", "result", "winner")}
                summary["worst_cpu"] = max(move_cpu) if move_cpu else 0
                print(json.dumps(summary), flush=True)
                break
            history = returned
            ai_moves.append(response.get("ai_action"))


if __name__ == "__main__":
    main()
