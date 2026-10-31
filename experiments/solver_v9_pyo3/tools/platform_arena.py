"""Frozen 20-game hosted comparison: v9 Rust vs v7/v6, mirrored random starts."""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from botlab_api import api

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from validate_package import drop, winner

V9 = "414ab6a5-a6c3-45dd-a48c-74ed152469c7"
OPPONENTS = {"v7": "65a6ca5f-a365-4f8d-a03a-31e02c33062e",
             "v6": "418cae24-b542-427a-ac9b-1572359f1ae8"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELED", "REJECTED", "ERROR"}


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def audit(detail, live, logs):
    frames = live["frames"]
    board, player = (0,) * 42, 1
    history = []
    for f in frames[1:]:
        assert not winner(board, -player), "Frame after terminal state"
        assert f["player"] == player and board[f["action"]] == 0
        board = drop(board, f["action"], player)
        assert list(board) == [cell for row in f["board"] for cell in row]
        history.append(f["action"])
        player = -player
    opening = detail["opening_actions"]
    assert len(opening) == 2 and opening[0] + opening[1] == 6
    assert history[:2] == opening and detail["opening_pairs"] == 1
    v9_color = "black" if detail["black"]["id"] == V9 else "white"
    assert '"native_ready": true' in logs[v9_color]
    assert '"native_backend": "rust-pyo3-abi3"' in logs[v9_color]
    if detail["phase"] == "COMPLETED" and detail["end_reason"] == "NORMAL":
        actual = 1 if winner(board, 1) else -1 if winner(board, -1) else 0
        assert detail["winner"] == actual
        if actual == 0:
            assert all(board), "Non-full board marked as a normal draw"
    side = 1 if v9_color == "black" else -1
    moves = [f for f in frames[1:] if f.get("player") == side and not f.get("opening")]
    return {"v9_color": v9_color, "history": history,
            "v9_max_cpu_ms": max((f.get("cpu_ms", 0) for f in moves), default=0),
            "v9_max_wall_ms": max((f.get("wall_ms", 0) for f in moves), default=0),
            "native_verified": True, "rules_verified": True}


def summarize(results):
    report = {"completed_games": len(results), "opponents": {}}
    for name in OPPONENTS:
        rows = [r for r in results if r["opponent"] == name]
        count = collections.Counter(r["result"] for r in rows)
        report["opponents"][name] = {
            "games": len(rows), "wins": count["win"], "draws": count["draw"],
            "losses": count["loss"], "unresolved": count["unresolved"],
            "win_rate": count["win"] / len(rows) if rows else None,
            "score_rate": (count["win"] + .5 * count["draw"]) / len(rows) if rows else None,
            "technical_games": sum(r["end_reason"] != "NORMAL" for r in rows),
            "colors": dict(collections.Counter(r["v9_color"] for r in rows)),
            "openings_0_based": [r["opening_actions"] for r in rows],
            "max_v9_cpu_ms": max((r["v9_max_cpu_ms"] for r in rows), default=0),
            "max_v9_wall_ms": max((r["v9_max_wall_ms"] for r in rows), default=0),
            "by_initial_v9_wdl": {
                str(value): dict(collections.Counter(r["result"] for r in rows
                                                     if r["initial_v9_wdl"] == value))
                for value in (-1, 0, 1)}}
    report["results"] = results
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "runs/arena_v9_vs_v7_v6_20261101_20"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"v9": V9, "opponents": OPPONENTS, "games_per_opponent": 10,
              "opening_pairs": 1, "same_opening_color_pairs_controlled": False,
              "description": "20 independent hosted games; each opponent 5 black/5 white; platform mirrored random opening"}
    if (out / "config.json").exists():
        assert json.loads((out / "config.json").read_text()) == config
    else:
        save(out / "config.json", config)
        save(out / "quota_before.json", api("/quotas"))
        versions = {}
        for name, version in {"v9": V9, **OPPONENTS}.items():
            detail = api("/versions/" + version)
            assert detail["status"] == "READY" and detail["can_play"]
            versions[name] = detail
        save(out / "frozen_versions.json", versions)
    # Oracle labels the four mirror-equivalent opening classes independently.
    initial_values = {}
    for col in range(7):
        history = [col, 6 - col]
        run = subprocess.run([str(PROJECT / "experiments/exact12_v7/tools/score_one"),
                              "/tmp/7x6.book", "5", "".join(str(c + 1) for c in history)],
                             capture_output=True, text=True, timeout=12, check=True)
        values = list(map(int, run.stdout.split()))
        assert len(values) == 7 and all(v in (-1000, -1, 0, 1) for v in values)
        initial_values[col] = max(values)
    save(out / "initial_opening_black_wdl.json", initial_values)
    results = []
    for round_index in range(5):
        for name, opponent in OPPONENTS.items():
            for color in ("black", "white"):
                label = "%s_%02d_%s" % (name, round_index + 1, color)
                folder = out / label
                folder.mkdir(exist_ok=True)
                if (folder / "result.json").exists():
                    results.append(json.loads((folder / "result.json").read_text()))
                    continue
                black, white = (V9, opponent) if color == "black" else (opponent, V9)
                payload = {"mode": "compare", "version_a": black, "version_b": white,
                           "opening_pairs": 1}
                request_path = folder / "request.json"
                if request_path.exists():
                    request = json.loads(request_path.read_text())
                    assert request["payload"] == payload
                else:
                    request = {"payload": payload, "key": str(uuid.uuid4())}
                    save(request_path, request)
                if (folder / "creation.json").exists():
                    creation = json.loads((folder / "creation.json").read_text())
                else:
                    creation = api("/evaluations", payload, key=request["key"])
                    save(folder / "creation.json", creation)
                    print(json.dumps({"created": label, "match": creation["match"]}), flush=True)
                match = creation["match"]
                deadline = time.monotonic() + 900
                while True:
                    detail = api("/matches/" + match)
                    save(folder / "detail.json", detail)
                    if detail["phase"] in TERMINAL:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Match still active; resume this same output directory")
                    time.sleep(2)
                assert detail["black"]["id"] == black and detail["white"]["id"] == white
                live = api("/matches/" + match + "/live")
                logs = api("/matches/" + match + "/logs")
                save(folder / "live.json", live)
                save(folder / "logs.json", logs)
                checked = audit(detail, live, logs)
                side = 1 if color == "black" else -1
                winner_side = detail.get("winner")
                result = ("draw" if winner_side == 0 else "win" if winner_side == side
                          else "loss" if winner_side == -side else "unresolved")
                row = {"label": label, "opponent": name, "match": match,
                       "phase": detail["phase"], "end_reason": detail["end_reason"],
                       "winner": winner_side, "result": result,
                       "opening_actions": detail["opening_actions"],
                       "opening_seed": detail.get("opening_seed"),
                       "initial_v9_wdl": side * initial_values[detail["opening_actions"][0]],
                       **checked}
                save(folder / "result.json", row)
                results.append(row)
                summary = summarize(results)
                save(out / "summary.json", summary)
                print(json.dumps({"finished": len(results), "opponent": name, "color": color,
                                  "result": result, "opening": row["opening_actions"],
                                  "initial_v9_wdl": row["initial_v9_wdl"],
                                  "scores": {n: {k: v for k, v in s.items() if k in
                                                  {"games", "wins", "draws", "losses"}}
                                             for n, s in summary["opponents"].items()}},
                                 ensure_ascii=False), flush=True)
    save(out / "summary.json", summarize(results))
    save(out / "quota_after.json", api("/quotas"))
    print(json.dumps({"complete": True, "summary": str(out / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
