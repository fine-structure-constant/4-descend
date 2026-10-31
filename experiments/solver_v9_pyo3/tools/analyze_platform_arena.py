"""Independently audit v9 choices in the frozen hosted arena; never tune a bot."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "runs/arena_v9_vs_v7_v6_20261101_20"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    cache_path = out / "oracle_position_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    last_count = -1
    while True:
        folders = sorted(p.parent for p in out.glob("*/result.json"))
        if len(folders) == last_count and args.watch:
            time.sleep(3)
            continue
        tasks = {}
        games = []
        for folder in folders:
            result = json.loads((folder / "result.json").read_text())
            live = json.loads((folder / "live.json").read_text())
            side = 1 if result["v9_color"] == "black" else -1
            history, choices = [], []
            for frame in live["frames"][1:]:
                if frame["player"] == side and not frame.get("opening"):
                    forward = "".join(str(c + 1) for c in history)
                    reverse = "".join(str(7 - c) for c in history)
                    canonical = min(forward, reverse)
                    flipped = reverse < forward
                    choices.append({"step": frame["step"], "action": frame["action"],
                                    "history": list(history), "cache_key": canonical,
                                    "flipped": flipped})
                    if canonical not in cache:
                        tasks[canonical] = canonical
                history.append(frame["action"])
            games.append({"label": result["label"], "opponent": result["opponent"],
                          "result": result["result"], "initial_v9_wdl": result["initial_v9_wdl"],
                          "choices": choices})

        def solve(text):
            run = subprocess.run([str(PROJECT / "experiments/exact12_v7/tools/score_one"),
                                  "/tmp/7x6.book", "1.5", text],
                                 capture_output=True, text=True, timeout=12)
            fields = run.stdout.split()
            if run.returncode or len(fields) != 7:
                return text, {"solved": False, "response": run.stdout.strip()}
            values = list(map(int, fields))
            if any(v not in (-1000, -1, 0, 1) for v in values):
                return text, {"solved": False, "response": values}
            return text, {"solved": True, "wdl": values}

        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(solve, t) for t in tasks]):
                key, value = future.result()
                cache[key] = value
        save(cache_path, cache)
        checked = unresolved = 0
        drops = []
        for game in games:
            for choice in game["choices"]:
                answer = cache[choice["cache_key"]]
                if not answer["solved"]:
                    choice["oracle_solved"] = False
                    unresolved += 1
                    continue
                values = answer["wdl"][::-1] if choice["flipped"] else answer["wdl"]
                value = values[choice["action"]]
                assert value != -1000
                best = max(values)
                choice.update({"oracle_solved": True, "best_wdl": best,
                               "chosen_wdl": value, "wdl_optimal": value == best})
                checked += 1
                if value != best:
                    drops.append({"game": game["label"], "opponent": game["opponent"], **choice})
        report = {"games_audited": len(games), "v9_decisions_checked": checked,
                  "v9_decisions_unresolved": unresolved, "v9_wdl_drops": drops,
                  "oracle": "Independent C++ exact solver, with public 7x6 book",
                  "no_bot_changes": True, "games": games}
        save(out / "oracle_audit.json", report)
        progress = {k: v for k, v in report.items() if k not in {"games", "v9_wdl_drops"}}
        progress["wdl_drops"] = len(drops)
        print(json.dumps(progress), flush=True)
        last_count = len(games)
        if len(games) >= 20 or not args.watch:
            break


if __name__ == "__main__":
    main()
