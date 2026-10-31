"""Harvest exact game histories from v6 run logs into a plain JSONL.

Each line: a game history (list of 0-based columns). Sources: every *.log
line that parses as a JSON object with a "history" list field.
"""
import json
import sys
from pathlib import Path

V6_RUNS = Path('/home/stu2026/stu2400011444/homework4/experiments/exact12_v6/runs')
OUT = Path('/home/stu2026/stu2400011444/homework4/experiments/exact12_v7/data/harvested_games.jsonl')


def main():
    seen = set()
    count = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        for log in sorted(V6_RUNS.glob('*.log')):
            for line in log.read_text(errors='ignore').splitlines():
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                history = obj.get('history')
                if not isinstance(history, list) or not history:
                    continue
                history = tuple(int(c) for c in history)
                if history in seen:
                    continue
                seen.add(history)
                f.write(json.dumps({'history': list(history), 'game': obj.get('game'),
                                    'opponent': obj.get('opponent'), 'result': obj.get('result'),
                                    'source': log.name}) + '\n')
                count += 1
    print(f'harvested {count} unique games -> {OUT}')


if __name__ == '__main__':
    main()
