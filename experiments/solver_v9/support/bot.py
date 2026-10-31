"""v9 baseline: exact books and bounded search, without a neural network."""
import time
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from support import movebook, midbook, pons_book, native
from support.endgame import ExactSolver
from support.tactics import (ORDER, COLUMNS, BOARD, position, possible,
                             alignment, safe_moves, popcount, root_guidance)

MOVE_SECONDS = .90
NATIVE_MS = 820
BOOK_NAMES = {"movebook_keys", "movebook_data", "deepbook", "midbook_count",
              "midbook_own", "midbook_mask", "midbook_wdl"}


class Bot:
    def __init__(self, context, books_path=None):
        torch.set_num_threads(1)
        path = Path(books_path) if books_path is not None else Path(__file__).resolve().parent.parent / "models/model.pt"
        # These tensors contain fixed lookup data only. No model is created.
        data = torch.load(path, map_location="cpu", weights_only=True)
        if set(data) != BOOK_NAMES:
            raise ValueError("Expected v9 book-only data, without learned weights")
        self.books = SimpleNamespace(**data)
        if not (movebook.configure(self.books) and pons_book.configure(self.books)
                and midbook.configure(self.books)):
            raise ValueError("Invalid v9 exact books")
        self.native_ready = native.warmup()
        self.fallback = ExactSolver()
        self.ponder_proofs = {}
        self.known_floors = {}
        self.current_bits = None
        self.last_info = {}
        # Initialization has a separate platform CPU budget. Warm both sides'
        # likely centre opening before the first timed decision.
        self.initial_ponder = []
        now = time.monotonic()
        self.initial_ponder.append(self._ponder(0, 0, 3, now, seconds=1.2))
        centre = 1 << 21
        self.initial_ponder.append(self._ponder(0, centre, 3, time.monotonic(), seconds=1.2))
        print(json.dumps({"baseline": "v9", "native_ready": self.native_ready,
                          "native_error": native.LAST_ERROR,
                          "initial_ponder": self.initial_ponder}), file=sys.stderr, flush=True)

    def _done(self, action, phase, started, proven=False, outcome=None):
        ponder = {}
        if phase in {"movebook12", "exact_scorebook", "forced_move"}:
            own, mask = self.current_bits
            if popcount(mask) <= 13:
                ponder = self._ponder(own, mask, action, started)
        if outcome in (0, 1):
            own, mask = self.current_bits
            after = mask | (possible(mask) & COLUMNS[action])
            opponent = own ^ mask
            if not alignment(own | (after ^ mask)):
                for c in range(7):
                    reply = possible(after) & COLUMNS[c]
                    if reply and not alignment(opponent | reply):
                        key = (opponent ^ after, after | reply)
                        self.known_floors[key] = max(outcome, self.known_floors.get(key, -1))
        self.last_info = {"phase": phase, "action": action,
                          "proven_optimal": proven, "outcome": outcome,
                          "native_ready": self.native_ready,
                          "wall_seconds": time.monotonic() - started}
        self.last_info.update(ponder)
        return int(action)

    def _ponder(self, own, mask, action, started, seconds=.80):
        """Use this move's spare time, never time while waiting for the opponent.

        Follow our book choices and branch over possible opponent responses.
        Probe the first own-turn roots beyond the opening books. The persistent
        Rust table and completed root proofs are reused on later actual turns.
        """
        deadline = started + seconds
        visited = [0]

        def frontier(us, occupied, our_turn):
            visited[0] += 1
            if visited[0] > 4096 or time.monotonic() >= deadline:
                return
            if alignment(us ^ occupied) or occupied == BOARD:
                return
            n = popcount(occupied)
            if our_turn and n >= 13:
                yield us, occupied
                return
            moves, status = safe_moves(us, occupied)
            if status or not moves:
                return
            order = [c for c in ORDER if moves & COLUMNS[c]]
            entry = movebook.lookup(us, occupied) if n <= 12 else None
            if our_turn:
                if entry is not None:
                    order = [c for c in order if entry[0] & (1 << c)][:1]
                elif n == 13:
                    values = {}
                    for c in order:
                        bit = possible(occupied) & COLUMNS[c]
                        value = pons_book.score(us ^ occupied, occupied | bit)
                        if value is not None:
                            values[c] = -int(np.sign(value))
                    if len(values) == len(order):
                        best = max(values.values())
                        order = [c for c in order if values[c] == best][:1]
                    else:
                        order = order[:1]
            elif entry is not None:
                order.sort(key=lambda c: not bool(entry[0] & (1 << c)))
            for c in order:
                bit = possible(occupied) & COLUMNS[c]
                if not alignment(us | bit):
                    yield from frontier(us ^ occupied, occupied | bit, not our_turn)

        calls = solved = 0
        bit = possible(mask) & COLUMNS[action]
        if alignment(own | bit):
            return {"ponder_calls": 0, "ponder_solved": 0}
        for us, occupied in frontier(own ^ mask, mask | bit, False):
            if (us, occupied) in self.ponder_proofs:
                continue
            exact = midbook.lookup(us, occupied)
            legal = [c for c in ORDER if possible(occupied) & COLUMNS[c]]
            if exact is not None and (any(exact[c] == 1 for c in legal) or
                                      all(exact[c] is not None for c in legal)):
                best = max(exact[c] for c in legal if exact[c] is not None)
                col = next(c for c in legal if exact[c] == best)
                self.ponder_proofs[us, occupied] = (col, best)
                solved += 1
                continue
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining < 5 or calls >= 32:
                break
            if self.native_ready and native.available():
                result = native.analyze(us, occupied, min(40, remaining))
            else:
                known = pons_book.score(us, occupied) if popcount(occupied) <= 14 else None
                known = None if known is None else int(np.sign(known))
                bounds = self.fallback.outcome_bounds(us, occupied, seconds=min(.04, remaining / 1000),
                                                     known=known, minimum=self.known_floors.get((us, occupied)))
                complete = [c for c in ORDER if bounds[c] == (1, 1)]
                all_done = all(b is None or b[0] == b[1] for b in bounds)
                if complete or all_done:
                    best = max(b[0] for b in bounds if b is not None)
                    col = next(c for c in ORDER if bounds[c] is not None and bounds[c][0] == best)
                    result = (col, best, True, [])
                else:
                    result = (0, -127, False, [])
            calls += 1
            if result is None:
                break
            col, score, complete, _ = result
            if complete:
                self.ponder_proofs[us, occupied] = (col, score)
                solved += 1
        return {"ponder_calls": calls, "ponder_solved": solved,
                "ponder_cached": len(self.ponder_proofs)}

    def act(self, observation):
        started = time.monotonic()
        if observation.get("game") != "connect4":
            raise ValueError("Only connect4 is supported")
        board = np.asarray(observation["board"], dtype=np.int8)
        player = observation["player"]
        if board.shape != (6, 7) or player not in (-1, 1):
            raise ValueError("Invalid board or player")
        own, mask = position(board, player)
        self.current_bits = (own, mask)
        if alignment(own) or alignment(own ^ mask):
            raise ValueError("Cannot act on a terminal position")
        playable = possible(mask)
        legal = {c for c in range(7) if playable & COLUMNS[c]}
        if not legal or legal != set(observation["legal_actions"]):
            raise ValueError("Invalid legal actions")
        count = popcount(mask)
        if count <= 12:
            entry = movebook.lookup(own, mask)
            if entry is not None:
                actions, score = entry
                options = [c for c in ORDER if c in legal and actions & (1 << c)]
                if options:
                    return self._done(options[0], "movebook12", started, True,
                                      int(np.sign(score)))
        cached = self.ponder_proofs.get((own, mask))
        if cached is not None and cached[0] in legal:
            return self._done(cached[0], "ponder_exact", started, True, cached[1])
        safe, status = safe_moves(own, mask)
        if status == 1:
            return self._done(next(c for c in ORDER if safe & COLUMNS[c]),
                              "immediate_win", started, True, 1)
        if status == -1:
            return self._done(next(c for c in ORDER if c in legal),
                              "forced_loss", started, True, -1)
        if mask == BOARD:
            raise ValueError("Full board")
        candidates = {c for c in legal if safe & COLUMNS[c]}
        if len(candidates) == 1:
            return self._done(next(iter(candidates)), "forced_move", started, True)

        known = pons_book.score(own, mask) if count <= 14 else None
        known = None if known is None else int(np.sign(known))
        floor = self.known_floors.get((own, mask))
        if floor == 1:
            known = 1
        outcomes = {c: -1 for c in legal if c not in candidates}
        # A 13-disc root can evaluate its 14-disc children directly.
        if count <= 13:
            for c in candidates:
                bit = playable & COLUMNS[c]
                value = pons_book.score(own ^ mask, mask | bit)
                if value is not None:
                    outcomes[c] = -int(np.sign(value))
        mid = midbook.lookup(own, mask) if count <= 21 else None
        if mid is not None:
            for c in legal:
                if mid[c] is not None:
                    if c in outcomes and outcomes[c] != mid[c]:
                        raise ValueError("Contradictory exact books")
                    outcomes[c] = mid[c]

        wins = [c for c in ORDER if outcomes.get(c) == 1]
        if wins or all(c in outcomes for c in legal):
            best = max(outcomes.values())
            action = next(c for c in ORDER if outcomes.get(c) == best)
            return self._done(action, "exact_scorebook", started, True, best)

        remaining_ms = int(max(0, MOVE_SECONDS - (time.monotonic() - started) - .08) * 1000)
        result = native.analyze(own, mask, min(NATIVE_MS, remaining_ms)) if remaining_ms > 0 else None
        native_action = None
        if result is not None:
            native_action, score, solved, wdl = result
            for c in legal:
                if wdl[c] != 9:
                    if c in outcomes and outcomes[c] != wdl[c]:
                        raise ValueError("Search contradicts exact book")
                    outcomes[c] = wdl[c]
            if solved:
                return self._done(native_action, "native_exact", started, True, score)
        else:
            # Execution-restricted hosts still get a pure search baseline.
            seconds = max(0, MOVE_SECONDS - (time.monotonic() - started) - .08)
            if seconds > .01:
                bounds = self.fallback.outcome_bounds(own, mask, seconds=seconds,
                                                     known=known, minimum=floor)
                for c in legal:
                    if bounds[c] is not None and bounds[c][0] == bounds[c][1]:
                        outcomes[c] = bounds[c][0]
                certified = [c for c in ORDER if bounds[c] is not None and bounds[c][0] >= 0]
                if certified:
                    action = max(certified, key=lambda c: bounds[c][0])
                    lo, hi = bounds[action]
                    return self._done(action, "python_proof", started,
                                      lo == 1 or known == lo, lo)

        wins = [c for c in ORDER if outcomes.get(c) == 1]
        if wins or all(c in outcomes for c in legal):
            best = max(outcomes.values())
            return self._done(next(c for c in ORDER if outcomes.get(c) == best),
                              "exact_proof", started, True, best)
        draws = [c for c in ORDER if outcomes.get(c) == 0]
        if draws and known != 1:
            return self._done(draws[0], "proved_draw", started, known == 0, 0)
        # No heuristic is allowed to override a proved better alternative.
        unresolved = {c for c in candidates if c not in outcomes}
        options = unresolved or candidates
        seconds = min(.06, max(0, MOVE_SECONDS - (time.monotonic() - started) - .01))
        guidance, depth = root_guidance(own, mask, seconds=seconds) if seconds > .005 else (None, 0)
        if guidance is not None:
            action = max((c for c in ORDER if c in options), key=lambda c: guidance[c])
        else:
            action = native_action if native_action in options else next(c for c in ORDER if c in options)
        action = self._done(action, "bounded_search", started)
        self.last_info["guidance_depth"] = depth
        return action
