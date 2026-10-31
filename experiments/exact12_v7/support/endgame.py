"""Exact endgame search with proven bounds and a strict CPU deadline.

Uses standard bitboard, non-losing-move and null-window search ideas described
by Pascal Pons. Implemented here in Python; no remote solver or binary engine.
A timeout leaves an unknown result, never a heuristic masquerading as proof.
"""
import time
from support import pons_book
from support.tactics import (BOARD, BOTTOM, COLUMNS, ORDER, winning_cells,
                            popcount, possible, alignment)


class Timeout(Exception):
    pass


class ExactSolver:
    def __init__(self, capacity=250000):
        self.table = {}
        self.capacity = capacity
        self.nodes = 0
        self.deadline = float('inf')

    def _check_time(self):
        self.nodes += 1
        if self.nodes & 255 == 0 and time.process_time() >= self.deadline:
            raise Timeout()

    def negamax(self, own, mask, moves, alpha, beta):
        self._check_time()
        if moves == 42:
            return 0
        playable = (mask + BOTTOM) & BOARD
        if winning_cells(own, mask) & playable:
            return (43 - moves) // 2
        if moves <= 14:
            exact = pons_book.score(own, mask)
            if exact is not None:
                return exact
        opponent = own ^ mask
        threats = winning_cells(opponent, mask)
        forced = playable & threats
        if forced:
            if forced & (forced - 1):
                return -((42 - moves) // 2)
            playable = forced
        playable &= ~(threats >> 1)
        if not playable:
            return -((42 - moves) // 2)
        if moves >= 40:
            return 0
        # Without an immediate win, our earliest victory is on the third ply.
        min_score = -((42 - moves) // 2)
        max_score = (41 - moves) // 2
        if alpha < min_score:
            alpha = min_score
            if alpha >= beta:
                return alpha
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta
        # (own, mask) is injective; the sum own + mask is not (e.g. (1,7)
        # and (2,6) both sum to 8 but are different positions).
        key = (own, mask)
        lower, upper, preferred = self.table.get(key, (-22, 22, -1))
        if lower >= beta:
            return lower
        if upper <= alpha:
            return upper
        alpha = max(alpha, lower)
        beta = min(beta, upper)
        if alpha >= beta:
            return alpha
        initial_alpha, initial_beta = alpha, beta
        candidates = []
        for col in ORDER:
            bit = playable & COLUMNS[col]
            if bit:
                rank = popcount(winning_cells(own | bit, mask | bit))
                candidates.append((rank + (100 if col == preferred else 0), col, bit))
        candidates.sort(key=lambda item: item[0], reverse=True)
        best, best_col = -22, candidates[0][1]
        for _, col, bit in candidates:
            value = -self.negamax(opponent, mask | bit, moves + 1, -beta, -alpha)
            if value > best:
                best, best_col = value, col
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        if best <= initial_alpha:
            upper = min(upper, best)
        elif best >= initial_beta:
            lower = max(lower, best)
        else:
            lower = upper = best
        self.table[key] = (lower, upper, best_col)
        return best

    def solve(self, own, mask, moves):
        # Null-window searches refine exact integer outcome/distance bounds.
        lower, upper = -((42 - moves) // 2), (43 - moves) // 2
        while lower < upper:
            middle = lower + (upper - lower) // 2
            if middle <= 0 and lower // 2 < middle:
                middle = lower // 2
            elif middle >= 0 and upper // 2 > middle:
                middle = upper // 2
            value = self.negamax(own, mask, moves, middle, middle + 1)
            if value <= middle:
                upper = value
            else:
                lower = value
        return lower

    def score_moves(self, own, mask, seconds=.55, preferred=None):
        """Each returned score is exact; None means unresolved after timeout."""
        self.nodes = 0
        self.deadline = time.process_time() + seconds
        if len(self.table) > self.capacity:
            self.table.clear()
        count = popcount(mask)
        scores = [None] * 7
        playable = possible(mask)
        ordering = list(ORDER)
        if preferred in ordering:
            ordering.remove(preferred)
            ordering.insert(0, preferred)
        try:
            for col in ordering:
                bit = playable & COLUMNS[col]
                if not bit:
                    continue
                if alignment(own | bit):
                    scores[col] = (43 - count) // 2
                else:
                    scores[col] = -self.solve(own ^ mask, mask | bit, count + 1)
        except Timeout:
            pass
        return scores


    def outcome_bounds(self, own, mask, seconds=.5, preferred=None, known=None, minimum=None):
        """Prove win/draw/loss bounds fairly across columns before mate distance.

        Bounds use {-1,0,+1}; (0,+1) means non-losing but not yet a proved win.
        Every narrow-window call can be interrupted without invalidating bounds.
        """
        self.nodes = 0
        if len(self.table) > self.capacity:
            self.table.clear()
        global_deadline = time.process_time() + seconds
        count = popcount(mask)
        playable = possible(mask)
        own_wins = winning_cells(own, mask) & playable
        bounds = [None] * 7
        order = list(ORDER)
        if preferred in order:
            order.remove(preferred)
            order.insert(0, preferred)
        for col in order:
            bit = playable & COLUMNS[col]
            if not bit:
                continue
            if bit & own_wins:
                bounds[col] = (1, 1)
                continue
            child_mask = mask | bit
            if child_mask == BOARD:
                bounds[col] = (0, 0)
            elif winning_cells(own ^ mask, child_mask) & possible(child_mask):
                bounds[col] = (-1, -1)
            else:
                bounds[col] = (-1, 0) if known == 0 else (-1, 1)
        if own_wins:
            return bounds
        if known == -1:
            return [(-1, -1) if b is not None else None for b in bounds]
        def infer_last():
            target = known if known in (0, 1) else minimum
            if target not in (0, 1):
                return False
            candidates = [col for col in order if bounds[col] and bounds[col][1] >= target]
            if len(candidates) == 1:
                col = candidates[0]
                bounds[col] = (target, target if known is not None else bounds[col][1])
                return True
            return False
        if infer_last():
            return bounds
        # Round-robin slices avoid spending the entire budget on the first move.
        if known == 1:
            # The root is known to win from an exact source. A single completed
            # window search settles each move (fail-low: our move wins;
            # fail-high: the move is only a draw), so hunt for a proven win
            # with growing slices and stop as soon as one is found. Refuted
            # draws leave the pool; one survivor inherits the known win.
            for sweep in range(8):
                pending = [col for col in order
                           if bounds[col] is not None and bounds[col][0] != bounds[col][1]
                           and bounds[col] != (-1, 0)]
                if not pending:
                    break
                if time.process_time() >= global_deadline:
                    break
                if len(pending) == 1 and sweep > 0:
                    bounds[pending[0]] = (1, 1)
                    return bounds
                for col in pending:
                    if time.process_time() >= global_deadline:
                        break
                    bit = playable & COLUMNS[col]
                    self.deadline = min(global_deadline, time.process_time() + .10 * (1 + sweep))
                    try:
                        value = self.negamax(own ^ mask, mask | bit, count + 1, -1, 0)
                        bounds[col] = (1, 1) if value < 0 else (-1, 0)
                    except Timeout:
                        continue
                    if bounds[col] == (1, 1):
                        return bounds
            return bounds
        for sweep in range(12):
            pending = [col for col in order if bounds[col] and bounds[col][0] != bounds[col][1]]
            if not pending or time.process_time() >= global_deadline:
                break
            for col in pending:
                if time.process_time() >= global_deadline:
                    break
                bit = playable & COLUMNS[col]
                lo, hi = bounds[col]
                self.deadline = min(global_deadline, time.process_time() + .035 * (1 + sweep))
                try:
                    if lo < 0:
                        # Is the opponent able to force a win? If no, this move
                        # preserves at least a draw for the root player.
                        value = self.negamax(own ^ mask, mask | bit, count + 1, 0, 1)
                        if value > 0:
                            bounds[col] = (-1, -1)
                            if infer_last():
                                return bounds
                            continue
                        lo = 0
                        bounds[col] = (lo, hi)
                        if minimum is not None and minimum >= 0:
                            return bounds
                    if hi > 0:
                        value = self.negamax(own ^ mask, mask | bit, count + 1, -1, 0)
                        bounds[col] = (1, 1) if value < 0 else (lo, 0)
                        if bounds[col] == (1, 1):
                            return bounds
                except Timeout:
                    pass
        return bounds


SOLVER = ExactSolver()
