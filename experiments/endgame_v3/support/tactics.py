"""Bounded alpha-beta teacher and bitboard tactics, implemented locally.

The seven-bit-column representation and losing-move pruning follow the
techniques described by Pascal Pons, blog.gamesolver.org (see REFERENCES.md).
"""
popcount = getattr(int, "bit_count", lambda value: bin(value).count("1"))

import time

class SearchTimeout(Exception):
    pass

BOTTOM = sum(1 << (7 * col) for col in range(7))
BOARD = sum(63 << (7 * col) for col in range(7))
COLUMNS = tuple(63 << (7 * col) for col in range(7))
ORDER = (3, 2, 4, 1, 5, 0, 6)
CENTER = COLUMNS[3]


def position(board, player):
    own = mask = 0
    for col in range(7):
        for row in range(6):
            bit = 1 << (7 * col + 5 - row)
            cell = board[row, col]
            if cell:
                mask |= bit
                if cell == player:
                    own |= bit
    return own, mask


def alignment(bits):
    for shift in (1, 7, 6, 8):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


def winning_cells(bits, mask):
    result = (bits << 1) & (bits << 2) & (bits << 3)
    for shift in (7, 6, 8):
        pair = (bits << shift) & (bits << (2 * shift))
        result |= pair & (bits << (3 * shift))
        result |= pair & (bits >> shift)
        pair = (bits >> shift) & (bits >> (2 * shift))
        result |= pair & (bits << shift)
        result |= pair & (bits >> (3 * shift))
    return result & (BOARD ^ mask)


def possible(mask):
    return (mask + BOTTOM) & BOARD


def safe_moves(own, mask):
    moves = possible(mask)
    wins = winning_cells(own, mask) & moves
    if wins:
        return wins, 1
    opponent_wins = winning_cells(own ^ mask, mask)
    forced = moves & opponent_wins
    if forced:
        if forced & (forced - 1):
            return 0, -1
        moves = forced
    moves &= ~(opponent_wins >> 1)
    return moves, 0 if moves else -1


WINDOWS = []
for r in range(6):
    for c in range(7):
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7:
                WINDOWS.append(sum(1 << (7 * (c + i * dc) + r + i * dr) for i in range(4)))


def heuristic(own, mask):
    opponent = own ^ mask
    a = winning_cells(own, mask)
    b = winning_cells(opponent, mask)
    score = 12 * (popcount(a) - popcount(b))
    score += 2 * (popcount(own & CENTER) - popcount(opponent & CENTER))
    for line in WINDOWS:
        mine, theirs = own & line, opponent & line
        if not theirs:
            score += (0, 1, 4, 14, 0)[popcount(mine)]
        elif not mine:
            score -= (0, 1, 4, 14, 0)[popcount(theirs)]
    return max(-500, min(500, score))


STARTS = []
for shift, dr, dc in ((1, 1, 0), (7, 0, 1), (8, 1, 1), (6, -1, 1)):
    starts = 0
    for r in range(6):
        for c in range(7):
            if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7:
                starts |= 1 << (7 * c + r)
    STARTS.append((shift, starts))


def fast_heuristic(own, mask):
    opponent = own ^ mask
    own_threats = winning_cells(own, mask)
    opp_threats = winning_cells(opponent, mask)
    score = 20 * (popcount(own_threats) - popcount(opp_threats))
    score += 2 * (popcount(own & CENTER) - popcount(opponent & CENTER))
    for pieces, blockers, sign in ((own, opponent, 1), (opponent, own, -1)):
        for d, starts in STARTS:
            a, b, c, e = pieces, pieces >> d, pieces >> (2*d), pieces >> (3*d)
            valid = starts & ~(blockers | (blockers >> d)
                               | (blockers >> (2*d)) | (blockers >> (3*d)))
            two = (a & b) | (a & c) | (a & e) | (b & c) | (b & e) | (c & e)
            score += sign * 4 * popcount(two & valid)
    return max(-500, min(500, score))


class Teacher:
    def __init__(self, depth=7, node_limit=100000, evaluator=heuristic):
        self.evaluator = evaluator
        self.depth = depth
        self.node_limit = node_limit
        self.table = {}
        self.nodes = 0
        self.deadline = None

    def negamax(self, own, mask, depth, alpha, beta):
        self.nodes += 1
        if self.deadline is not None and self.nodes % 128 == 0:
            if time.process_time() >= self.deadline:
                raise SearchTimeout()
        remaining = 42 - popcount(mask)
        if not remaining:
            return 0
        moves, status = safe_moves(own, mask)
        if status:
            return status * (10000 + remaining)
        if depth <= 0 or self.nodes >= self.node_limit:
            return self.evaluator(own, mask)
        key = (own, mask)
        entry = self.table.get(key)
        preferred = -1
        if entry:
            old_depth, value, bound, preferred = entry
            if old_depth >= depth:
                if bound == 0:
                    return value
                if bound == 1:
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if alpha >= beta:
                    return value
        original_alpha = alpha
        opponent = own ^ mask
        candidates = []
        for col in ORDER:
            bit = moves & COLUMNS[col]
            if bit:
                priority = (100 if col == preferred else 0) + popcount(winning_cells(own | bit, mask | bit))
                candidates.append((priority, col, bit))
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = -20000
        best_col = candidates[0][1]
        for _, col, bit in candidates:
            value = -self.negamax(opponent, mask | bit, depth - 1, -beta, -alpha)
            if value > best:
                best, best_col = value, col
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        bound = -1 if best <= original_alpha else 1 if best >= beta else 0
        self.table[key] = (depth, best, bound, best_col)
        return best

    def scores(self, own, mask, depth=None):
        self.nodes = 0
        self.table = {}
        depth = self.depth if depth is None else depth
        moves = possible(mask)
        scores = [-30000.] * 7
        opponent = own ^ mask
        for col in ORDER:
            bit = moves & COLUMNS[col]
            if not bit:
                continue
            if alignment(own | bit):
                scores[col] = 10000 + 42 - popcount(mask)
            else:
                scores[col] = -self.negamax(opponent, mask | bit, depth - 1, -20000, 20000)
        return scores


def root_guidance(own, mask, seconds=0.50):
    """Publish only fully completed search depths within the CPU budget."""
    teacher = Teacher(node_limit=10000000)
    teacher.deadline = time.process_time() + seconds
    scores = None
    depth = 0
    for target in (4, 6, 8, 10, 12, 14, 16, 20, 42):
        try:
            candidate = teacher.scores(own, mask, target)
        except SearchTimeout:
            break
        scores, depth = candidate, target
        legal_scores = [s for s in scores if s > -20000]
        if max(legal_scores) > 10000 or all(abs(s) > 10000 for s in legal_scores):
            break
    return scores, depth
