"""Connect Four rules and the six-plane encoding used by V10.

Bitboards use row 0 at the bottom and row 5 at the top.  A position is the
tuple ``(own, opp, heights, to_move)`` where ``own``/``opp`` are 42-bit masks
for the side to move and its opponent, ``heights`` counts stones per column,
and ``to_move`` is 1 (black, first) or -1 (white, second).
"""

import numpy as np

ROWS, COLS, CELLS = 6, 7, 42
BITS64 = np.array([1 << (r * 7 + c) for r in range(ROWS) for c in range(COLS)],
                  dtype=np.uint64).reshape(ROWS, COLS)

WINDOWS = [[] for _ in range(CELLS)]
for _r in range(ROWS):
    for _c in range(COLS):
        for _dr, _dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            _cells = [(_r + i * _dr, _c + i * _dc) for i in range(4)]
            if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in _cells):
                _mask = sum(1 << (rr * 7 + cc) for rr, cc in _cells)
                for rr, cc in _cells:
                    WINDOWS[rr * 7 + cc].append(_mask)
WINDOWS = [tuple(w) for w in WINDOWS]


def initial_position():
    return (0, 0, (0,) * COLS, 1)


def legal_columns(heights):
    return [c for c in range(COLS) if heights[c] < ROWS]


def move_count(position):
    return sum(position[2])


def is_win(bits, cell):
    for window in WINDOWS[cell]:
        if (bits & window) == window:
            return True
    return False


def make_move(position, column):
    """Return ``(child, terminal_value)`` for the child's side to move.

    ``terminal_value`` is None for a live position, -1 when the mover won
    (the child, now to move, has lost), and 0 for a draw.
    """
    own, opp, heights, player = position
    row = heights[column]
    bit = 1 << (row * 7 + column)
    child_heights = heights[:column] + (row + 1,) + heights[column + 1:]
    new_own = own | bit
    child = (opp, new_own, child_heights, -player)
    if is_win(new_own, row * 7 + column):
        return child, -1
    if sum(child_heights) == CELLS:
        return child, 0
    return child, None


def result_value(winner, player):
    if winner == 0:
        return 0.0
    return 1.0 if winner == player else -1.0


def _popcount(value):
    return bin(value).count("1")


def encode(position):
    """Return a ``(6, 6, 7)`` uint8 tensor with row 0 at the bottom."""
    own, opp, heights, player = position
    own_bits = np.uint64(own)
    opp_bits = np.uint64(opp)
    planes = np.empty((6, ROWS, COLS), dtype=np.uint8)
    planes[0] = (BITS64 & own_bits) != 0
    planes[1] = (BITS64 & opp_bits) != 0
    planes[2] = 1 - planes[0] - planes[1]
    playable = np.zeros((ROWS, COLS), dtype=np.uint8)
    for c in range(COLS):
        if heights[c] < ROWS:
            playable[heights[c], c] = 1
    planes[3] = playable
    first = _popcount(own) == _popcount(opp)
    rows = np.arange(ROWS)
    favoured = (rows % 2 == 0) if first else (rows % 2 == 1)
    planes[4] = favoured[:, None]
    planes[5] = 1 if player == 1 else 0
    return planes


def encode_batch(positions):
    return np.stack([encode(p) for p in positions]).astype(np.float32)


def from_board(board, player):
    """Convert a top-row-first platform board with values 1/-1/0."""
    arr = np.asarray(board, dtype=np.int8)
    if arr.shape != (ROWS, COLS):
        raise ValueError("board must be 6x7")
    own = 0
    opp = 0
    heights = [0] * COLS
    for r_top in range(ROWS):
        r_bottom = ROWS - 1 - r_top
        for c in range(COLS):
            value = int(arr[r_top, c])
            if value == 0:
                continue
            if value != 1 and value != -1:
                raise ValueError("board values must be 1, -1 or 0")
            bit = 1 << (r_bottom * 7 + c)
            if value == player:
                own |= bit
            else:
                opp |= bit
            heights[c] += 1
    for c, h in enumerate(heights):
        for r_top in range(ROWS - h):
            if arr[r_top, c] != 0:
                raise ValueError("board violates gravity")
    return (own, opp, tuple(heights), int(player))


def flip_columns(position):
    own, opp, heights, player = position

    def mirror(bits):
        out = 0
        for r in range(ROWS):
            for c in range(COLS):
                if (bits >> (r * 7 + c)) & 1:
                    out |= 1 << (r * 7 + (COLS - 1 - c))
        return out

    return (mirror(own), mirror(opp), tuple(reversed(heights)), player)
