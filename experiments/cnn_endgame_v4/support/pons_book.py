"""Exact scores for every position with at most 14 discs.

Data file and key format: Pascal Pons, Connect 4 Game Solver, release "book"
(7x6.book), https://github.com/PascalPons/connect4 (AGPL v3).  The file is a
hash table keyed by the symmetric base-3 position key; slot index and the low
8 key bits reconstruct the full key without collisions for <=14 discs.

This module only reads that public data; the encoding below is an independent
implementation of the published key description.  Score convention: positive
when the side to move wins, magnitude grows with the number of remaining
moves (faster wins are larger); 0 is a draw.
"""

import numpy as np

WIDTH = 7
HEIGHT = 6
DEPTH = 14
SIZE = 16777259          # smallest prime above 2^24
HEADER = 6
OFFSET = 19              # stored byte = score + 19; 0 means missing
EXPECTED = HEADER + 2 * SIZE

_KEYS = None
_VALUES = None
_SOURCE = None


def configure(model):
    """Bind to the ``deepbook`` buffer of a loaded model.

    Returns True when a usable book is present.  The buffer is zero-copy.
    """
    global _KEYS, _VALUES, _SOURCE
    data = getattr(model, "deepbook", None)
    if data is None:
        return False
    source = data.data_ptr()
    if _SOURCE == source:
        return _KEYS is not None
    _SOURCE = source
    raw = data.detach().cpu().numpy()
    if len(raw) != EXPECTED:
        _KEYS = _VALUES = None
        return False
    if int(raw[0]) != WIDTH or int(raw[1]) != HEIGHT or int(raw[2]) != DEPTH:
        _KEYS = _VALUES = None
        return False
    _KEYS = raw[HEADER:HEADER + SIZE]
    _VALUES = raw[HEADER + SIZE:HEADER + 2 * SIZE]
    return True


def key3(own, mask):
    """Symmetric base-3 key of the position (see module docstring)."""
    def partial(columns):
        key = 0
        for col in columns:
            base = 7 * col
            bits = (mask >> base) & 0x7f
            while bits:
                low = bits & -bits
                row = low.bit_length() - 1
                key = key * 3 + (1 if (own >> (base + row)) & 1 else 2)
                bits ^= low
            key *= 3
        return key
    forward = partial(range(WIDTH))
    reverse = partial(range(WIDTH - 1, -1, -1))
    return (forward if forward < reverse else reverse) // 3


def score(own, mask):
    """Exact score of the position, or None when outside the 14-disc book."""
    if _KEYS is None:
        return None
    key = key3(own, mask)
    slot = key % SIZE
    if int(_KEYS[slot]) != (key & 0xff):
        return None
    value = int(_VALUES[slot])
    if value == 0:
        return None
    return value - OFFSET


def available():
    return _KEYS is not None
