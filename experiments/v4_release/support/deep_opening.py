"""Read exact 12-ply outcomes from a model buffer, with short retrograde search.

Database and format reference: Markus Thill, bitbully-databases 0.0.2 (MIT).
https://github.com/MarkusThill/bitbully-databases
The neural network is trained locally; the opening buffer is public search data.

MIT License
Copyright (c) 2025 Markus Thill
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adaptations: NumPy storage, bitboard Huffman encoding, and exact minimax backup.
"""
from functools import lru_cache
import numpy as np
import time
from support.tactics import (COLUMNS, ORDER, possible, safe_moves, popcount,
                            alignment)

_TABLE = None
_SOURCE = None
_DEADLINE = float("inf")
_NODES = 0

class OpeningTimeout(Exception):
    pass

def set_deadline(deadline):
    global _DEADLINE, _NODES
    _DEADLINE = deadline
    _NODES = 0

def safe_lookup(own, mask, occupied):
    try:
        return lookup(own, mask, occupied)
    except OpeningTimeout:
        return None



def configure(model):
    global _TABLE, _SOURCE
    data = getattr(model, 'opening12', None)
    if data is None:
        return False
    source = data.data_ptr()
    if _SOURCE == source:
        return _TABLE is not None
    _SOURCE = source
    raw = data.detach().cpu().numpy()
    if len(raw) != 6943780 or not raw.any():
        _TABLE = None
        return False
    table = np.frombuffer(raw, dtype='>i4').astype(np.int32)
    if np.any(table[1:] < table[:-1]):
        raise ValueError('Opening database must be sorted')
    _TABLE = table
    lookup.cache_clear()
    return True


def huffman(own, mask, mirrored=False):
    value = 0
    for col in (range(6, -1, -1) if mirrored else range(7)):
        pieces = (own >> (7*col)) & 63
        height = popcount((mask >> (7*col)) & 63)
        for row in range(height):
            value = (value << 2) | (2 if pieces & (1 << row) else 3)
        value <<= 1
    value <<= 1
    # Exactly twelve discs -> 24 disc bits + 7 separators + one pad = 32 bits.
    return value - (1 << 32) if value >= (1 << 31) else value


def _point(own, mask):
    for reflected in (False, True):
        key = huffman(own, mask, reflected)
        index = int(_TABLE.searchsorted(key))
        if index < len(_TABLE):
            item = int(_TABLE[index])
            if (item & ~3) == key:
                return -(item & 3)
    # This specific complete 12-ply book omits first-player wins by design.
    # This default is NEVER applied at other depths or without the full table.
    return 1


@lru_cache(maxsize=150000)
def lookup(own, mask, occupied):
    global _NODES
    _NODES += 1
    if _NODES % 32 == 0 and time.process_time() >= _DEADLINE:
        raise OpeningTimeout()
    if _TABLE is None or occupied > 12:
        return None
    safe, status = safe_moves(own, mask)
    if status:
        return status
    if occupied == 12:
        return _point(own, mask)
    # Only called near the 12-ply frontier; four plies at most during inference.
    if occupied < 8:
        return None
    best = -1
    for col in ORDER:
        bit = safe & COLUMNS[col]
        if bit:
            value = -lookup(own ^ mask, mask | bit, occupied + 1)
            best = max(best, value)
            if best == 1:
                break
    return best
