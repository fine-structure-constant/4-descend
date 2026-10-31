"""Exact WDL-optimal moves before each of the first twelve plies (0..11 discs).

The table was backed up offline from the complete BitBully twelve-ply WDL
frontier (MIT), with 29,992 independent checks against Pascal Pons' exact
solver (AGPL v3, https://github.com/PascalPons/connect4); see REFERENCES.md.  It stores, for each symmetric position,
the set of optimal moves (a 7-bit mask) and the exact score.  Scores are +1 (win), 0 (draw), -1 (loss) for the side to move.
Every recorded move maximizes that game-theoretic WDL objective; shortest
mate distance is not used as a secondary objective.  Only non-terminal positions are stored.

Two model buffers hold the table: ``movebook_keys`` (sorted uint32 keys,
little endian) and ``movebook_data`` (interleaved move mask, int8 score).
"""

import numpy as np

WIDTH = 7
COUNT = 3356580
MAX_KEY = 3 ** 18

_KEYS = None
_MASKS = None
_SCORES = None
_SOURCE = None


def configure(model):
    """Bind to the movebook buffers of a loaded model (zero-copy)."""
    global _KEYS, _MASKS, _SCORES, _SOURCE
    raw_keys = getattr(model, "movebook_keys", None)
    raw_data = getattr(model, "movebook_data", None)
    if raw_keys is None or raw_data is None:
        _KEYS = _MASKS = _SCORES = None
        return False
    source = (raw_keys.data_ptr(), raw_data.data_ptr())
    if _SOURCE == source:
        return _KEYS is not None
    _SOURCE = source
    keys = raw_keys.detach().cpu().numpy()
    data = raw_data.detach().cpu().numpy()
    if keys.shape != (4 * COUNT,) or data.shape != (2 * COUNT,):
        _KEYS = _MASKS = _SCORES = None
        return False
    keys = keys.view("<u4")
    data = data.reshape(COUNT, 2)
    # A freshly initialised Network has zero buffers; treat that as absent.
    # The table is only usable when the keys form one strictly increasing
    # array, so a corrupt or per-depth-sorted upload falls back to search.
    if int(keys[0]) != 0 or int(keys[-1]) >= MAX_KEY:
        _KEYS = _MASKS = _SCORES = None
        return False
    if not np.all(keys[1:] > keys[:-1]):
        _KEYS = _MASKS = _SCORES = None
        return False
    if np.any(data[:, 0] == 0) or np.any(data[:, 0] > 127):
        _KEYS = _MASKS = _SCORES = None
        return False
    signed_scores = data[:, 1].view(np.int8)
    if np.any(signed_scores < -21) or np.any(signed_scores > 21):
        _KEYS = _MASKS = _SCORES = None
        return False
    _KEYS = keys
    _MASKS = data[:, 0]
    _SCORES = data[:, 1].view(np.int8)
    return True


def forward_key(own, mask):
    key = 0
    for col in range(WIDTH):
        base = 7 * col
        bits = (mask >> base) & 0x7f
        while bits:
            low = bits & -bits
            row = low.bit_length() - 1
            key = key * 3 + (1 if (own >> (base + row)) & 1 else 2)
            bits ^= low
        key *= 3
    return key


def reverse_key(own, mask):
    key = 0
    for col in range(WIDTH - 1, -1, -1):
        base = 7 * col
        bits = (mask >> base) & 0x7f
        while bits:
            low = bits & -bits
            row = low.bit_length() - 1
            key = key * 3 + (1 if (own >> (base + row)) & 1 else 2)
            bits ^= low
        key *= 3
    return key


def reverse_mask(mask):
    return (((mask & 0x01) << 6) | ((mask & 0x02) << 4) | ((mask & 0x04) << 2)
            | (mask & 0x08) | ((mask & 0x10) >> 2) | ((mask & 0x20) >> 4)
            | ((mask & 0x40) >> 6))


def lookup(own, mask):
    """Return ``(move_mask, score)`` for the side to move, or None."""
    if _KEYS is None:
        return None
    forward = forward_key(own, mask)
    reverse = reverse_key(own, mask)
    mirrored = reverse < forward
    key = (reverse if mirrored else forward) // 3
    index = int(np.searchsorted(_KEYS, key))
    if index >= COUNT or int(_KEYS[index]) != key:
        return None
    move_mask = int(_MASKS[index])
    if mirrored:
        move_mask = reverse_mask(move_mask)
    return move_mask, int(_SCORES[index])


def available():
    return _KEYS is not None
