"""Exact per-column WDL for midgame positions, labelled offline by the C++ solver.

The table stores the canonical (mirror-minimal) orientation of every covered
position; the query orientation is canonicalised on lookup and the column
outcomes are mirrored back when needed. Values are proven exact; a lookup
miss falls back to ordinary search.
"""
import numpy as np

MAX = 60000

_OWN = None
_MASK = None
_WDL = None
_SOURCE = None


def _reset():
    global _OWN, _MASK, _WDL
    _OWN = _MASK = _WDL = None


def _mirror(own, mask):
    mo = mm = 0
    for c in range(7):
        src = 63 << (7 * c)
        shift = 7 * (6 - 2 * c)
        if shift >= 0:
            mo |= (own & src) << shift
            mm |= (mask & src) << shift
        else:
            mo |= (own & src) >> (-shift)
            mm |= (mask & src) >> (-shift)
    return mo, mm


def configure(model):
    global _OWN, _MASK, _WDL, _SOURCE
    count = getattr(model, "midbook_count", None)
    own = getattr(model, "midbook_own", None)
    mask = getattr(model, "midbook_mask", None)
    wdl = getattr(model, "midbook_wdl", None)
    if count is None or own is None or mask is None or wdl is None:
        _reset()
        return False
    source = (count.data_ptr(), own.data_ptr(), mask.data_ptr(), wdl.data_ptr())
    if _SOURCE == source:
        return _OWN is not None
    _SOURCE = source
    n = int(count.detach().cpu().numpy()[0])
    if not 0 < n <= MAX:
        _reset()
        return False
    own = own.detach().cpu().numpy()[:n].view(np.int64)
    mask = mask.detach().cpu().numpy()[:n].view(np.int64)
    wdl = wdl.detach().cpu().numpy()[:n].view(np.int16)
    for i in range(1, n):
        if (own[i - 1], mask[i - 1]) >= (own[i], mask[i]):
            _reset()
            return False
    _OWN, _MASK, _WDL = own, mask, wdl
    return True


def lookup(own, mask):
    """Return the 7 per-column outcomes for the side to move, or None."""
    if _OWN is None:
        return None
    mo, mm = _mirror(own, mask)
    flipped = (mo, mm) < (own, mask)
    if flipped:
        own, mask = mo, mm
    idx = int(np.searchsorted(_OWN, own, side="left"))
    n = len(_OWN)
    while idx < n and _OWN[idx] == own:
        if _MASK[idx] == mask:
            packed = int(_WDL[idx])
            cols = range(7)
            if flipped:
                packed = sum(((packed >> (2 * c)) & 3) << (2 * (6 - c)) for c in cols)
            out = [None] * 7
            for col in cols:
                code = (packed >> (2 * col)) & 3
                out[col] = -1 if code == 0 else 0 if code == 1 else 1 if code == 2 else None
            return out
        idx += 1
    return None
