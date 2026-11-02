"""Call the packaged Rust abi3 Python extension directly in this process."""
import threading
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

from support.tactics import popcount, possible, COLUMNS

_SOLVER = None
_FAILED = False
_LOCK = threading.Lock()
LAST_ERROR = None
LAST_STATS = {}
BACKEND = "rust-pyo3-abi3"

try:
    from support import _c4solver
except ImportError as exc:
    # Course-approved baseline packaging: the ELF extension is submitted under
    # a .py filename. Load it explicitly as a native module, not Python source.
    path = Path(__file__).with_name("_c4solver_native.py")
    try:
        loader = importlib.machinery.ExtensionFileLoader("support._c4solver", str(path))
        spec = importlib.util.spec_from_loader("support._c4solver", loader)
        _c4solver = importlib.util.module_from_spec(spec)
        loader.exec_module(_c4solver)
        sys.modules["support._c4solver"] = _c4solver
    except (ImportError, OSError) as load_error:
        _c4solver = None
        LAST_ERROR = repr(load_error)


def available():
    return _c4solver is not None and not _FAILED


def warmup():
    global _SOLVER, _FAILED, LAST_ERROR
    if not available():
        return False
    with _LOCK:
        try:
            if _SOLVER is None:
                _SOLVER = _c4solver.Solver()
            result = _SOLVER.analyze(0, 0, 1)
            return len(result) == 6 and len(result[5]) == 7
        except Exception as exc:
            LAST_ERROR = repr(exc)
            _FAILED = True
            return False


def analyze(own, mask, budget_ms):
    global _SOLVER, LAST_STATS
    if not available():
        raise RuntimeError("Rust Python extension unavailable: " + str(LAST_ERROR))
    with _LOCK:
        if _SOLVER is None:
            _SOLVER = _c4solver.Solver()
        col, score, nodes, elapsed_ms, solved, wdl = _SOLVER.analyze(
            own, mask, max(1, min(int(budget_ms), 5000)))
        if not (0 <= col < 7 and score in (-127, -1, 0, 1) and len(wdl) == 7
                and all(v in (-1, 0, 1, 9) for v in wdl)):
            raise RuntimeError("Invalid Rust result")
        LAST_STATS = {"nodes": nodes, "elapsed_ms": elapsed_ms, "solved": solved,
                      "backend": BACKEND}
        return col, score, solved, wdl


def close():
    global _SOLVER
    with _LOCK:
        _SOLVER = None


def resist(own, mask, budget_ms):
    """Choose a longer defence; horizon scores are never WDL certificates."""
    global _SOLVER
    if not available():
        raise RuntimeError("Rust Python extension unavailable: " + str(LAST_ERROR))
    with _LOCK:
        if _SOLVER is None:
            _SOLVER = _c4solver.Solver()
        col, depth, score, nodes, elapsed_ms, complete = _SOLVER.resist(
            own, mask, max(1, min(int(budget_ms), 5000)))
    if not 0 <= col < 7 or not (possible(mask) & COLUMNS[col]):
        raise RuntimeError("Invalid Rust resistance action")
    remaining = 42 - popcount(mask)
    plies = (remaining + 1 - (abs(score) - 10000)
             if complete and abs(score) >= 10000 else None)
    return col, {"resistance_depth": depth, "resistance_complete": complete,
                 "resistance_plies": plies, "resistance_nodes": nodes,
                 "resistance_elapsed_ms": elapsed_ms}
