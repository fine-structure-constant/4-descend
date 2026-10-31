"""Self-contained exact-search helper used by the deeper search stages.

The solver is shipped inside the Python package as compressed text, unpacked to
a private temporary directory at runtime, and driven over a pipe. If the host
forbids execution the caller simply keeps its pure-Python search: every public
function here returns None instead of raising.
"""

import atexit
import base64
import hashlib
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
import zlib
from pathlib import Path

try:
    from support.native_payload import PAYLOAD, SHA256
except Exception:  # pragma: no cover - payload is optional at runtime
    PAYLOAD = None
    SHA256 = ""

_LOCK = threading.Lock()
_CALL_TIMEOUT_MARGIN = .40
_START_TIMEOUT = 6.
_SERVER = None
_BINARY = None
_WORKDIR = None
_FAILED = False
_FAILURES = 0


class _Server:
    def __init__(self, proc):
        self.proc = proc
        self.buffer = b""

    def call(self, own, mask, budget_ms, timeout):
        if self.proc.poll() is not None:
            raise RuntimeError("solver exited")
        request = f"{own} {mask} {budget_ms}\n".encode("ascii")
        self.proc.stdin.write(request)
        self.proc.stdin.flush()
        return self._readline(timeout)

    def _readline(self, timeout):
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("solver response timed out")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError("solver response timed out")
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError("solver closed its pipe")
            self.buffer += chunk
        line, _, self.buffer = self.buffer.partition(b"\n")
        return line.decode("ascii")

    def close(self):
        try:
            self.proc.stdin.write(b"quit\n")
            self.proc.stdin.flush()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=.3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                pass


def _decode_payload():
    global _BINARY
    if _BINARY is None:
        raw = zlib.decompress(base64.b64decode(PAYLOAD))
        if hashlib.sha256(raw).hexdigest() != SHA256:
            raise RuntimeError("solver payload checksum mismatch")
        _BINARY = raw
    return _BINARY


def _write_binary(directory):
    path = Path(directory) / "c4core"
    with open(path, "wb") as handle:
        handle.write(_decode_payload())
    os.chmod(path, 0o700)
    return path


def _spawn(path):
    proc = subprocess.Popen(
        [str(path), "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, bufsize=0, close_fds=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C"},
    )
    server = _Server(proc)
    fields = server.call(0, 0, 1, _START_TIMEOUT).split()
    if len(fields) != 12:
        server.close()
        raise RuntimeError("bad probe response")
    return server


def _start():
    global _SERVER, _WORKDIR
    if _SERVER is not None:
        return _SERVER
    binary = os.environ.get("C4CORE_PATH")
    if binary and Path(binary).is_file():
        try:
            _SERVER = _spawn(Path(binary))
            return _SERVER
        except OSError as exc:
            raise RuntimeError(f"cannot start solver: {exc}") from exc
    last = None
    bases = [tempfile.gettempdir(), "/dev/shm", os.getcwd(),
             str(Path(__file__).resolve().parent.parent)]
    for base in bases:
        try:
            directory = tempfile.mkdtemp(prefix=".c4core-", dir=base)
        except OSError as exc:
            last = exc
            continue
        try:
            path = _write_binary(directory)
            server = _spawn(path)
        except Exception as exc:
            last = exc
            shutil.rmtree(directory, ignore_errors=True)
            continue
        _SERVER = server
        _WORKDIR = directory
        return _SERVER
    raise RuntimeError(f"no usable solver location: {last}")


def available():
    return PAYLOAD is not None and not _FAILED


def warmup():
    """Start the solver before the first timed move; False means fall back."""
    global _FAILED, _FAILURES
    if not available():
        return False
    with _LOCK:
        try:
            _start()
            return True
        except Exception:
            _FAILED = True
            _FAILURES += 1
            return False


def analyze(own, mask, budget_ms):
    """Return (column, score, solved, per-column WDL) or None on any failure."""
    global _FAILED, _FAILURES
    if not available():
        return None
    if own < 0 or mask < 0 or own & ~mask or mask >= (1 << 49):
        return None
    budget = max(1, min(int(budget_ms), 5000))
    with _LOCK:
        try:
            server = _start()
            line = server.call(own, mask, budget,
                               budget / 1000. + _CALL_TIMEOUT_MARGIN)
            fields = line.split()
            if len(fields) != 12:
                raise RuntimeError("bad solver response")
            col, score, nodes, elapsed, solved = (int(v) for v in fields[:5])
            wdl = [int(v) for v in fields[5:]]
            if not (0 <= col < 7 and score in (-1, 0, 1, -127)
                    and solved in (0, 1)
                    and all(v in (-1, 0, 1, 9) for v in wdl)):
                raise RuntimeError("invalid solver values")
            return col, score, bool(solved), wdl
        except Exception:
            _kill_locked()
            _FAILURES += 1
            if _FAILURES >= 2:
                _FAILED = True
            return None


def _kill_locked():
    global _SERVER
    if _SERVER is not None:
        try:
            _SERVER.close()
        except Exception:
            pass
        _SERVER = None


def close():
    global _WORKDIR
    with _LOCK:
        _kill_locked()
        if _WORKDIR is not None:
            shutil.rmtree(_WORKDIR, ignore_errors=True)
            _WORKDIR = None


atexit.register(close)
