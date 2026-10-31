"""Reproducible platform peer tests; authentication is supplied only via environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

BASE = "https://botlab.aiphys.cn"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/flash_probe"
BOT = "1e619e88-82a0-42d6-88f2-66199e68b402"
FLASH = "34b95b93-3588-4bd8-baf9-25d7170a8aef"
OLD = "95236ab0-16da-4220-89f8-6d4b4bd34daf"
TERMINAL = {"COMPLETED", "FAILED", "CANCELED", "REJECTED", "ERROR"}


def save(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def api(path, payload=None, method=None, content_type="application/json", key=None):
    cookie = os.environ["BOTLAB_COOKIE"]
    csrf = next(x.strip().split("=", 1)[1] for x in cookie.split(";")
                if x.strip().startswith("csrftoken="))
    method = method or ("GET" if payload is None else "POST")
    headers = {"Cookie": cookie, "X-CSRFToken": csrf, "Origin": BASE,
               "Referer": BASE + "/games/connect4/challenge", "Content-Type": content_type}
    if key:
        headers["Idempotency-Key"] = key
    body = payload if isinstance(payload, bytes) else None if payload is None else json.dumps(payload).encode()
    url = BASE + "/api/v1" + path
    for attempt in range(4):
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=50) as response:
                if response.status == 204:
                    return {"status": 204}
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:2000]
            if error.code in (307, 308) and attempt < 3:
                target = urllib.parse.urljoin(url, error.headers.get("Location", ""))
                parsed = urllib.parse.urlsplit(target)
                if parsed.hostname != "botlab.aiphys.cn":
                    raise RuntimeError("Unexpected redirect host") from None
                url = urllib.parse.urlunsplit(parsed._replace(scheme="https"))
                continue
            if error.code in (502, 503, 504) and attempt < 3 and (method == "GET" or key):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError("HTTP %d: %s" % (error.code, detail)) from None

