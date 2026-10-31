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


def wait_match(match_id, label):
    end = time.monotonic() + 600
    last = None
    while time.monotonic() < end:
        detail = api("/matches/" + match_id)
        state = (detail["phase"], detail["step"])
        if state != last:
            print(json.dumps({"match": match_id, "phase": state[0], "step": state[1]}, ensure_ascii=False), flush=True)
            last = state
        save(label + "_detail.json", detail)
        if detail["phase"] in TERMINAL:
            live = api("/matches/" + match_id + "/live")
            save(label + "_live.json", live)
            print(json.dumps({"match": match_id, "phase": detail["phase"],
                              "winner": detail["winner"], "end_reason": detail["end_reason"],
                              "black": detail["black"]["id"], "white": detail["white"]["id"]}), flush=True)
            return detail
        time.sleep(3)
    raise RuntimeError("Polling deadline reached; match remains recorded")


def challenge(version, color, label, opening_pairs=1):
    creation = OUT / (label + "_creation.json")
    if creation.exists():
        response = json.loads(creation.read_text())
    else:
        payload = {"mode": "peer", "version_a": version, "version_b": FLASH,
                   "color": color, "opening_pairs": opening_pairs}
        key_path = OUT / (label + "_request.json")
        if key_path.exists():
            request = json.loads(key_path.read_text())
            assert request["payload"] == payload
        else:
            request = {"key": str(uuid.uuid4()), "payload": payload}
            save(key_path.name, request)
        response = api("/evaluations", payload, key=request["key"])
        save(creation.name, response)
        print(json.dumps({"created": response}, ensure_ascii=False), flush=True)
    return wait_match(response["match"], label)


def upload():
    manifest = json.loads((ROOT / "runs/release_manifest.json").read_text())
    folder = Path(manifest["output"])
    with zipfile.ZipFile(folder / "code.zip") as archive:
        unsupported = [name for name in archive.namelist()
                       if not name.endswith("/") and Path(name).suffix not in {".py", ".txt", ".md", ".json"}]
    if unsupported:
        raise ValueError("Platform requires Python/text files; unsupported: " + ", ".join(unsupported))
    for name, record in manifest["artifacts"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == record["sha256"]
    state_path = OUT / "final_upload.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if "session" not in state:
        state["session"] = api("/upload-sessions", {"bot": BOT, "base_version": OLD,
                    "code_size": (folder / "code.zip").stat().st_size,
                    "models_size": (folder / "model.pt").stat().st_size,
                    "remove_models": False}, key="homework4-v6-tempo-proof-upload")
        save(state_path.name, state)
    session = state["session"]["id"]
    receipts = state.setdefault("receipts", {})
    for kind, name in [("code", "code.zip"), ("models", "model.pt")]:
        if kind in receipts:
            continue
        boundary = "botlab-" + uuid.uuid4().hex
        body = (("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n" % (boundary, name)).encode()
                + (folder / name).read_bytes() + ("\r\n--%s--\r\n" % boundary).encode())
        receipts[kind] = api("/upload-sessions/%s/files/%s" % (session, kind), body,
                             method="PUT", content_type="multipart/form-data; boundary=" + boundary)
        save(state_path.name, state)
        print("Uploaded " + kind, flush=True)
    if "commit" not in state:
        state["commit"] = api("/upload-sessions/" + session + "/commit",
                              {"note": "v6 exact12; exact spare-tempo endings; regret-trained CNN; Python only",
                               "receipts": {k: v["id"] for k, v in receipts.items()}},
                              key="homework4-v6-tempo-proof-commit")
        save(state_path.name, state)
    print(json.dumps(state["commit"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["reverse", "upload", "challenge", "wait"])
    parser.add_argument("--version")
    parser.add_argument("--color", choices=["black", "white"], default="white")
    parser.add_argument("--label", default="final_white_1")
    parser.add_argument("--opening-pairs", type=int, default=1)
    parser.add_argument("--match")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.command == "reverse":
        challenge(OLD, "white", "old_reversed")
    elif args.command == "upload":
        upload()
    elif args.command == "wait":
        wait_match(args.match, args.label)
    else:
        if not args.version:
            parser.error("--version is required")
        challenge(args.version, args.color, args.label, args.opening_pairs)
