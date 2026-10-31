"""Upload the current release as a new daily version.

This never declares a competition version. Credentials come from
BOTLAB_COOKIE and are not printed.
"""
import argparse
import hashlib
import json
import os
import time
import uuid
import zipfile
from pathlib import Path

from botlab_api import api

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/platform"

TERMINAL = {"READY", "FAILED", "CANCELED", "REJECTED", "ERROR"}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def verify(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    folder = Path(manifest["output"])
    assert folder.is_dir(), folder
    for name, record in manifest["artifacts"].items():
        digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        assert digest == record["sha256"], (name, digest)
    with zipfile.ZipFile(folder / "code.zip") as archive:
        unsupported = [name for name in archive.namelist()
                       if not name.endswith("/")
                       and Path(name).suffix not in {".py", ".txt", ".md", ".json"}]
    assert not unsupported, unsupported
    return manifest, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", required=True, help="Dedicated baseline Bot UUID")
    parser.add_argument("--manifest", default=str(ROOT / "runs/release_manifest.json"))
    parser.add_argument("--state", default="final_upload_v9.json")
    parser.add_argument("--base-version", default=None)
    parser.add_argument("--note", default=(
        "v9; pure solver baseline; exact opening/midgame books + bounded static Rust search; no CNN/MCTS"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("BOTLAB_COOKIE"):
        raise SystemExit("BOTLAB_COOKIE is not set")
    manifest, folder = verify(args.manifest)
    assert manifest["learned_parameters"] == 0
    state_path = OUT / args.state
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    code = folder / "code.zip"
    model = folder / "model.pt"

    if args.dry_run:
        detail = {}
        existing = OUT / "platform_version.json"
        if existing.exists():
            version_id = json.loads(existing.read_text()).get("id")
            if version_id:
                probe = api("/versions/" + version_id)
                detail = {"current_version": probe.get("number"),
                          "status": probe.get("status")}
        print(json.dumps({"code_bytes": code.stat().st_size,
                          "model_bytes": model.stat().st_size,
                          "state": str(state_path), "has_commit": "commit" in state,
                          **detail}, ensure_ascii=False))
        return

    if "commit" in state:
        print(json.dumps({"already_committed": state["commit"]}, ensure_ascii=False))
        return

    tag = args.tag or ("homework4-v9-%s-%s" % (
        manifest["artifacts"]["code.zip"]["sha256"][:12],
        manifest["artifacts"]["model.pt"]["sha256"][:12]))
    if "session" not in state:
        payload = {"bot": args.bot, "base_version": args.base_version,
                   "code_size": code.stat().st_size,
                   "models_size": model.stat().st_size,
                   "remove_models": False}
        state["session"] = api("/upload-sessions", payload, key=tag + "-upload")
        save(state_path, state)
        print(json.dumps({"session": state["session"].get("id"),
                          "state": state["session"].get("state")}, ensure_ascii=False),
              flush=True)
    session = state["session"]["id"]
    receipts = state.setdefault("receipts", {})
    for kind, path in (("code", code), ("models", model)):
        if kind in receipts:
            continue
        boundary = "botlab-" + uuid.uuid4().hex
        body = (("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n" % (boundary, path.name)).encode()
                + path.read_bytes() + ("\r\n--%s--\r\n" % boundary).encode())
        receipts[kind] = api("/upload-sessions/%s/files/%s" % (session, kind), body,
                             method="PUT", content_type="multipart/form-data; boundary=" + boundary)
        save(state_path, state)
        print("Uploaded " + kind, flush=True)
    if "commit" not in state:
        state["commit"] = api("/upload-sessions/" + session + "/commit",
                              {"note": args.note,
                               "receipts": {k: v["id"] for k, v in receipts.items()}},
                              key=tag + "-commit")
        save(state_path, state)
    print(json.dumps({"commit": state["commit"]}, ensure_ascii=False), flush=True)

    if args.wait:
        version_id = state["commit"].get("id")
        deadline = time.monotonic() + 600
        last = None
        while version_id and time.monotonic() < deadline:
            detail = api("/versions/" + version_id)
            status = detail.get("status")
            if status != last:
                print(json.dumps({"version": version_id, "number": detail.get("number"),
                                  "status": status, "note": detail.get("note")},
                                 ensure_ascii=False), flush=True)
                last = status
            save(OUT / "final_upload_v9_version.json", detail)
            if status in TERMINAL:
                return
            time.sleep(5)
        raise RuntimeError("version validation did not finish in time")


if __name__ == "__main__":
    main()
