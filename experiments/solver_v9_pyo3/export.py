"""Export v9 with book tensors only; never overwrite an existing release."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
BOOK_NAMES = {"movebook_keys", "movebook_data", "deepbook", "midbook_count",
              "midbook_own", "midbook_mask", "midbook_wdl"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("books_from")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        parser.error("Choose a new output directory")
    for name in ("support/_c4solver.abi3.so", "REFERENCES.md", "AGPL-3.0.txt", "THIRD_PARTY_LICENSES.txt"):
        if not (ROOT / name).is_file():
            parser.error("Missing " + name + "; run python native/build_extension.py first")
    state = torch.load(args.books_from, map_location="cpu", weights_only=True)
    books = {k: state[k].detach().cpu().clone() for k in sorted(BOOK_NAMES)}
    out.mkdir(parents=True)
    torch.save(books, out / "model.pt")
    with zipfile.ZipFile(out / "code.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("strategy.py", "from support.bot import Bot\n")
        files = list((ROOT / "support").glob("*.py"))
        extension = ROOT / "support/_c4solver.abi3.so"
        assert extension.is_file(), "Run python native/build_extension.py first"
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(extension, "support/_c4solver_native.py")
        for name in ("REFERENCES.md", "AGPL-3.0.txt", "THIRD_PARTY_LICENSES.txt"):
            archive.write(ROOT / name, name)
    artifacts = {name: {"bytes": (out / name).stat().st_size,
                        "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
                 for name in ("code.zip", "model.pt")}
    assert artifacts["code.zip"]["bytes"] < 5_000_000
    assert artifacts["model.pt"]["bytes"] < 100_000_000
    report = {"output": str(out), "artifacts": artifacts, "learned_parameters": 0,
              "book_tensors": sorted(books), "native_budget_ms": 820,
              "description": "v9 exact books + Rust PyO3 solver; distance-sensitive defence in proven losses; no CNN or MCTS",
              "losing_policy": "Iterative deepening with terminal distances, only after a proven WDL loss; incomplete iterations discarded",
              "native_extension": {"filename": "support/_c4solver_native.py",
                                   "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
                                   "abi": "CPython abi3 >= 3.8", "required": True,
                                   "packaging": "ELF Python extension under .py filename; explicitly loaded"}}
    with zipfile.ZipFile(out / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(p in {"runs", "target", "__pycache__"} for p in relative.parts):
                continue
            if path.suffix in {".py", ".rs", ".cpp", ".toml", ".lock", ".md", ".txt"}:
                archive.write(path, str(Path("solver_v9") / relative))
    report["source_archive"] = {"bytes": (out / "source.zip").stat().st_size,
                                "sha256": hashlib.sha256((out / "source.zip").read_bytes()).hexdigest()}
    (ROOT / "runs/release_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
