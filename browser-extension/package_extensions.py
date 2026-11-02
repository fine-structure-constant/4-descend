"""Package only extension source and sanitized validation artifacts."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "releases"


def main():
    OUT.mkdir(exist_ok=True)
    core = (ROOT / "ladder-scout/core.js").read_bytes()
    assert core == (ROOT / "ladder-scout-lite/core.js").read_bytes()
    sums = []
    for name in ("ladder-scout-lite", "ladder-scout"):
        source = ROOT / name
        manifest = json.loads((source / "manifest.json").read_text())
        assert manifest["manifest_version"] == 3
        assert not manifest.get("host_permissions")
        report = json.loads((source / "docs/browser-test.json").read_text())
        assert report["passed"]
        table = json.loads((source / "docs/opening-table-verification.json").read_text())
        assert table["verified"] and not table["runtime_solver_in_browser"]
        validation = {"date": "2026-11-02", "edition": name, "unit_tests_passed": 21,
                      "shared_core_sha256": hashlib.sha256(core).hexdigest(),
                      "browser_checks_passed": len(report["checks"]),
                      "opening_classes_verified": len(table["openings"]),
                      "fetches_replays": False, "automatic_challenges": False}
        live = source / "docs/live-browser-test.json"
        validation["live_site_checked"] = live.exists() and json.loads(live.read_text()).get("passed", False)
        (source / "docs/validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2))
        relative = ["manifest.json", "core.js", "content.js", "README.md"]
        if name == "ladder-scout":
            relative += ["popup.html", "popup.js", "tests/core.test.cjs", "tests/browser-smoke.cjs"]
        relative += [str(p.relative_to(source)) for p in sorted((source / "docs").iterdir())
                     if p.suffix in {".json", ".png"}]
        output = OUT / ("botlab-" + name + "-" + manifest["version"] + ".zip")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for item in relative:
                content = (source / item).read_bytes()
                assert not re.search(rb"(?:sessionid|csrftoken)\s*=\s*[A-Za-z0-9]{16,}", content), item
                assert ".env" not in Path(item).parts
                archive.write(source / item, str(Path("botlab-" + name) / item))
        with zipfile.ZipFile(output) as archive:
            assert archive.testzip() is None
            for scripts in manifest["content_scripts"]:
                for script in scripts["js"]:
                    assert str(Path("botlab-" + name) / script) in archive.namelist()
        sha = hashlib.sha256(output.read_bytes()).hexdigest()
        sums.append(sha + "  " + output.name)
        print(json.dumps({"file": str(output), "bytes": output.stat().st_size,
                          "sha256": sha, **validation}, ensure_ascii=False))
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
