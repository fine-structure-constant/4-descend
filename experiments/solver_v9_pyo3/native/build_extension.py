"""Build the CPython stable-ABI Rust module with the shared local toolchain."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    env.setdefault("RUSTUP_HOME", "/tmp/rustup")
    env.setdefault("CARGO_HOME", "/tmp/cargo")
    env["PATH"] = env["CARGO_HOME"] + "/bin:" + env.get("PATH", "")
    # A shared Python module must not inherit the executable's crt-static flags.
    env.pop("RUSTFLAGS", None)
    subprocess.run(["cargo", "build", "--release", "--offline", "--lib", "--features", "python-extension",
                    "--manifest-path", str(ROOT / "native/c4core/Cargo.toml")], env=env, check=True)
    binary = ROOT / "native/c4core/target/release/lib_c4solver.so"
    target = ROOT / "support/_c4solver.abi3.so"
    # A running bot may still map the previous extension. Replace the inode
    # atomically rather than truncating that process's loaded shared object.
    with tempfile.TemporaryDirectory(prefix=".native-build-", dir=target.parent) as stage:
        replacement = Path(stage) / target.name
        shutil.copy2(binary, replacement)
        os.replace(replacement, target)
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--offline", "--format-version", "1", "--features", "python-extension",
        "--filter-platform", "x86_64-unknown-linux-gnu",
        "--manifest-path", str(ROOT / "native/c4core/Cargo.toml")], env=env, text=True))
    notices = []
    for package in metadata["packages"]:
        directory = Path(package["manifest_path"]).parent
        if directory == ROOT / "native/c4core":
            continue
        notices.append("\n## " + package["name"] + " " + package["version"] +
                       " (" + str(package.get("license")) + ")\n")
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING")):
                notices.append(path.name + "\n" + path.read_text(errors="replace"))
    (ROOT / "THIRD_PARTY_LICENSES.txt").write_text("\n".join(notices))
    result = {"output": str(target), "bytes": target.stat().st_size,
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
