"""Install Stargazer's locked REBOUND release with correct ARM detection.

Run with the Python interpreter of the target environment; a C compiler is
required. The lockfile is resolved relative to this script, not the working
directory, so the same installer works in local and Docker environments.
"""

import hashlib
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import tomllib


def main():
    lock = tomllib.loads((Path(__file__).resolve().parents[1] / "uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == "rebound")
    source = package["sdist"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "rebound.tar.gz"
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            archive.write_bytes(response.read())
        digest = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != source["hash"]:
            raise RuntimeError("REBOUND source checksum does not match uv.lock")
        with tarfile.open(archive) as stream:
            stream.extractall(root, filter="data")
        project = root / f"rebound-{package['version']}"
        if platform.machine().lower() in {"aarch64", "arm64"}:
            setup = project / "setup.py"
            original = setup.read_text()
            old = 'return struct.calcsize("P")*8 == 64'
            if original.count(old) != 1:
                raise RuntimeError("Review the REBOUND ARM patch for this release")
            setup.write_text(
                original.replace(
                    old, 'return platform.machine().lower() in {"x86_64", "amd64"}'
                )
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                str(project),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
