"""Host execution for explicitly selected local Corral MD tasks."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path

from corral_md.workspace import ASSET_DIRECTORIES, local_path


def assets_directory(value: str | Path | None = None) -> Path:
    """Locate assets prepared by modal_app/setup_assets.py --output-dir."""
    configured = value or os.getenv("CORRAL_MD_ASSETS_DIR")
    # The benchmark controller uses the host path. Docker trials use the same
    # immutable catalogs baked into the image under a worker-visible /usr tree.
    image_assets = Path("/usr/local/share/corral-md-assets")
    if (
        configured
        and not Path(configured).expanduser().exists()
        and image_assets.is_dir()
    ):
        configured = image_assets
    if not configured:
        raise ValueError(
            "Local MD mode needs CORRAL_MD_ASSETS_DIR pointing to prepared assets; "
            "run modal_app/setup_assets.py --output-dir first"
        )
    root = Path(configured).expanduser().resolve()
    missing = sorted(kind for kind in ASSET_DIRECTORIES if not (root / kind).is_dir())
    if missing:
        raise FileNotFoundError(
            f"Local MD asset catalogs missing from {root}: {', '.join(missing)}"
        )
    return root


def run_lammps_locally(
    workspace: str | Path, input_file: str, *, assets_dir: str | Path
) -> Path:
    """Run the host LAMMPS executable with task paths mapped to host paths."""
    root = Path(workspace).resolve()
    source = local_path(root, input_file)
    if not source.is_file():
        raise FileNotFoundError(f"LAMMPS input not found: {input_file}")
    executable = shutil.which(os.getenv("CORRAL_MD_LAMMPS_BIN", "lmp"))
    if executable is None:
        raise FileNotFoundError(
            "Local LAMMPS executable not found; set CORRAL_MD_LAMMPS_BIN"
        )
    assets = assets_directory(assets_dir)
    output = local_path(root, "/workspace/output", allow_root=True)
    output.mkdir(exist_ok=True)
    log = output / f"{source.stem}.log"
    # LAMMPS scripts commonly contain public absolute paths. Run a temporary
    # copy so the original remains an unchanged benchmark artifact.
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"(?im)^\s*log\s+[^\n]*$", "", text)
    for kind in sorted(ASSET_DIRECTORIES):
        text = text.replace(f"/workspace/{kind}/", f"{assets / kind}/")
    text = text.replace("/workspace/", f"{root}/")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".in",
        prefix=".corral-md-",
        dir=root,
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        with (
            tempfile.TemporaryFile() as stdout_stream,
            tempfile.TemporaryFile() as stderr_stream,
        ):
            process = subprocess.Popen(
                [executable, "-in", str(temporary), "-log", str(log)],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                timed_out = True
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            stdout_stream.seek(0)
            stderr_stream.seek(0)
            with (output / f"{source.stem}.stdout.txt").open("wb") as saved_stdout:
                shutil.copyfileobj(stdout_stream, saved_stdout)
            with (output / f"{source.stem}.stderr.txt").open("wb") as saved_stderr:
                shutil.copyfileobj(stderr_stream, saved_stderr)
            stderr_stream.seek(max(0, stderr_stream.tell() - 4000))
            error_tail = stderr_stream.read().decode("utf-8", errors="replace")
    finally:
        temporary.unlink(missing_ok=True)
    if timed_out:
        raise TimeoutError("LAMMPS exceeded its 7200-second local timeout")
    if process.returncode:
        raise ValueError(
            f"LAMMPS failed with exit code {process.returncode}: {error_tail}"
        )
    return log
