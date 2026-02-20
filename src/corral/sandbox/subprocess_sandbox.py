"""When Docker is unavailable, we can use this."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from loguru import logger

from corral.sandbox._capture import generate_output_capture_code, parse_execution_output
from corral.sandbox._state import generate_state_restore_code, generate_state_save_code
from corral.sandbox.base import Sandbox
from corral.sandbox.config import SandboxConfig
from corral.sandbox.result import ExecResult


class SubprocessSandbox(Sandbox):
    """Enhanced subprocess sandbox with persistent state.

    Each sandbox instance gets a dedicated temporary working directory.
    Python variable state is persisted across ``execute()`` calls within
    the same sandbox session via pickle serialization.
    """

    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._work_dir: Path | None = None
        self._state_file: Path | None = None
        self._owns_work_dir: bool = False

    def start(self) -> None:
        if self._started:
            return

        if self.config.state_dir:
            self._work_dir = Path(self.config.state_dir).resolve()
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._owns_work_dir = False
        else:
            self._work_dir = Path(tempfile.mkdtemp(prefix="corral_sandbox_")).resolve()
            self._owns_work_dir = True

        if self.config.persistent_state:
            self._state_file = self._work_dir / ".sandbox_state.pkl"

        self._started = True
        logger.info(f"Started subprocess sandbox in {self._work_dir}")

    def stop(self) -> None:
        if self._work_dir and self._work_dir.exists() and self._owns_work_dir:
            # Only remove if the sandbox created its own temp dir
            shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir = None
        self._state_file = None
        self._owns_work_dir = False
        self._started = False

    def set_work_dir(self, work_dir: str | Path) -> None:
        """Update the sandbox working directory for subsequent executions.

        Switches the sandbox to execute code in *work_dir* and resets
        any persisted state so that trial state does not leak across
        trial boundaries.  The path is always resolved to an absolute
        path to avoid double-prefix issues when used as subprocess cwd.
        """
        new_dir = Path(work_dir).resolve()
        new_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir = new_dir
        # Environment manages this directory — don't delete it on stop()
        self._owns_work_dir = False
        # Reset persistent state for trial isolation
        if self.config.persistent_state:
            self._state_file = self._work_dir / ".sandbox_state.pkl"
        logger.info(f"Sandbox work_dir updated to {self._work_dir}")

    def execute(
        self,
        code: str,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        self._ensure_started()
        assert self._work_dir is not None

        timeout = timeout or self.config.resources.timeout_seconds
        start_time = time.monotonic()

        # Build the full script
        parts: list[str] = []

        if (
            self.config.persistent_state
            and self._state_file
            and self._state_file.exists()
        ):
            parts.append(generate_state_restore_code(str(self._state_file)))

        parts.append(code)
        parts.append(generate_output_capture_code())

        if self.config.persistent_state and self._state_file:
            parts.append(generate_state_save_code(str(self._state_file)))

        full_code = "\n".join(parts)

        # Write to temp file in the working directory
        script_path = self._work_dir / f"_exec_{uuid.uuid4().hex[:8]}.py"
        script_path.write_text(full_code)

        # Build process environment
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._work_dir),
                env=proc_env,
                check=False,
            )
            duration = time.monotonic() - start_time

            execution_result, output_lines = parse_execution_output(proc.stdout)

            stdout = "\n".join(output_lines)
            if len(stdout) > self.config.resources.max_output_bytes:
                stdout = (
                    stdout[: self.config.resources.max_output_bytes]
                    + "\n...[truncated]"
                )

            return ExecResult(
                success=proc.returncode == 0,
                stdout=stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                execution_result=execution_result,
                error=proc.stderr if proc.returncode != 0 else None,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start_time
            return ExecResult(
                success=False,
                error=f"Execution timed out after {timeout} seconds",
                timed_out=True,
                duration_seconds=duration,
            )
        finally:
            script_path.unlink(missing_ok=True)

    def execute_command(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
    ) -> ExecResult:
        self._ensure_started()
        assert self._work_dir is not None

        timeout = timeout or self.config.resources.timeout_seconds
        start_time = time.monotonic()

        work_cwd = cwd or str(self._work_dir)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_cwd,
                check=False,
            )
            duration = time.monotonic() - start_time

            stdout = proc.stdout
            if len(stdout) > self.config.resources.max_output_bytes:
                stdout = (
                    stdout[: self.config.resources.max_output_bytes]
                    + "\n...[truncated]"
                )

            return ExecResult(
                success=proc.returncode == 0,
                stdout=stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                error=proc.stderr if proc.returncode != 0 else None,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start_time
            return ExecResult(
                success=False,
                error=f"Command timed out after {timeout} seconds",
                timed_out=True,
                duration_seconds=duration,
            )

    def upload_file(self, local_path: str | Path, sandbox_path: str) -> None:
        self._ensure_started()
        assert self._work_dir is not None
        src = Path(local_path)
        dest = self._work_dir / sandbox_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    def download_file(self, sandbox_path: str, local_path: str | Path) -> None:
        self._ensure_started()
        assert self._work_dir is not None
        src = self._work_dir / sandbox_path
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    def _ensure_started(self) -> None:
        if not self._started:
            msg = "Sandbox is not started. Call start() first."
            raise RuntimeError(msg)
