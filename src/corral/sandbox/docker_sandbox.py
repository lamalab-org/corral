"""Docker-based sandbox: one long-lived container per trial."""

import shutil
import subprocess
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

_STATE_FILE_PATH = "/workspace/.sandbox_state.pkl"


class DockerSandbox(Sandbox):
    """Docker-based sandbox with a persistent container per trial.

    The container is created on ``start()`` and kept alive for the duration
    of the trial.  Each ``execute()`` call uses ``docker exec`` to run code
    inside the same container, preserving filesystem state.  Python variable
    state is persisted across calls via pickle.
    """

    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._container_name = f"corral_sandbox_{uuid.uuid4().hex[:12]}"
        self._container_id: str | None = None
        self._image_tag: str | None = None

    def start(self) -> None:
        if self._started:
            return

        if not shutil.which("docker"):
            msg = "Docker is not installed or not on PATH"
            raise RuntimeError(msg)

        image = self._resolve_image()

        cmd: list[str] = [
            "docker",
            "run",
            "-d",
            "--name",
            self._container_name,
            f"--cpus={self.config.resources.cpu_count}",
            f"--memory={self.config.resources.memory_mb}m",
            "-w",
            self.config.working_dir,
        ]

        # Volume mount for state dir if provided
        if self.config.state_dir:
            cmd.extend(["-v", f"{self.config.state_dir}:{self.config.working_dir}"])

        # Network: start with network for package installation
        # We'll disable it after setup if needed
        cmd.extend([image, "sleep", "infinity"])

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            msg = f"Failed to start Docker container: {result.stderr.strip()}"
            raise RuntimeError(msg)

        self._container_id = result.stdout.strip()
        self._started = True
        logger.info(f"Started sandbox container {self._container_name}")

        # Install packages while network is available
        self._install_packages()

        # Disable network after setup if policy says so
        if not self.config.network.allow_network:
            self._disable_network()

    def stop(self) -> None:
        if self._container_id:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                check=False,
            )
            logger.info(f"Stopped sandbox container {self._container_name}")
            self._container_id = None
        self._started = False

    def execute(
        self,
        code: str,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        self._ensure_started()
        timeout = timeout or self.config.resources.timeout_seconds
        start_time = time.monotonic()

        # Build the full script
        parts: list[str] = []

        if self.config.persistent_state:
            parts.append(generate_state_restore_code(_STATE_FILE_PATH))

        parts.append(code)
        parts.append(generate_output_capture_code())

        if self.config.persistent_state:
            parts.append(generate_state_save_code(_STATE_FILE_PATH))

        full_code = "\n".join(parts)

        # Write script to a local temp file then docker cp it in
        script_name = f"_exec_{uuid.uuid4().hex[:8]}.py"
        container_script = f"/tmp/{script_name}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(full_code)
            local_script = tmp.name

        try:
            # Copy script into container
            cp_result = subprocess.run(
                [
                    "docker",
                    "cp",
                    local_script,
                    f"{self._container_name}:{container_script}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp_result.returncode != 0:
                return ExecResult(
                    success=False,
                    error=f"Failed to copy script into container: {cp_result.stderr.strip()}",
                )

            # Execute inside container
            docker_cmd: list[str] = ["docker", "exec"]
            if env:
                for k, v in env.items():
                    docker_cmd.extend(["-e", f"{k}={v}"])
            docker_cmd.extend([self._container_name, "python", container_script])

            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
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
            Path(local_script).unlink(missing_ok=True)
            # Clean up script inside container
            subprocess.run(
                ["docker", "exec", self._container_name, "rm", "-f", container_script],
                capture_output=True,
                check=False,
            )

    def execute_command(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
    ) -> ExecResult:
        self._ensure_started()
        timeout = timeout or self.config.resources.timeout_seconds
        start_time = time.monotonic()

        docker_cmd: list[str] = ["docker", "exec"]
        if cwd:
            docker_cmd.extend(["-w", cwd])
        docker_cmd.extend([self._container_name, "sh", "-c", command])

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
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
        result = subprocess.run(
            ["docker", "cp", str(local_path), f"{self._container_name}:{sandbox_path}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            msg = f"upload_file failed: {result.stderr.strip()}"
            raise RuntimeError(msg)

    def download_file(self, sandbox_path: str, local_path: str | Path) -> None:
        self._ensure_started()
        result = subprocess.run(
            ["docker", "cp", f"{self._container_name}:{sandbox_path}", str(local_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            msg = f"download_file failed: {result.stderr.strip()}"
            raise RuntimeError(msg)

    def _ensure_started(self) -> None:
        if not self._started:
            msg = "Sandbox is not started. Call start() first."
            raise RuntimeError(msg)

    def _resolve_image(self) -> str:
        if self.config.dockerfile:
            tag = f"corral_sandbox:{uuid.uuid4().hex[:8]}"
            dockerfile_path = Path(self.config.dockerfile)
            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    tag,
                    "-f",
                    str(dockerfile_path),
                    str(dockerfile_path.parent),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                msg = f"Docker build failed: {result.stderr.strip()}"
                raise RuntimeError(msg)
            self._image_tag = tag
            return tag
        return self.config.docker_image

    def _install_packages(self) -> None:
        packages = list(self.config.python_packages)
        if not packages:
            return

        pkg_str = " ".join(packages)
        logger.info(f"Installing packages in sandbox: {pkg_str}")
        result = self.execute_command(
            f"pip install --quiet --disable-pip-version-check {pkg_str}",
            timeout=self.config.pip_install_timeout,
        )
        if not result.success:
            logger.warning(f"Package installation had issues: {result.stderr}")

    def _disable_network(self) -> None:
        """Disconnect the container from all networks."""
        # Find networks the container is connected to
        # TODO: Harbor does this ?
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{range $key, $val := .NetworkSettings.Networks}}{{$key}} {{end}}",
                self._container_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                f"Could not inspect container networks: {result.stderr.strip()}"
            )
            return

        networks = result.stdout.strip().split()
        for network in networks:
            if not network:
                continue
            disc = subprocess.run(
                [
                    "docker",
                    "network",
                    "disconnect",
                    "-f",
                    network,
                    self._container_name,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if disc.returncode == 0:
                logger.info(f"Disconnected sandbox from network: {network}")
            else:
                logger.warning(
                    f"Failed to disconnect from {network}: {disc.stderr.strip()}"
                )
