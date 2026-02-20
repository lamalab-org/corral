from abc import ABC, abstractmethod
from pathlib import Path

from corral.sandbox.config import SandboxConfig
from corral.sandbox.result import ExecResult


class Sandbox(ABC):
    """Abstract base for sandboxed code execution backends."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self._started = False

    @abstractmethod
    def start(self) -> None:
        """Initialize the sandbox.

        Called once per trial, not per execution bcause we want to keep for all subtasks the same sandbox.
        """

    @abstractmethod
    def stop(self) -> None:
        """Tear down the sandbox and clean up all resources."""

    @abstractmethod
    def execute(
        self,
        code: str,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute Python code in the sandbox.

        Parameters
        ----------
        code
            Python source code string to execute.
        timeout
            Override for the default timeout in seconds.
        env
            Additional environment variables for this execution.

        Returns
        -------
        ExecResult
        """

    @abstractmethod
    def execute_command(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
    ) -> ExecResult:
        """Execute a shell command in the sandbox.

        Parameters
        ----------
        command
            Shell command string.
        timeout
            Override timeout in seconds.
        cwd
            Working directory override.

        Returns
        -------
        ExecResult
        """

    @abstractmethod
    def upload_file(self, local_path: str | Path, sandbox_path: str) -> None:
        """Copy a file from the host into the sandbox."""

    @abstractmethod
    def download_file(self, sandbox_path: str, local_path: str | Path) -> None:
        """Copy a file from the sandbox to the host."""

    def set_work_dir(self, work_dir: str | Path) -> None:
        """Update the sandbox working directory for subsequent executions.

        Called when the environment's trial directory changes.  The default
        implementation is a no-op so that existing sandbox backends remain
        backward-compatible.
        """
        _ = work_dir  # no-op; subclasses may override

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
