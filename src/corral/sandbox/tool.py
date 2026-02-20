"""Corral Tool wrappers for sandboxed execution and factory functions."""

from corral.backend.schema import ToolArgument
from corral.backend.tool import Tool
from corral.sandbox.base import Sandbox
from corral.sandbox.config import SandboxConfig
from corral.sandbox.docker_sandbox import DockerSandbox
from corral.sandbox.subprocess_sandbox import SubprocessSandbox


class SandboxCodeTool(Tool):
    """Tool that executes Python code inside a sandbox."""

    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="execute_python_code",
            description=(
                "[BRIEF] Execute Python code in a sandboxed environment with "
                "persistent state across calls. [/BRIEF]"
                "[DETAILED] Runs arbitrary Python code inside an isolated sandbox. "
                "Variables and imports persist between calls within the same trial, "
                "so you can build up analysis incrementally. Output is captured "
                "automatically — assign your main result to a variable named "
                "'result' or 'output' for best capture. Standard scientific packages "
                "(numpy, scipy, pandas, etc.) are available if configured. [/DETAILED]"
            ),
            arguments=[
                ToolArgument(
                    name="python_code",
                    type="str",
                    description=(
                        "Python code string to execute. Assign your main output to "
                        "a variable named 'result' or 'output' for best capture."
                    ),
                    required=True,
                ),
                ToolArgument(
                    name="timeout",
                    type="int",
                    description="Maximum execution time in seconds. Defaults to 300.",
                    required=False,
                    default=300,
                ),
            ],
        )
        self._sandbox = sandbox

    def execute(self, **kwargs) -> str:
        code = kwargs["python_code"]
        timeout = kwargs.get("timeout", 300)
        if isinstance(timeout, str):
            try:
                timeout = int(float(timeout))
            except (ValueError, TypeError):
                timeout = 300
        result = self._sandbox.execute(code, timeout=timeout)
        return result.to_tool_result()


class SandboxTerminalTool(Tool):
    """Tool that executes shell commands inside a sandbox."""

    def __init__(self, sandbox: Sandbox):
        super().__init__(
            name="run_in_terminal",
            description=(
                "[BRIEF] Execute shell commands in a sandboxed terminal. [/BRIEF]"
                "[DETAILED] Runs a shell command inside the sandbox environment. "
                "Useful for file manipulation, running scripts, or using CLI tools "
                "available in the sandbox. [/DETAILED]"
            ),
            arguments=[
                ToolArgument(
                    name="command",
                    type="str",
                    description="Shell command to execute.",
                    required=True,
                ),
                ToolArgument(
                    name="timeout",
                    type="int",
                    description="Maximum execution time in seconds. Defaults to 300.",
                    required=False,
                    default=300,
                ),
            ],
        )
        self._sandbox = sandbox

    def execute(self, **kwargs) -> str:
        command = kwargs["command"]
        timeout = kwargs.get("timeout", 300)
        if isinstance(timeout, str):
            try:
                timeout = int(float(timeout))
            except (ValueError, TypeError):
                timeout = 300
        result = self._sandbox.execute_command(command, timeout=timeout)
        return result.to_tool_result()


_BACKENDS: dict[str, type[Sandbox]] = {
    "docker": DockerSandbox,
    "subprocess": SubprocessSandbox,
}


def create_sandbox(config: SandboxConfig | None = None) -> Sandbox:
    """Create a sandbox instance from configuration.

    Parameters
    ----------
    config
        Sandbox configuration. Uses defaults (Docker backend) when *None*.

    Returns
    -------
    Sandbox
        An unstarted sandbox instance. Call ``start()`` before use.
    """
    if config is None:
        config = SandboxConfig()

    cls = _BACKENDS.get(config.backend)
    if cls is None:
        msg = (
            f"Unknown sandbox backend: {config.backend!r}. "
            f"Available: {sorted(_BACKENDS)}"
        )
        raise ValueError(msg)

    return cls(config)


def create_sandbox_tools(
    config: SandboxConfig | None = None,
) -> tuple[Sandbox, dict[str, Tool]]:
    """Create a sandbox and its associated Corral tools.

    Parameters
    ----------
    config
        Sandbox configuration. Uses defaults when *None*.

    Returns
    -------
    tuple[Sandbox, dict[str, Tool]]
        ``(sandbox, {"execute_python_code": ..., "run_in_terminal": ...})``.
        The caller owns the sandbox lifecycle (must call ``start()`` / ``stop()``).
    """
    sandbox = create_sandbox(config)
    tools: dict[str, Tool] = {
        "execute_python_code": SandboxCodeTool(sandbox),
        "run_in_terminal": SandboxTerminalTool(sandbox),
    }
    return sandbox, tools
