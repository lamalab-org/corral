import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecResult:
    """Result of a sandboxed execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    execution_result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timed_out: bool = False
    duration_seconds: float | None = None

    def to_tool_result(self) -> str:
        """Convert to JSON string matching the existing execute_python_code output format."""
        d: dict[str, Any] = {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_result": self.execution_result,
        }
        if self.error:
            d["error"] = self.error
        if self.timed_out:
            d["timed_out"] = True
        return json.dumps(d, indent=2)
