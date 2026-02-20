import json

from corral.sandbox.result import ExecResult


class TestExecResult:
    def test_success_result(self):
        result = ExecResult(
            success=True,
            stdout="hello",
            stderr="",
            return_code=0,
            execution_result={"result": 42},
        )
        assert result.success is True
        assert result.return_code == 0
        assert result.execution_result["result"] == 42

    def test_failure_result(self):
        result = ExecResult(
            success=False,
            error="something failed",
            return_code=1,
        )
        assert result.success is False
        assert result.error == "something failed"

    def test_timeout_result(self):
        result = ExecResult(
            success=False,
            timed_out=True,
            error="timed out",
        )
        assert result.timed_out is True

    def test_to_tool_result_format(self):
        """Verify JSON output matches existing execute_python_code format."""
        result = ExecResult(
            success=True,
            stdout="output line",
            stderr="",
            return_code=0,
            execution_result={"result": 10},
        )
        parsed = json.loads(result.to_tool_result())
        assert parsed["success"] is True
        assert parsed["stdout"] == "output line"
        assert parsed["stderr"] == ""
        assert parsed["return_code"] == 0
        assert parsed["execution_result"]["result"] == 10
        assert "timed_out" not in parsed

    def test_to_tool_result_with_error(self):
        result = ExecResult(
            success=False,
            error="bad code",
            timed_out=True,
        )
        parsed = json.loads(result.to_tool_result())
        assert parsed["success"] is False
        assert parsed["error"] == "bad code"
        assert parsed["timed_out"] is True
