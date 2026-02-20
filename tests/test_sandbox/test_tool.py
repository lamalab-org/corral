import json

import pytest

from corral.backend.tool import Tool
from corral.sandbox.config import SandboxConfig
from corral.sandbox.tool import (
    create_sandbox,
    create_sandbox_tools,
)


class TestCreateSandbox:
    def test_default_creates_docker(self):
        sb = create_sandbox()
        from corral.sandbox.docker_sandbox import DockerSandbox

        assert isinstance(sb, DockerSandbox)

    def test_subprocess_backend(self):
        config = SandboxConfig(backend="subprocess")
        sb = create_sandbox(config)
        from corral.sandbox.subprocess_sandbox import SubprocessSandbox

        assert isinstance(sb, SubprocessSandbox)

    def test_unknown_backend_raises(self):
        config = SandboxConfig(backend="unknown")
        with pytest.raises(ValueError, match="Unknown sandbox backend"):
            create_sandbox(config)


class TestCreateSandboxTools:
    def test_returns_sandbox_and_tools(self):
        config = SandboxConfig(backend="subprocess")
        sandbox, tools = create_sandbox_tools(config)
        assert sandbox is not None
        assert "execute_python_code" in tools
        assert "run_in_terminal" in tools

    def test_tools_are_tool_instances(self):
        config = SandboxConfig(backend="subprocess")
        _, tools = create_sandbox_tools(config)
        for tool in tools.values():
            assert isinstance(tool, Tool)


class TestSandboxCodeTool:
    def test_tool_name_and_args(self):
        config = SandboxConfig(backend="subprocess")
        _, tools = create_sandbox_tools(config)
        code_tool = tools["execute_python_code"]
        assert code_tool.name == "execute_python_code"
        arg_names = {a.name for a in code_tool.arguments}
        assert "python_code" in arg_names
        assert "timeout" in arg_names

    def test_execute_returns_json(self):
        config = SandboxConfig(backend="subprocess")
        sandbox, tools = create_sandbox_tools(config)
        code_tool = tools["execute_python_code"]

        with sandbox:
            output = code_tool.execute(python_code="result = 99")
            parsed = json.loads(output)
            assert parsed["success"] is True
            assert parsed["execution_result"]["result"] == 99

    def test_timeout_as_string(self):
        config = SandboxConfig(backend="subprocess")
        sandbox, tools = create_sandbox_tools(config)
        code_tool = tools["execute_python_code"]

        with sandbox:
            output = code_tool.execute(python_code="result = 1", timeout="60")
            parsed = json.loads(output)
            assert parsed["success"] is True


class TestSandboxTerminalTool:
    def test_tool_name(self):
        config = SandboxConfig(backend="subprocess")
        _, tools = create_sandbox_tools(config)
        terminal_tool = tools["run_in_terminal"]
        assert terminal_tool.name == "run_in_terminal"

    def test_execute_command(self):
        config = SandboxConfig(backend="subprocess")
        sandbox, tools = create_sandbox_tools(config)
        terminal_tool = tools["run_in_terminal"]

        with sandbox:
            output = terminal_tool.execute(command="echo tool_test")
            parsed = json.loads(output)
            assert parsed["success"] is True
            assert "tool_test" in parsed["stdout"]
