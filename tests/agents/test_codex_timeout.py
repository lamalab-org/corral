"""Fixed MCP timeout configuration on the standard Codex SDK path."""

import anyio
import pytest
import tomllib

pytest.importorskip("openai_codex")

from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
from openai_codex.errors import CodexError
from openai_codex.generated.v2_all import McpServerToolCallResponse

from corral.agents.codex import CodexAgent, _RunState
from corral.backend.mcp import open_mcp_host
from corral.core.tool import ToolResponse
from corral.core.tool_catalog import ToolCatalogSnapshot


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("timeout", [None, 30.5])
@pytest.mark.parametrize("native_workspace", [False, True])
@pytest.mark.parametrize("api_endpoint", [None, "https://openrouter.ai/api/v1"])
def test_standard_client_registers_all_tools_with_fixed_timeout(
    monkeypatch, tmp_path, timeout, native_workspace, api_endpoint
):
    captured = {}
    thread = object()

    class FakeCodex:
        def __init__(self, *, config):
            captured["config"] = config

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def login_api_key(self, _key):
            pass

        def thread_start(self, **kwargs):
            captured["thread"] = kwargs
            return thread

    def drive_turn(self, actual_thread, prompt, run):
        assert actual_thread is thread
        assert prompt == "solve"
        return "done"

    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setattr("corral.agents.codex.Codex", FakeCodex)
    monkeypatch.setattr(CodexAgent, "_drive_turn", drive_turn)
    agent = CodexAgent(
        system_prompt="system",
        api_endpoint=api_endpoint,
        **({"tool_timeout_s": timeout} if timeout else {}),
    )
    run = _RunState(native_workspace=str(tmp_path) if native_workspace else None)
    assert (
        agent._execute_codex_turn(
            codex_home=tmp_path,
            workspace=tmp_path,
            mcp_url="http://unused/mcp",
            tool_names=["terminal", "submit_answer"],
            developer_instructions="system",
            prompt="solve",
            run=run,
        )
        == "done"
    )
    config = (tmp_path / "config.toml").read_text()
    assert tomllib.loads(config).get("openai_base_url") == api_endpoint
    assert 'enabled_tools = ["terminal", "submit_answer"]' in config
    assert f"tool_timeout_sec = {timeout if timeout else 10000}" in config
    assert captured["config"].env["CODEX_HOME"] == str(tmp_path)
    assert captured["thread"] == {
        "model": agent.harness_model,
        "cwd": str(tmp_path),
        "developer_instructions": "system",
        "approval_mode": ApprovalMode.auto_review,
        "sandbox": None if native_workspace else Sandbox.read_only,
        "ephemeral": True,
    }


def test_metadata_and_prompt_describe_fixed_sdk_wait():
    agent = CodexAgent(system_prompt="system")
    tools = [
        {
            "function": {
                "name": "terminal",
                "parameters": {"properties": {"timeout_seconds": {"default": 120}}},
            }
        }
    ]
    metadata = agent._harness_metadata("http://unused/mcp", "brief", tools, None, 1)
    assert metadata["tool_timeout_s"] == metadata["effective_tool_timeout_s"] == 10_000
    assert metadata["tool_timeout_policy"] == "fixed_sdk_wait"
    assert "10000 seconds" in agent._developer_instructions(False)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_tool_timeout_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        CodexAgent(tool_timeout_s=value)


@pytest.mark.anyio
async def test_pinned_runtime_uses_fixed_mcp_timeout_and_recovers(tmp_path):
    """Exercise the bundled runtime's MCP client without requesting a model turn."""
    schema = {
        "type": "function",
        "function": {
            "name": "measure",
            "description": "A measurement whose arguments must not change the SDK limit",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "number"},
                    "timeout_seconds": {"type": "number", "default": 0.01},
                },
            },
        },
    }
    calls = []

    async def execute(action):
        calls.append(dict(action.arguments))
        await anyio.sleep(action.arguments["duration"])
        return ToolResponse(success=True, result="measured", error=None)

    async with (
        open_mcp_host() as host,
        host.bind(
            catalog=ToolCatalogSnapshot.capture([schema]), execute=execute
        ) as endpoint,
    ):

        def call_tools():
            agent = CodexAgent(system_prompt="system", tool_timeout_s=1)
            (tmp_path / "config.toml").write_text(
                agent._render_config_toml(endpoint.url, ["measure"])
            )
            with Codex(
                config=CodexConfig(cwd=str(tmp_path), env={"CODEX_HOME": str(tmp_path)})
            ) as codex:
                codex.login_api_key("offline-test-key")
                thread = codex.thread_start(
                    model="gpt-5.4", cwd=str(tmp_path), ephemeral=True
                )

                def invoke(arguments):
                    return codex._client.request(
                        "mcpServer/tool/call",
                        {
                            "threadId": thread.id,
                            "server": "corral",
                            "tool": "measure",
                            "arguments": arguments,
                        },
                        response_model=McpServerToolCallResponse,
                    )

                result = invoke({"duration": 0.15, "timeout_seconds": 0.01})
                assert not result.is_error
                assert result.content[0]["text"] == "measured"
                with pytest.raises(CodexError, match="(?i)timed out|timeout"):
                    invoke({"duration": 1.2, "timeout_seconds": 60})
                assert not invoke({"duration": 0}).is_error

        await anyio.to_thread.run_sync(call_tools)
    assert calls == [
        {"duration": 0.15, "timeout_seconds": 0.01},
        {"duration": 1.2, "timeout_seconds": 60},
        {"duration": 0},
    ]
