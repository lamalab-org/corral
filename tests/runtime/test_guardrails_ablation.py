"""Offline end-to-end Docker ablation probe; run in the prepared trial image.

CORRAL_ABLATION_TESTS=1 python -m pytest tests/runtime/test_guardrails_ablation.py
Each case needs a fresh process: the protected runtime permanently changes
filesystem ownership and process permissions.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from corral.agents.schema import AgentOutcome
from corral.agents.session import AgentSessionCapabilities
from corral.core.action import Action
from corral.core.environment import Environment
from corral.core.task import TaskDefinition
from corral.orchestration import DockerSandboxSpec, SandboxProfile
from corral.orchestration.internal import run_task_from_files
from corral.orchestration.models import RunTaskInput
from corral.orchestration.registry import RuntimeRegistry
from corral.persistence import SQLiteCommitStore

pytestmark = pytest.mark.skipif(
    os.environ.get("CORRAL_ABLATION_TESTS") != "1",
    reason="requires a fresh prepared Docker trial container",
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class BoundaryProbeAgent:
    session_capabilities = AgentSessionCapabilities(native_workspace=True)

    async def run_session(self, session):
        # This same probe runs inside the worker or directly in the controller.
        import shutil

        assert all(shutil.which(name) for name in ("curl", "wget", "git"))
        root = Path(session.native_tool_workspace)
        try:
            Path("/opt/corral/pyproject.toml").read_text()
            answer = "readable"
        except (PermissionError, FileNotFoundError):
            answer = "denied"
        response = await session.execute(
            Action(
                name="terminal",
                arguments={
                    "command": "if test -r /opt/corral/pyproject.toml; then echo readable; else echo denied; fi"
                },
            )
        )
        assert response.success, response
        assert answer in str(response.result)
        (root / "native.txt").write_text(answer)
        await session.execute(
            Action(name="submit_answer", arguments={"answer": answer})
        )
        return AgentOutcome(status="completed", answer=answer)


def create_registry(request):
    task = TaskDefinition(
        name="ablation-probe",
        description="Probe container filesystem access.",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    return RuntimeRegistry(
        agents={"probe": BoundaryProbeAgent()},
        environments={
            "probe": Environment(
                "probe",
                task,
                base_work_dir="/workspace",
                task_execution_id="probe",
            )
        },
    )


@pytest.mark.anyio
async def test_docker_guardrails_private_reads_and_native_checkpoint():
    guardrails = os.environ["CORRAL_TEST_GUARDRAILS"]
    request = RunTaskInput(
        execution_id="ablation-probe",
        task_id="probe",
        environment_id="probe",
        agent_id="probe",
        max_iterations=5,
        sandbox=SandboxProfile(
            mode="docker",
            docker=DockerSandboxSpec(
                guardrails=guardrails,
                registry_module="tests.runtime.test_guardrails_ablation:create_registry",
            ),
        ),
    )
    checkpoint = Path("/corral-state")
    checkpoint.mkdir(mode=0o700, exist_ok=True)
    request_file = checkpoint / "request.json"
    request_file.write_text(json.dumps(asdict(request)))
    result_file = checkpoint / "result.json"
    assert await run_task_from_files(request_file, result_file) == 0
    result = json.loads(result_file.read_text())
    expected = "readable" if guardrails == "off" else "denied"
    assert result["submission"] == expected, result
    async with SQLiteCommitStore(
        checkpoint / "commits.sqlite3", "ablation-probe"
    ) as store:
        state = await store.materialize("main")
        assert "native.txt" in state.workspace.files
        assert state.task.scaffold["sandbox_guardrails"] == guardrails


@pytest.mark.anyio
async def test_claude_root_bypass_initializes_without_model_calls():
    if os.environ["CORRAL_TEST_GUARDRAILS"] != "off":
        pytest.skip("bypass startup is specific to the unconfined condition")
    pytest.importorskip("claude_agent_sdk")
    import anyio

    from corral.agents.claude_code import ClaudeCodeAgent, ClaudeSDKClient

    options = ClaudeCodeAgent()._build_options(
        server={},
        allowed_tools=[],
        enable_surrender=False,
        cwd="/workspace",
        max_turns=1,
        corral_sandbox=True,
        native_sandbox_disabled=True,
    )
    options.mcp_servers = {}
    options.env.update(
        {
            "CLAUDE_CONFIG_DIR": "/tmp/claude-probe",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "ANTHROPIC_API_KEY": "unused-offline-probe",
        }
    )
    with anyio.fail_after(30):
        async with ClaudeSDKClient(options=options) as client:
            info = await client.get_server_info()
            assert info is not None
