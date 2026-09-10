"""Exercise real Unix permission checks in the trial image (no model API calls).

Run with CORRAL_PERMISSION_TESTS=1 inside the Docker image. Opting in makes
missing root privileges or an incorrectly protected image a failure, not a skip.
"""

# Exercise imports after privilege dropping; SDK extras are optional locally.
# ruff: noqa: PLC0415

import os
import threading

import pytest

from corral.agents.ai_scientist.agent import AIScientistAgent
from corral.agents.session import AgentSessionCapabilities
from corral.core.environment import Environment
from corral.core.tool import tool
from corral.runtime import permissions

pytestmark = pytest.mark.skipif(
    os.environ.get("CORRAL_PERMISSION_TESTS") != "1",
    reason="requires the real Docker trial permission boundary",
)


@pytest.fixture(autouse=True)
def restricted_runtime(tmp_path):
    assert os.geteuid() == 0
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(mode=0o700)
    permissions.configure(checkpoints)
    yield
    permissions._enabled = False


@pytest.fixture()
def workspace():
    import tempfile

    return tempfile.mkdtemp(prefix="permission-test-", dir="/workspace")


@tool
def permission_probe() -> str:
    """Attempt reads and writes both directly and through detached descendants."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    result = {"uid": os.getuid(), "gids": os.getgroups(), "denials": {}}
    for target in (
        "/opt/corral/pyproject.toml",
        "/opt/corral/tasks",
        "/corral-state",
        "/etc/passwd",
        "/proc/1/environ",
    ):
        try:
            with open(target, "rb"):
                pass
        except (PermissionError, FileNotFoundError) as exc:
            result["denials"][target] = str(exc)
    for target in ("/opt/escape", "/escape"):
        try:
            Path(target).write_text("escaped")
        except (PermissionError, OSError) as exc:
            result["denials"][target] = str(exc)
    Path("allowed.txt").write_text("workspace access")
    result["workspace"] = Path("allowed.txt").read_text()
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,pathlib; os.setsid(); pathlib.Path('/opt/corral/pyproject.toml').read_text()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result["child_error"] = child.stderr
    result["child_returncode"] = child.returncode
    try:
        os.setuid(0)
    except PermissionError:
        result["cannot_regain_root"] = True
    Path("source-link").symlink_to("/opt/corral/pyproject.toml")
    try:
        Path("source-link").read_text()
    except PermissionError:
        result["symlink_denied"] = True
    finally:
        Path("source-link").unlink()
    return json.dumps(result)


def assert_probe(result):
    import json

    probe = json.loads(result)
    assert probe["uid"] != 0
    assert probe["gids"] == []
    assert len(probe["denials"]) == 7
    assert probe["workspace"] == "workspace access"
    assert probe["child_returncode"] != 0
    assert "Permission denied" in probe["child_error"]
    assert probe["cannot_regain_root"]
    assert probe["symlink_denied"]


def test_foreground_tool_and_descendants(workspace):
    environment = object.__new__(Environment)
    environment.workspace_path = workspace
    assert_probe(permissions.execute_tool(environment, None, permission_probe, {}))


def test_terminal_inherits_worker_permissions(workspace):
    import json
    from pathlib import Path

    from corral.workspace import WorkspaceFilesystem, build_terminal_tool

    environment = object.__new__(Environment)
    environment.workspace_path = workspace
    terminal = build_terminal_tool(WorkspaceFilesystem(workspace))
    result = json.loads(
        permissions.execute_tool(
            environment,
            None,
            terminal,
            {
                "command": "id -u; printf 'workspace access' > allowed.txt; cat /opt/corral/pyproject.toml"
            },
        )
    )

    assert int(result["output"].splitlines()[0]) != 0
    assert result["exit_code"] != 0
    assert "Permission denied" in result["output"]
    assert Path(workspace, "allowed.txt").read_text() == "workspace access"


@tool
def filesystem_root_probe() -> str:
    """Probe unpublished paths, runtime aliases, and descriptor escapes."""
    import ctypes
    import errno
    import json
    import os
    import sys
    from pathlib import Path

    runtime = Path("/workspace/.corral-runtime-system")
    for alias in (
        "/usr",
        "/bin/sh",
        "/dev/null",
        "/proc",
        "/etc/hosts",
        sys.executable,
    ):
        assert Path(alias).resolve(strict=True).is_relative_to(runtime), alias
    for target in (
        "/outside-workspace.txt",
        "/opt/unrelated/data.txt",
        "/proc/1/root/outside-workspace.txt",
        "/proc/self/root/outside-workspace.txt",
        "/etc/hostname",
    ):
        try:
            Path(target).read_bytes()
        except (PermissionError, FileNotFoundError):
            pass
        else:
            raise AssertionError(f"original image path is visible: {target}")

    # Native libc calls must observe the same boundary as Python file APIs.
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.open(b"/outside-workspace.txt", os.O_RDONLY) == -1
    assert ctypes.get_errno() in {errno.EACCES, errno.ENOENT}
    link = Path("outside-link")
    link.symlink_to("/outside-workspace.txt")
    try:
        try:
            link.read_bytes()
        except (PermissionError, FileNotFoundError):
            pass
        else:
            raise AssertionError("workspace symlink exposed the original root")
    finally:
        link.unlink()

    # No inherited log/directory handle may give access to a controller file.
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            target = descriptor.readlink()
        except FileNotFoundError:
            continue
        assert not str(target).startswith(
            ("/corral-state", "/opt/corral", "/tmp/")
        ), target
    for alias in ("/tmp", "/dev/shm"):
        target = Path(alias).resolve(strict=True)
        assert target.is_relative_to(Path.cwd())
        (target / "scratch-check").write_text("scratch")
    with pytest.raises(OSError) as error:
        Path("/usr/corral-write-check").write_text("forbidden")
    assert error.value.errno in {errno.EACCES, errno.EROFS}
    return json.dumps({"isolated_root": True})


def test_original_container_paths_are_not_mounted(workspace):
    import json
    from pathlib import Path

    # These sentinels are read-only mounts provided by the Docker test runner.
    assert Path("/outside-workspace.txt").is_file()
    assert Path("/opt/unrelated/data.txt").is_file()
    environment = object.__new__(Environment)
    environment.workspace_path = workspace
    result = permissions.execute_tool(environment, None, filesystem_root_probe, {})
    assert json.loads(result) == {"isolated_root": True}


def test_bootstrap_exit_kills_its_worker(tmp_path):
    import signal
    import time
    from contextlib import suppress
    from pathlib import Path

    from corral.runtime._worker_filesystem import bind_bootstrap_parent

    ready = tmp_path / "bootstrap-worker.pid"
    bootstrap = os.fork()
    if bootstrap == 0:
        parent = os.getpid()
        child = os.fork()
        if child == 0:
            bind_bootstrap_parent(parent)
            temporary = ready.with_suffix(".tmp")
            temporary.write_text(str(os.getpid()))
            temporary.replace(ready)
        while True:
            signal.pause()
    worker = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            if time.monotonic() > deadline:
                pytest.fail("bootstrap worker did not start")
            time.sleep(0.01)
        worker = int(ready.read_text())
        os.kill(bootstrap, signal.SIGKILL)
        os.waitpid(bootstrap, 0)
        bootstrap = None
        while time.monotonic() < deadline:
            try:
                status = Path(f"/proc/{worker}/status").read_text()
            except FileNotFoundError:
                return
            if status.split("State:", 1)[1].split()[0] == "Z":
                return
            time.sleep(0.01)
        pytest.fail("worker survived the trusted bootstrap")
    finally:
        for pid in (bootstrap, worker):
            if pid is not None:
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        if bootstrap is not None:
            os.waitpid(bootstrap, 0)


def test_filesystem_setup_requires_privileges(workspace, tmp_path):
    import errno

    from corral.runtime._worker_filesystem import enter_workspace

    child = os.fork()
    if child == 0:
        os.setgroups([])
        os.setresgid(60000, 60000, 60000)
        os.setresuid(60000, 60000, 60000)
        try:
            enter_workspace(
                workspace, tmp_path / "jail", 60000, 60000, keep_fds={0, 1, 2}
            )
        except OSError as exc:
            os._exit(0 if exc.errno == errno.EPERM else 1)
        os._exit(1)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert not (tmp_path / "jail").exists()


@pytest.mark.parametrize("executor_name", ["thread", "process", "subprocess"])
def test_background_executor_preferences_cannot_bypass_permissions(
    workspace, executor_name
):
    from corral.backend.executors import JobWork, RestrictedExecutor
    from corral.backend.jobs import JobManager

    permission_probe.executor = executor_name
    manager = JobManager()
    executor = manager._resolve_executor(permission_probe)
    assert isinstance(executor, RestrictedExecutor)
    work = JobWork(
        job_id="probe",
        tool_name=permission_probe.name,
        tool=permission_probe,
        call_arguments={},
        workspace=workspace,
    )
    assert_probe(executor.run_tool(work, threading.Event()))
    manager.shutdown()


class ProbeAgent:
    tool_transport = "python"

    async def run_session(self, session):
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        from corral.agents.schema import AgentOutcome
        from corral.core.action import Action

        assert os.getuid() != 0
        try:
            Path("/opt/corral/tasks/wetlab/wetlab/env.py").read_text()
        except PermissionError:
            pass
        else:
            raise AssertionError("agent could inspect task source")
        child = subprocess.run(  # noqa: ASYNC221 - intentionally test synchronous SDK-style spawning
            [sys.executable, "-c", "import os; print(os.getuid())"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(child.stdout) == os.getuid()
        answer = json.dumps({"uid": os.getuid(), "child_uid": int(child.stdout)})
        await session.execute(
            Action(name="submit_answer", arguments={"answer": answer})
        )
        return AgentOutcome(status="completed", answer=answer)


class DelegatingProbeAgent:
    tool_transport = "python"
    session_capabilities = AgentSessionCapabilities(inspect_subagents=True)

    async def run_session(self, session):
        import os
        from pathlib import Path

        from corral.agents.schema import AgentOutcome
        from corral.core.action import Action

        class Child:
            async def run_session(self, child_session):
                assert os.getuid() != 0
                try:
                    Path("/opt/corral/pyproject.toml").read_text()
                except PermissionError:
                    return AgentOutcome(status="completed", answer="denied")
                raise AssertionError("subagent could inspect source")

        child_id = await session.spawn_subagent(Child(), handoff="probe")
        outcome = await session.wait_for_subagent(child_id)
        assert outcome.answer == "denied"
        await session.execute(
            Action(name="submit_answer", arguments={"answer": "denied"})
        )
        return AgentOutcome(status="completed", answer="denied")


def native_agent_with_hooks():
    from corral.agents.hooks import HookPoint
    from corral.agents.tool_calling import ToolCallingAgent

    def prepare(context):
        import os
        from pathlib import Path
        from types import SimpleNamespace

        import corral.agents.tool_calling as native

        assert os.getuid() != 0
        with pytest.raises(PermissionError):
            Path("/opt/corral/pyproject.toml").read_text()

        async def reply(*args, **kwargs):
            return SimpleNamespace(
                content=None,
                usage=None,
                tool_calls=[
                    {
                        "id": "submission",
                        "function": {
                            "name": "submit_answer",
                            "arguments": '{"answer":"native loop passed"}',
                        },
                    }
                ],
            )

        native.call_model = reply
        context.metadata["worker_uid"] = os.getuid()

    agent = ToolCallingAgent(model="offline-test")
    agent.hooks.register(HookPoint.BEFORE_TASK, prepare)
    return agent


def planner_agent_with_hooks():
    from corral.agents.hooks import HookPoint
    from corral.agents.llm_planner import LLMPlanner

    def prepare(context):
        import os
        from pathlib import Path
        from types import SimpleNamespace

        import corral.agents.base_agent as native

        assert os.getuid() != 0
        with pytest.raises(PermissionError):
            Path("/opt/corral/src/corral/agents/prompts/index.json").read_text()
        content = (
            "Use the executor to submit the answer."
            if isinstance(context.agent, LLMPlanner)
            else (
                "<action>submit_answer</action>"
                '<action_input>{"answer":"planner delegate passed"}</action_input>'
            )
        )

        async def reply(*args, **kwargs):
            return SimpleNamespace(content=content, usage=None)

        native.llm_call = reply

    agent = LLMPlanner(model="offline-test")
    agent.hooks.register(HookPoint.BEFORE_TASK, prepare)
    agent._executor.hooks.register(HookPoint.BEFORE_TASK, prepare)
    return agent


class DeserializeProbeAgent:
    async def run_session(self, session):
        from corral.agents.schema import AgentOutcome
        from corral.core.action import Action

        class Delegate:
            def __reduce__(self):
                return eval, (
                    "__import__('pathlib').Path('/opt/corral/pyproject.toml').read_text()",
                )

            async def run_session(self, child_session):
                raise AssertionError("deserialization should have been denied")

        with pytest.raises(RuntimeError, match="Permission denied"):
            await session.run_delegate(Delegate())
        await session.execute(
            Action(name="submit_answer", arguments={"answer": "denied"})
        )
        return AgentOutcome(status="completed", answer="denied")


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio()
@pytest.mark.parametrize(
    "agent_type",
    [
        ProbeAgent,
        DelegatingProbeAgent,
        native_agent_with_hooks,
        planner_agent_with_hooks,
        DeserializeProbeAgent,
    ],
)
async def test_agent_and_subagent_execution(workspace, tmp_path, agent_type):
    from datetime import datetime, timezone

    from corral.core.task import TaskDefinition
    from corral.observability import NoOpObserver
    from corral.persistence import SQLiteCommitStore
    from corral.runtime.task_runner import TaskRuntime

    task = TaskDefinition(
        name="probe",
        description="permission probe",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    environment = Environment(
        "probe", task, base_work_dir=workspace, task_execution_id="probe"
    )
    store = SQLiteCommitStore(tmp_path / "state.sqlite3", execution_id="probe")
    try:
        state = await TaskRuntime(store, NoOpObserver()).run(
            agent_type(),
            environment,
            execution_id="probe",
            max_iterations=3,
            started_at=datetime.now(timezone.utc),
        )
        assert state.submission is not None, state.runtime.model_dump()
    finally:
        await store.aclose()


@tool
def sdk_permission_probe(sdk: str) -> str:
    """Start actual SDK runtimes without making a model request."""
    import asyncio
    import json
    import os
    from pathlib import Path

    workspace = os.getcwd()
    uid = os.getuid()

    def assert_identity(pid):
        status = Path(f"/proc/{pid}/status").read_text()
        assert int(status.split("Uid:", 1)[1].split()[0]) == uid

    command = ["/bin/sh", "-c", "id -u; cat /opt/corral/pyproject.toml"]
    if sdk == "codex":
        from openai_codex import Codex, CodexConfig

        home = Path(workspace) / "codex-home"
        home.mkdir()
        with Codex(CodexConfig(cwd=workspace, env={"CODEX_HOME": str(home)})) as client:
            assert_identity(client._client._proc.pid)
            result = client._client._request_raw(
                "command/exec",
                {
                    "command": command,
                    "cwd": workspace,
                    "sandboxPolicy": {
                        "type": "externalSandbox",
                        "networkAccess": "enabled",
                    },
                    "timeoutMs": 10000,
                },
            )
            assert "Permission denied" in result["stderr"], result
            assert str(uid) in result["stdout"], result
    elif sdk == "claude":
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        async def check():
            async with ClaudeSDKClient(
                ClaudeAgentOptions(cwd=workspace, setting_sources=[])
            ) as client:
                assert await client.get_server_info()
                assert_identity(client._transport._process.pid)

        asyncio.run(check())
    elif sdk == "openhands":
        from openhands.tools.terminal import TerminalAction, TerminalExecutor

        executor = TerminalExecutor(working_dir=workspace)
        try:
            result = executor(
                TerminalAction(command="id -u; cat /opt/corral/pyproject.toml")
            )
            rendered = result.model_dump_json()
            assert "Permission denied" in rendered, rendered
            assert str(uid) in rendered, rendered
        finally:
            executor.close()
    return json.dumps({"sdk": sdk, "uid": uid})


@pytest.mark.skipif(
    os.environ.get("CORRAL_PERMISSION_SDK_TESTS") != "1",
    reason="requires all pinned SDK extras",
)
@pytest.mark.parametrize("sdk", ["codex", "claude", "openhands"])
def test_real_sdk_runtime(workspace, sdk):
    environment = object.__new__(Environment)
    environment.workspace_path = workspace
    permissions.execute_tool(environment, None, sdk_permission_probe, {"sdk": sdk})


def test_snapshot_rejects_a_symlink_swap(workspace, tmp_path, monkeypatch):
    import errno
    from pathlib import Path

    source = Path(workspace)
    entry = source / "data.txt"
    entry.write_text("safe")
    target = tmp_path / "snapshot"
    target.mkdir()
    original_open = os.open

    def swap(path, *args, **kwargs):
        if path == "data.txt":
            entry.unlink()
            entry.symlink_to("/opt/corral/pyproject.toml")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    with pytest.raises(OSError) as exc:
        permissions.copy_workspace(source, target)
    assert exc.value.errno == errno.ELOOP
    assert not (target / "data.txt").exists()


@tool
def detached_process() -> str:
    """Leave a detached descendant to check worker cleanup."""
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
    )
    return str(child.pid)


def test_detached_descendant_is_stopped(workspace):
    from pathlib import Path

    environment = object.__new__(Environment)
    environment.workspace_path = workspace
    pid = permissions.execute_tool(environment, None, detached_process, {})
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return
    assert status.split("State:", 1)[1].split()[0] == "Z"


@tool
def forbidden_read() -> str:
    """Read protected task code and let the permission error reach the caller."""
    from pathlib import Path

    return Path("/opt/corral/pyproject.toml").read_text()


def test_denial_message_reaches_the_tool_caller(workspace):
    environment = object.__new__(Environment)
    environment.workspace_path = workspace
    with pytest.raises(RuntimeError, match="Permission denied:.*not permitted"):
        permissions.execute_tool(environment, None, forbidden_read, {})


def test_sdk_scratch_is_not_a_task_artifact(workspace, tmp_path):
    from pathlib import Path

    source = Path(workspace)
    scratch = source / f"{permissions.SCRATCH_PREFIX}sdk"
    scratch.mkdir()
    (scratch / "sdk-executable").symlink_to("/bin/sh")
    (source / "answer.txt").write_text("answer")
    destination = tmp_path / "snapshot"
    destination.mkdir()
    assert permissions.copy_workspace(source, destination) == ["answer.txt"]
    assert not (destination / scratch.name).exists()


@tool
def sibling_probe(path: str) -> str:
    """Try to read another assigned workspace."""
    from pathlib import Path

    return Path(path).read_text()


def test_workers_cannot_read_a_different_workspace(workspace):
    import tempfile
    from pathlib import Path

    other = Path(tempfile.mkdtemp(prefix="other-workspace-", dir="/workspace"))
    (other / "answer.txt").write_text("hidden")
    permissions.workspace_identity(other)
    environment = object.__new__(Environment)
    environment.workspace_path = workspace
    with pytest.raises(RuntimeError, match="Permission denied|No such file"):
        permissions.execute_tool(
            environment, None, sibling_probe, {"path": str(other / "answer.txt")}
        )


def test_worker_rejects_replaced_workspace_before_mount(workspace, monkeypatch):
    from pathlib import Path

    original_popen = permissions.subprocess.Popen
    root = Path(workspace)
    original = root.with_name(root.name + "-original")

    def replace_before_bootstrap(*args, **kwargs):
        root.rename(original)
        root.mkdir()
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(permissions.subprocess, "Popen", replace_before_bootstrap)
    with pytest.raises(RuntimeError, match="workspace was replaced"):
        permissions.run_worker("tool", (permission_probe, {}), workspace)
    assert not (root / "allowed.txt").exists()
    assert not (original / "allowed.txt").exists()


@tool(background_capable=True)
def node_boundary_probe(parent_file: str, sibling_file: str) -> str:
    """Verify native and subprocess access is limited to this node's directory."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    assert os.getuid() != 0
    Path("own.txt").unlink(missing_ok=True)
    Path("own.txt").write_text("node data")
    for target in (
        parent_file,
        sibling_file,
        "/opt/corral/pyproject.toml",
        "/corral-state/commits.sqlite3",
    ):
        with pytest.raises((PermissionError, FileNotFoundError)):
            Path(target).read_bytes()
        with pytest.raises(OSError):
            Path(target).write_text("forbidden")
    link = Path(os.environ["TMPDIR"]) / "sibling-link"
    link.symlink_to(sibling_file)
    try:
        with pytest.raises((PermissionError, FileNotFoundError)):
            link.read_bytes()
    finally:
        link.unlink()
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).read_bytes()",
            sibling_file,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode != 0
    assert Path("own.txt").read_text() == "node data"
    Path("own.txt").chmod(0o600)
    Path("results").mkdir(mode=0o700, exist_ok=True)
    Path("results/plot.png").write_text("plot data")
    return json.dumps({"confined": True})


class IsolatedWorkspaceScientist(AIScientistAgent):
    """Exercise actual node workers, cloning, main access, and artifact promotion."""

    def _execute_session(self, session, owner, portal):
        import traceback

        from corral.agents.schema import AgentUsage

        try:
            return self._check_session(session, owner, portal)
        except Exception:
            # Preserve diagnostics across the harness's ExceptionGroup handling.
            return traceback.format_exc(), AgentUsage(), {}

    def _check_session(self, session, owner, portal):
        import json
        from pathlib import Path

        from corral.agents.ai_scientist.agent import _BranchSessionRegistry
        from corral.agents.ai_scientist.search.nodes import (
            ExperimentNode,
            PlannedAction,
        )
        from corral.agents.ai_scientist.search.tree import ExperimentTree
        from corral.agents.ai_scientist.tools.execution_pool import ExecutionPool
        from corral.agents.schema import AgentUsage
        from corral.core.action import Action

        assert os.getuid() != 0
        with pytest.raises(PermissionError):
            Path("/opt/corral/pyproject.toml").read_text()
        Path(session.workspace, "parent.txt").write_text("parent contents")
        sessions = _BranchSessionRegistry(session, portal)
        pool = ExecutionPool(
            sessions=sessions, tools=list(session.tools), max_tool_calls=30
        )
        try:
            first, second = pool.create(), pool.create()

            def invoke(branch, name, arguments):
                result = branch.executor.execute_plan(
                    [
                        PlannedAction(
                            purpose="Check node confinement",
                            tool_name=name,
                            arguments=arguments,
                            expected_information="Only assigned node files are accessible",
                        )
                    ]
                )[0]
                assert result.success, result
                return result.result

            assert (
                invoke(first, "read_file", {"path": "parent.txt"}) == "parent contents"
            )
            invoke(
                first, "write_file", {"path": "model.txt", "content": "winning model"}
            )
            invoke(
                second, "write_file", {"path": "model.txt", "content": "losing model"}
            )
            arguments = {
                "parent_file": str(Path(session.workspace, "parent.txt")),
                "sibling_file": str(Path(second.workspace, "model.txt")),
            }
            assert json.loads(invoke(first, "node_boundary_probe", arguments))[
                "confined"
            ]
            assert Path(first.workspace, "own.txt").read_text() == "node data"
            assert Path(first.workspace, "results/plot.png").read_text() == "plot data"
            job = json.loads(invoke(first, "start_node_boundary_probe", arguments))
            finished = json.loads(
                invoke(first, "get_job_result", {"job_id": job["job_id"], "wait": True})
            )
            assert finished["status"] == "succeeded", finished
            assert json.loads(finished["result"])["confined"]
            shell = json.loads(
                invoke(
                    first,
                    "terminal",
                    {
                        "command": "cat ../parent.txt; printf shell > shell.txt",
                        "timeout_seconds": 30,
                        "max_output_chars": 2000,
                    },
                )
            )
            assert (
                "No such file" in shell["output"]
                or "Permission denied" in shell["output"]
            )
            assert Path(first.workspace, "shell.txt").read_text() == "shell"

            main_workspace = session.workspace

            class NodeAgent:
                async def run_session(self, child_session):
                    from corral.agents.schema import AgentOutcome

                    assert (
                        Path(child_session.workspace, "model.txt").read_text()
                        == "winning model"
                    )
                    with pytest.raises((PermissionError, FileNotFoundError)):
                        Path(main_workspace, "parent.txt").read_text()
                    with pytest.raises(RuntimeError, match="calling agent"):
                        await child_session.fork_branch(workspace_parent=main_workspace)
                    nested = await child_session.fork_branch()
                    assert Path(nested.workspace).is_relative_to(
                        child_session.workspace
                    )
                    return AgentOutcome(
                        status="iteration_limit", error="node agent verified"
                    )

            outcome = portal.call(
                sessions.session(first.execution_id).run_delegate, NodeAgent()
            )
            assert outcome.error == "node agent verified", outcome

            clone = sessions.clone_branch(first.execution_id)
            try:
                assert len({branch.workspace for branch in (first, second, clone)}) == 3
                for branch in (first, second, clone):
                    assert Path(branch.workspace).parent == Path(
                        session.workspace, ".corral-nodes"
                    )
                    if branch is not first:
                        assert not Path(branch.workspace, ".corral-nodes").exists()
                    Path(branch.workspace, "from-main.txt").write_text("main can write")
                assert Path(clone.workspace, "model.txt").read_text() == "winning model"
                response = clone.execute(
                    Action(
                        name="write_file",
                        arguments={"path": "model.txt", "content": "changed clone"},
                    )
                )
                assert response.success, response
                assert Path(first.workspace, "model.txt").read_text() == "winning model"
            finally:
                sessions.close_branch(clone.execution_id)

            node = ExperimentNode(
                id="node_1",
                branch_id=first.branch_id,
                stage="research",
                node_type="research",
                hypothesis="independent artifacts",
                rationale="probe",
            )
            assert not Path(session.workspace, "model.txt").exists()
            Path(session.workspace, "model.txt").write_text("stale main output")
            promotion = pool.promote_artifacts(
                [node],
                ExperimentTree(),
                destination_workspace=session.workspace,
                destination_execution_id=session.execution_id,
            )
            assert promotion.source_workspace == first.workspace
            assert promotion.destination_workspace == session.workspace
            assert "model.txt" in promotion.files
            assert Path(session.workspace, "model.txt").read_text() == "winning model"
            assert (
                Path(session.workspace, "results/plot.png").read_text() == "plot data"
            )
            assert Path(second.workspace, "model.txt").read_text() == "losing model"
            pool.close(first.branch_id)
            assert invoke(second, "read_file", {"path": "model.txt"}) == "losing model"
        finally:
            assert pool.close_all() == []
        return "node isolation verified", AgentUsage(), {}


@pytest.mark.anyio()
async def test_scientist_node_workspaces_in_docker(workspace, tmp_path):
    from datetime import datetime, timezone
    from pathlib import Path

    from corral.core.environment import Toolset, default_file_tools
    from corral.core.task import TaskDefinition
    from corral.observability import NoOpObserver
    from corral.persistence import SQLiteCommitStore
    from corral.runtime.task_runner import TaskRuntime
    from corral.workspace import WorkspaceFilesystem, build_terminal_tool

    task = TaskDefinition(
        name="isolated-science",
        description="Isolate node artifacts",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    environment = Environment(
        "isolated-science",
        task,
        base_work_dir=workspace,
        task_execution_id="isolated",
        toolset=Toolset(
            pool={"node_boundary_probe": node_boundary_probe},
            select_all_when_unspecified=True,
            workspace_factory=lambda path: {
                **default_file_tools(path),
                "terminal": build_terminal_tool(WorkspaceFilesystem(path)),
            },
        ),
    )
    async with SQLiteCommitStore(
        tmp_path / "isolated.sqlite3", execution_id="isolated"
    ) as store:
        try:
            state = await TaskRuntime(store, NoOpObserver()).run(
                IsolatedWorkspaceScientist(model="offline-test"),
                environment,
                execution_id="isolated",
                max_iterations=20,
                started_at=datetime.now(timezone.utc),
            )
            assert state.submission == "node isolation verified", (
                state.submission or state.runtime.model_dump_json()
            )
            assert "model.txt" in state.workspace.files
            assert not any(
                name.startswith(".corral-nodes/") for name in state.workspace.files
            )
            assert (
                Path(environment.workspace_path, "model.txt").read_text()
                == "winning model"
            )
        finally:
            environment.shutdown_jobs()


@pytest.mark.anyio()
async def test_internal_entrypoint_protects_a_real_trial(tmp_path):
    from corral.orchestration.internal import run_task_from_files
    from corral.orchestration.models import (
        DockerSandboxSpec,
        RunTaskInput,
        SandboxMode,
        SandboxProfile,
    )

    request = RunTaskInput(
        execution_id="permission-trial",
        task_id="probe",
        environment_id="probe",
        agent_id="probe",
        started_at="2026-01-01T00:00:00+00:00",
        sandbox=SandboxProfile(
            mode=SandboxMode.DOCKER,
            docker=DockerSandboxSpec(
                image="permission-test",
                registry_module=f"{__name__}:permission_registry",
            ),
        ),
    )
    request_path = tmp_path / "request.json"
    import json
    from dataclasses import asdict

    request_path.write_text(json.dumps(asdict(request)))
    result_path = tmp_path / "result.json"
    assert await run_task_from_files(request_path, result_path) == 0
    assert json.loads(result_path.read_text())["submission"]


def permission_registry(request):
    """A model-free trial using the same private registry loader as production."""
    from corral.core.task import TaskDefinition
    from corral.orchestration.registry import RuntimeRegistry

    task = TaskDefinition(
        name="probe",
        description="probe",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    return RuntimeRegistry(
        agents={"probe": ProbeAgent()},
        environments={"probe": Environment("probe", task, base_work_dir="/workspace")},
    )


def test_wetlab_stateful_tool_retains_its_environment_update(workspace):
    engine_module = pytest.importorskip("wetlab.engine")
    from wetlab.env import QualitativeAnalysisEnvironment
    from wetlab.tools import mix_two_solutions

    from corral.core.state import EnvironmentState, ExecutionState
    from corral.core.transition import ToolExecutionResult

    engine = engine_module.WetlabEngine(
        engine_module.ChemicalSystemSpec(elements="K S(+6)")
    )
    inventory = {
        "acid": engine.stock_solution({"H+": 0.1, "HSO4-": 0.1}, description="acid"),
        "base": engine.stock_solution({"K+": 0.1, "OH-": 0.1}, description="base"),
    }
    initial = engine.snapshot(inventory).to_dict()
    state = ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="wetlab-permission",
        branch_id="main",
        environment=EnvironmentState(values={"hidden_arguments": {"wetlab": initial}}),
    )
    environment = object.__new__(QualitativeAnalysisEnvironment)
    environment.workspace_path = workspace
    result = permissions.execute_tool(
        environment,
        state,
        mix_two_solutions,
        {
            "wetlab": initial,
            "sol1_label": "acid",
            "sol2_label": "base",
            "sol1_vol": 5,
            "sol2_vol": 5,
            "test_label": "mixture",
        },
    )
    assert isinstance(result, ToolExecutionResult)
    assert result.environment["hidden_arguments"]["wetlab"] != initial


class WaitingAgent:
    async def run_session(self, session):
        import asyncio
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        child = subprocess.Popen(  # noqa: ASYNC220 - exercise synchronous SDK spawning
            [sys.executable, "-c", "import time; time.sleep(300)"],
            start_new_session=True,
        )
        Path(session.workspace, "started.json").write_text(
            json.dumps([os.getpid(), child.pid])
        )
        await asyncio.sleep(300)


@tool(hidden_args=["secret"], trusted=True, background_capable=True)
def trusted_private_probe(secret: str) -> str:
    """Use a private task input and publish only the controller identity."""
    assert secret
    return str(os.getuid())


@tool(background_capable=True)
def public_python_probe(code: str) -> str:
    """Execute model-controlled Python in a worker with no private objects."""
    namespace = {}
    exec(code, namespace)
    return "public code completed"


class PrivateProbeEnvironment(Environment):
    def execute_tool(self, state, selected_tool, arguments):
        from corral.core.transition import ToolExecutionResult

        if selected_tool.name != "trusted_private_probe":
            return super().execute_tool(state, selected_tool, arguments)
        assert selected_tool.trusted
        assert os.getuid() == 0
        assert (
            arguments["secret"]
            == state.environment.values["hidden_arguments"]["secret"]  # noqa: PD011 - EnvironmentState mapping, not pandas
        )
        return ToolExecutionResult(
            content=selected_tool.execute(**arguments),
            environment={
                **state.environment.values,  # noqa: PD011 - EnvironmentState mapping, not pandas
                "private_update": arguments["secret"],
            },
        )


_MEMORY_PROBE = """
import gc
import os
from corral.agents.session import AgentSession
from corral.core.environment import Environment
from corral.core.state import ExecutionState
assert os.getuid() != 0
assert not any(isinstance(obj, (AgentSession, ExecutionState, Environment)) for obj in gc.get_objects())
"""


class StateBoundaryProbe:
    session_capabilities = AgentSessionCapabilities(inspect_subagents=True)

    def __init__(self, transport):
        self.tool_transport = transport

    async def run_session(self, session):
        import traceback

        try:
            return await self.check_boundary(session)
        except Exception as exc:
            raise AssertionError(traceback.format_exc()) from exc

    async def check_boundary(self, session):
        import json

        from corral.agents.schema import AgentOutcome
        from corral.core.action import Action
        from corral.core.tool import ToolResponse

        exec(_MEMORY_PROBE, {})
        assert not hasattr(session, "state")
        assert not hasattr(session, "previous_state")
        assert session.get_agent_state("hidden_arguments") is None
        await session.set_agent_state("public-notes", {"note": "remember this"})
        assert session.get_agent_state("public-notes") == {"note": "remember this"}
        with pytest.raises(RuntimeError, match="not permitted"):
            session._sync_call("state", {})

        async def invoke(name, arguments):
            if self.tool_transport == "python":
                return await session.execute(Action(name=name, arguments=arguments))
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with (
                streamablehttp_client(session.tool_connection.mcp_url) as streams,
                ClientSession(streams[0], streams[1]) as client,
            ):
                await client.initialize()
                result = await client.call_tool(name, arguments)
                content = "".join(
                    item.text for item in result.content if item.type == "text"
                )
                return ToolResponse(
                    success=not result.isError, result=content, error=None
                )

        response = await invoke("trusted_private_probe", {})
        assert response.success, response
        assert response.result == "0", response
        rejected = await invoke("trusted_private_probe", {"secret": "forged"})
        assert not rejected.success
        response = await invoke("public_python_probe", {"code": _MEMORY_PROBE})
        assert response.success, response
        response = await invoke(
            "terminal",
            {
                "command": "id -u; cat /opt/corral/pyproject.toml",
                "timeout_seconds": 120,
                "max_output_chars": 20000,
            },
        )
        assert response.success, response
        assert "Permission denied" in response.result
        for name, arguments in (
            ("public_python_probe", {"code": _MEMORY_PROBE}),
            ("trusted_private_probe", {}),
        ):
            started = await invoke("start_" + name, arguments)
            assert started.success, started
            job_id = json.loads(started.result)["job_id"]
            finished = await invoke("get_job_result", {"job_id": job_id, "wait": True})
            assert finished.success, finished
            assert json.loads(finished.result)["result"] in {
                "0",
                "public code completed",
            }

        class Child:
            async def run_session(self, child_session):
                exec(_MEMORY_PROBE, {})
                response = await child_session.execute(
                    Action(name="trusted_private_probe", arguments={})
                )
                assert response.success, response
                assert response.result == "0", response
                return AgentOutcome(status="completed", answer="public child result")

        child_id = await session.spawn_subagent(Child(), handoff="Public child task")
        assert (
            await session.wait_for_subagent(child_id)
        ).answer == "public child result"
        trace = await session.inspect_subagent(child_id)
        assert trace.commits == ()
        assert trace.context.task.metadata == {}
        trace_tool = await invoke(
            "inspect_subagent", {"child_run_id": child_id, "include_commits": True}
        )
        assert json.loads(trace_tool.result)["commits"] == []
        imported = await session.import_subagent_context(child_id)
        assert isinstance(imported, str)
        assert len(imported) == 64
        branch = await session.fork_branch()
        assert not hasattr(branch, "state")
        assert branch.get_agent_state("hidden_arguments") is None
        await session.execute(
            Action(name="submit_answer", arguments={"answer": "boundary held"})
        )
        return AgentOutcome(status="completed", answer="boundary held")


@pytest.mark.anyio()
@pytest.mark.parametrize("transport", ["python", "mcp"])
async def test_private_state_stays_in_controller(
    workspace, tmp_path, monkeypatch, transport
):
    from datetime import datetime, timezone
    from uuid import uuid4

    from corral.core.environment import Toolset
    from corral.core.task import EnvironmentSetup, TaskDefinition
    from corral.observability import NoOpObserver
    from corral.persistence import SQLiteCommitStore
    from corral.runtime.task_runner import TaskRuntime
    from corral.workspace import WorkspaceFilesystem, build_terminal_tool

    secret = str(uuid4())
    requests = []
    original_worker = permissions.run_worker

    def audited_worker(kind, payload, workspace, **kwargs):
        serialized = permissions.serialize(payload)
        assert secret.encode() not in serialized
        requests.append(kind)
        return original_worker(kind, payload, workspace, **kwargs)

    monkeypatch.setattr(permissions, "run_worker", audited_worker)
    task = TaskDefinition(
        name="private-probe",
        description="Public task instructions",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        scoring_inputs={"answer": secret},
        submission_format={},
        resolve_answer=False,
        setup_fn=lambda env, state: EnvironmentSetup(
            hidden_arguments={"secret": secret}
        ),
    )
    environment = PrivateProbeEnvironment(
        "private-probe",
        task,
        base_work_dir=workspace,
        task_execution_id="boundary",
        toolset=Toolset(
            select_all_when_unspecified=True,
            pool={
                tool.name: tool for tool in (trusted_private_probe, public_python_probe)
            },
            workspace_factory=lambda path: {
                "terminal": build_terminal_tool(WorkspaceFilesystem(path))
            },
        ),
    )
    store = SQLiteCommitStore(tmp_path / "boundary.sqlite3", execution_id="boundary")
    try:
        state = await TaskRuntime(store, NoOpObserver()).run(
            StateBoundaryProbe(transport),
            environment,
            execution_id="boundary",
            max_iterations=20,
            started_at=datetime.now(timezone.utc),
        )
        assert state.submission == "boundary held", state.runtime.model_dump_json()
        assert state.environment.values["hidden_arguments"]["secret"] == secret  # noqa: PD011 - EnvironmentState mapping, not pandas
        assert state.environment.values["private_update"] == secret  # noqa: PD011 - EnvironmentState mapping, not pandas
        assert secret not in state.model_dump_json(include={"conversations"})
        assert {"agent", "tool", "terminal"} <= set(requests)
    finally:
        environment.shutdown_jobs()
        await store.aclose()


@pytest.mark.anyio()
async def test_cancelled_agent_stops_its_descendants_before_returning(
    workspace, tmp_path
):
    import asyncio
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from corral.core.task import TaskDefinition
    from corral.observability import NoOpObserver
    from corral.persistence import SQLiteCommitStore
    from corral.runtime.task_runner import TaskRuntime

    task = TaskDefinition(
        name="cancel",
        description="probe",
        tools=[],
        scoring_fn=lambda answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    environment = Environment(
        "cancel", task, base_work_dir=workspace, task_execution_id="cancel"
    )
    store = SQLiteCommitStore(tmp_path / "cancel.sqlite3", execution_id="cancel")
    running = asyncio.create_task(
        TaskRuntime(store, NoOpObserver()).run(
            WaitingAgent(),
            environment,
            execution_id="cancel",
            max_iterations=3,
            started_at=datetime.now(timezone.utc),
        )
    )
    started = Path(environment.workspace_path) / "started.json"
    try:
        async with asyncio.timeout(20):
            while not started.exists():
                if running.done():
                    raise AssertionError(
                        f"worker stopped before cancellation: {running.result()}"
                    )
                await asyncio.sleep(0.05)
            pids = json.loads(started.read_text())
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        for pid in pids:
            try:
                status = Path(f"/proc/{pid}/status").read_text()
            except FileNotFoundError:
                continue
            assert status.split("State:", 1)[1].split()[0] == "Z"
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await store.aclose()
