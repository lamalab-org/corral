"""Trusted bootstrap. No model-controlled callable runs before dropping UID."""

# Optional agent imports must occur only in agent bootstraps.
# ruff: noqa: PLC0415

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

import cloudpickle
from pydantic import TypeAdapter

from corral.runtime._worker_filesystem import bind_bootstrap_parent, enter_workspace
from corral.runtime.permissions import DENIED, drop_privileges, private_controller_types


def main() -> None:
    request, descriptor = sys.argv[1:]
    # This file is written by the root controller in its private directory.
    with Path(request).open("rb") as stream:
        kind, payload, uid, gid, workspace, workspace_fd = cloudpickle.load(stream)
    Path(request).unlink()
    output = os.fdopen(int(descriptor), "w")
    os.set_inheritable(output.fileno(), False)
    if kind == "agent":
        from corral.agents.ai_scientist.prompts import load_prompt
        from corral.agents.session import _run_agent_lifecycle
        from corral.runtime.agent_worker import RemoteSession, _DelegatedAgent

        for template in (
            Path(__file__)
            .parents[1]
            .joinpath("agents/ai_scientist/prompts")
            .glob("*.md")
        ):
            load_prompt(template.stem)

        agent, connection = payload
        if isinstance(agent, _DelegatedAgent):
            # Only trusted, installed agent modules may be preloaded as root.
            # The opaque delegate pickle is decoded AFTER the identity change.
            for path in Path(__file__).parents[1].joinpath("agents").rglob("*.py"):
                parts = (
                    path.relative_to(Path(__file__).parents[2]).with_suffix("").parts
                )
                module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
                with suppress(ImportError):
                    importlib.import_module(module)
        session = RemoteSession(*connection)
    elif kind == "tool":
        tool, arguments = payload
    elif kind == "terminal":
        from corral.workspace import WorkspaceFilesystem, build_terminal_tool

        tool = build_terminal_tool(WorkspaceFilesystem(workspace))
        arguments = payload
    else:
        raise ValueError("unknown restricted worker kind")
    private_controller_types()
    # Imported runtimes may have started threads. Fork a single-threaded child
    # so none can retain the original filesystem context across unshare/chroot.
    sys.stdout.flush()
    sys.stderr.flush()
    parent_pid = os.getpid()
    child = os.fork()
    if child:
        output.close()
        _, status = os.waitpid(child, 0)
        code = os.waitstatus_to_exitcode(status)
        os._exit(code if code >= 0 else 128 - code)
    bind_bootstrap_parent(parent_pid)
    scratch = enter_workspace(
        workspace,
        Path(request).parent / "root",
        uid,
        gid,
        keep_fds={0, 1, 2, output.fileno()},
        workspace_fd=workspace_fd,
    )
    drop_privileges(uid, gid, workspace, scratch=scratch)
    # Linux clears the parent-death signal when credentials change.
    bind_bootstrap_parent(parent_pid)
    try:
        if kind == "agent":
            if isinstance(agent, _DelegatedAgent):
                agent = cloudpickle.loads(base64.b64decode(agent.blob))
            session._hook_agent = agent
            session._hooks = getattr(agent, "hooks", None)
            result = asyncio.run(
                _run_agent_lifecycle(agent, session, lambda: agent.run_session(session))
            )
        else:
            result = tool.execute(**arguments)
            from corral.core.transition import ToolExecutionResult

            if isinstance(result, ToolExecutionResult):
                raise PermissionError(
                    "restricted tools cannot replace private task state"
                )
            result = {"content": result}
        response = {
            "ok": True,
            "result": TypeAdapter(type(result)).dump_python(result, mode="json"),
        }
    except PermissionError as exc:
        response = {"ok": False, "error": f"{DENIED}: {exc.filename or exc}"}
    except Exception as exc:
        response = {"ok": False, "error": str(exc)}
    with output:
        json.dump(response, output)


if __name__ == "__main__":
    main()
