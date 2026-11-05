"""Exercise historical psychometrics continuations without an LLM.

Local: PYTHONPATH=src:tasks/psychometrics/src python -m scripts.check_trial_continuation --output /tmp/continuation-local
Docker: python -m scripts.check_trial_continuation --sandbox docker --image IMAGE --output /tmp/continuation-docker
The image must include this checkout and the psychometrics task dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from corral.agents.schema import AgentOutcome
from corral.core import Action
from corral.observability import NoOpObserver
from corral.orchestration import (
    ContinuationSource,
    DockerSandboxSpec,
    DockerTaskLauncher,
    RunTaskInput,
    RuntimeRegistry,
    SandboxProfile,
)
from corral.persistence import ShardedCommitStore
from corral.run import execute_task

TASK = "psy_l1_t01_hsns_structure"


class ProbeAgent:
    async def run_session(self, session):
        if (
            session.messages
            and session.messages[-1].get("content") == "Check the restored state."
        ):
            assert session.messages[-1]["content"] == "Check the restored state."
            code = (
                "assert x == 41\n"
                "assert df.shape[1] == 26\n"
                "assert Path('checkpoint.txt').read_text() == 'before'\n"
                "assert np.random.randint(1000000) == expected_draw\n"
                "Path('checkpoint.txt').write_text('child')\n"
                "x = 123\nprint('restored', df.shape)"
            )
            result = await session.execute(
                Action(name="PythonREPL", arguments={"input_code": code})
            )
            assert result.success, result
            assert "restored" in (result.result or ""), result
        else:
            await session.record_message(
                {"role": "user", "content": "Create a checkpoint."}
            )
            code = (
                "import pandas as pd\nimport numpy as np\nfrom pathlib import Path\n"
                "df = pd.read_csv('data.csv', sep='\\t')\n"
                "x = 41\nnp.random.seed(123)\n"
                "expected_draw = np.random.randint(1000000)\nnp.random.seed(123)\n"
                "Path('checkpoint.txt').write_text('before')\nprint(df.shape)"
            )
            result = await session.execute(
                Action(name="PythonREPL", arguments={"input_code": code})
            )
            assert result.success, result
            assert "26" in (result.result or ""), result
            result = await session.execute(
                Action(
                    name="PythonREPL",
                    arguments={
                        "input_code": "x = 99\nPath('checkpoint.txt').write_text('after')\nprint(x)"
                    },
                )
            )
            assert result.success, result
        result = await session.execute(
            Action(
                name="submit_answer", arguments={"answer": "checkpoint probe complete"}
            )
        )
        assert result.success, result
        return AgentOutcome(status="completed", answer="checkpoint probe complete")


def registry_factory(_request):
    """Docker controller registry, using the real psychometrics environment."""
    from corral_psychometrics.env import create_environments

    return RuntimeRegistry(
        agents={"probe": ProbeAgent()},
        environments=create_environments(level=1, work_dir="/workspace"),
    )


def fingerprints(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


async def check(args):
    from corral_psychometrics.env import create_environments

    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("use a fresh output directory")
    store = ShardedCommitStore(output)
    registry = RuntimeRegistry(
        agents={"probe": ProbeAgent()},
        environments=create_environments(level=1, work_dir=str(output / "workspaces")),
        workspace_manager_factory=store.workspace_manager,
    )
    launcher = None
    sandbox = SandboxProfile.local()
    if args.sandbox == "docker":
        spec = await DockerTaskLauncher.preflight(
            DockerSandboxSpec(
                image=args.image,
                registry_module="scripts.check_trial_continuation:registry_factory",
                environment_allowlist=(),
                retention="never",
            ),
            docker_executable=args.docker_executable,
        )
        sandbox = SandboxProfile(mode="docker", docker=spec)
        launcher = DockerTaskLauncher(store, docker_executable=args.docker_executable)
    request = RunTaskInput(
        execution_id=f"{output.name}:parent",
        task_id=TASK,
        environment_id=TASK,
        agent_id="probe",
        max_iterations=10,
        sandbox=sandbox,
    )
    try:
        parent = await execute_task(
            task=request,
            state_store=store,
            registry=registry,
            docker_launcher=launcher,
            observer=NoOpObserver(),
        )
        assert parent.status == "submitted", parent
        source_store = store.for_execution(request.execution_id)
        commits = [c async for c in source_store.iter_commits("main")]
        cutoff = next(c for c in commits if c.event.type == "tool.completed")
        original = await source_store.materialize("main", cutoff.hash)
        source_dir = store.execution_dir(request.execution_id)
        await store.close_execution(request.execution_id)
        before = fingerprints(source_dir)
        source = ContinuationSource(
            trial=str(source_dir),
            commit_hash=cutoff.hash,
            message="Check the restored state.",
        )
        children = []
        for i in range(4):
            child_request = replace(
                request,
                execution_id=f"{output.name}:child-{i}",
                trial_index=i,
                continuation=source,
            )
            child = await execute_task(
                task=child_request,
                state_store=store,
                registry=registry,
                docker_launcher=launcher,
                observer=NoOpObserver(),
            )
            assert child.status == "submitted", child
            state = await store.for_execution(child.execution_id).materialize("main")
            assert state.runtime.metadata["lineage"]["commit_hash"] == cutoff.hash
            children.append(child.execution_id)
        resumed = await execute_task(
            task=child_request,
            state_store=store,
            registry=registry,
            docker_launcher=launcher,
            observer=NoOpObserver(),
        )
        assert resumed.commit_hash == child.commit_hash
        assert fingerprints(source_dir) == before, "source files were modified"
        summary = {
            "sandbox": args.sandbox,
            "cutoff": cutoff.hash,
            "source": str(source_dir),
            "children": children,
            "resumed_child": resumed.execution_id,
            "checkpoint_bytes": len(
                original.environment.values["hidden_arguments"]["analysis_session"]
            ),
            "checks": "workspace, dataframe, variable, NumPy RNG, conversation, four independent children, unchanged source",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))  # noqa: T201
    finally:
        registry.close()
        await store.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sandbox", choices=("local", "docker"), default="local")
    parser.add_argument("--image", default="corral-psychometrics:latest")
    parser.add_argument("--docker-executable", default="docker")
    asyncio.run(check(parser.parse_args()))
