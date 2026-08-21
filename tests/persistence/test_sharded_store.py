import asyncio
import json
from datetime import datetime, timezone

import pytest

from corral.core import (
    ActorRef,
    CommitRequest,
    ExecutionCompleted,
    ExecutionStarted,
    RuntimeUpdate,
)
from corral.persistence import ShardedCommitStore


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def test_execution_shards_keep_state_and_snapshot_manifests_separate(tmp_path):
    store = ShardedCommitStore(tmp_path / ".corral")
    first_dir = store.execution_dir("benchmark:task:0")
    second_dir = store.execution_dir("benchmark:task:1")

    assert first_dir != second_dir
    assert first_dir.parent == second_dir.parent
    assert json.loads((first_dir / "metadata.json").read_text())["execution_id"] == (
        "benchmark:task:0"
    )

    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    first_source.mkdir()
    second_source.mkdir()
    (first_source / "result.txt").write_text("trial zero")
    (second_source / "result.txt").write_text("trial one")

    first_manager = store.workspace_manager("benchmark:task:0")
    second_manager = store.workspace_manager("benchmark:task:1")
    first = asyncio.run(first_manager.snapshot(first_source))
    second = asyncio.run(second_manager.snapshot(second_source))

    assert first.files["result.txt"].sha256 != second.files["result.txt"].sha256
    assert (
        json.loads((first_dir / "snapshots" / "latest.json").read_text())["files"][
            "result.txt"
        ]["sha256"]
        == first.files["result.txt"].sha256
    )
    assert (
        json.loads((second_dir / "snapshots" / "latest.json").read_text())["files"][
            "result.txt"
        ]["sha256"]
        == second.files["result.txt"].sha256
    )

    store.close()


def test_snapshot_manifest_is_published_only_after_blob_storage(tmp_path):
    store = ShardedCommitStore(tmp_path / ".corral")
    execution_id = "benchmark:task:0"
    source = tmp_path / "workspace"
    source.mkdir()
    (source / "data.txt").write_text("first")
    manager = store.workspace_manager(execution_id)

    first = asyncio.run(manager.snapshot(source))
    (source / "data.txt").write_text("second")
    second = asyncio.run(manager.snapshot(source, previous=first))

    snapshots = store.execution_dir(execution_id) / "snapshots"
    assert (snapshots / "00000000.json").is_file()
    assert (snapshots / "00000001.json").is_file()
    latest = json.loads((snapshots / "latest.json").read_text())
    assert latest["revision"] == second.revision == 1
    assert asyncio.run(
        manager.artifact_store.contains(second.files["data.txt"].blob_ref)
    )
    store.close()


def test_benchmark_run_uses_descriptive_task_and_k_directories(tmp_path):
    run_id = "agent-tool-calling__model-gpt-5.6__env-wetlab__k-2"
    run_dir = tmp_path / "runs" / run_id
    store = ShardedCommitStore(run_dir, benchmark_run_id=run_id)

    execution_dir = store.execution_dir(f"{run_id}:qualysis_task_01:1")

    assert execution_dir == run_dir / "task-qualysis_task_01" / "k-2"
    assert (execution_dir / "workspace-snapshots").is_dir()
    assert (execution_dir / "state-snapshots").is_dir()
    metadata = json.loads((execution_dir / "metadata.json").read_text())
    assert metadata["task_id"] == "qualysis_task_01"
    assert metadata["trial_index"] == 1
    assert metadata["k"] == 2
    store.close()


@pytest.mark.anyio()
async def test_benchmark_shard_exports_periodic_and_final_state_snapshots(tmp_path):
    run_id = "benchmark"
    execution_id = f"{run_id}:task:0"
    store = ShardedCommitStore(
        tmp_path / "runs" / run_id,
        benchmark_run_id=run_id,
        snapshot_interval=50,
    )
    shard = store.for_execution(execution_id)
    actor = ActorRef(
        kind="runtime",
        actor_id="corral",
        run_id=f"runtime:{execution_id}",
    )
    started = await shard.append(
        CommitRequest(
            request_id="execution:started",
            execution_id=execution_id,
            branch_id="main",
            author=actor,
            event=ExecutionStarted(
                runtime=RuntimeUpdate(
                    status="running", started_at=datetime.now(timezone.utc)
                )
            ),
        )
    )
    completed = await shard.append(
        CommitRequest(
            request_id="execution:completed",
            execution_id=execution_id,
            branch_id="main",
            based_on_hash=started.hash,
            author=actor,
            event=ExecutionCompleted(status="terminal"),
        )
    )

    state_snapshots = shard.root / "state-snapshots"
    assert (
        state_snapshots / f"{started.sequence:08d}-{started.hash[:12]}.json"
    ).is_file()
    assert (
        state_snapshots / f"{completed.sequence:08d}-{completed.hash[:12]}.json"
    ).is_file()
    final = json.loads((state_snapshots / "final.json").read_text())
    assert final["through_commit_hash"] == completed.hash
    assert final["runtime"]["status"] == "terminal"
    store.close()
