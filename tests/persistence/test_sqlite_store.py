"""Database compatibility, atomicity and async lifecycle regressions."""

import asyncio
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from corral.core import (
    ActorRef,
    AgentStarted,
    CommitRequest,
    ExecutionStarted,
    RuntimeUpdate,
)
from corral.persistence import (
    CommitConflictError,
    ShardedCommitStore,
    SQLiteCommitStore,
)


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def start_request(execution_id="execution"):
    return CommitRequest(
        request_id="start",
        branch_id="main",
        author=ActorRef(
            kind="runtime", actor_id="corral", run_id=f"runtime:{execution_id}"
        ),
        event=ExecutionStarted(
            runtime=RuntimeUpdate(
                status="running", started_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
        ),
    )


def agent_request(root, index=0):
    return CommitRequest(
        request_id=f"agent:{index}",
        branch_id="main",
        based_on_hash=root.hash,
        author=root.author,
        event=AgentStarted(agent_run_id=f"run:{index}", agent_id="agent"),
    )


@pytest.mark.anyio()
async def test_reopens_legacy_ledger_without_changing_schema_or_records(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    fixture = Path(__file__).with_name("fixtures") / "legacy_commits.sql"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(fixture.read_text())
        schema = connection.execute(
            "SELECT name, sql FROM sqlite_master ORDER BY name"
        ).fetchall()
        records = connection.execute(
            "SELECT * FROM commits ORDER BY sequence"
        ).fetchall()

    async with SQLiteCommitStore(path, "legacy") as store:
        commits = [commit async for commit in store.iter_commits()]
        assert [commit.hash for commit in commits] == [row[0] for row in records]
        assert await store.append(start_request("legacy")) == commits[0]
        assert await store.get_commit(commits[1].hash) == commits[1]
        assert await store.snapshot_count() == 2
        state = await store.materialize("main")
        assert state.conversations["agent-run"][0]["content"] == "legacy message"
        assert (
            await store.materialize("main", commits[1].hash)
        ).through_commit_hash == commits[1].hash
        assert await store.head("experiment") == commits[1]
        assert [commit async for commit in store.iter_commits("experiment")] == commits[
            :2
        ]
        assert [
            commit
            async for commit in store.iter_commits("main", through_hash=commits[1].hash)
        ] == commits[:2]
        assert await store.trace_commits("agent-run") == tuple(commits[1:])
        assert await store.trace_commits("agent-run", branch_id="experiment") == (
            commits[1],
        )
        added = await store.append(agent_request(commits[0]))
        assert added.sequence == added.branch_sequence == 3

    with closing(sqlite3.connect(path)) as connection:
        assert (
            connection.execute(
                "SELECT name, sql FROM sqlite_master ORDER BY name"
            ).fetchall()
            == schema
        )
        assert (
            connection.execute(
                "SELECT * FROM commits WHERE sequence < 3 ORDER BY sequence"
            ).fetchall()
            == records
        )


@pytest.mark.anyio()
async def test_concurrent_stores_serialize_appends_and_idempotent_retries(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    first = SQLiteCommitStore(path, "execution")
    second = SQLiteCommitStore(path, "execution")
    try:
        # First use races schema initialization as well as the initial append.
        roots = await asyncio.gather(
            first.append(start_request()), second.append(start_request())
        )
        assert roots[0] == roots[1]
        root = roots[0]
        requests = [agent_request(root, index) for index in range(8)]
        results = await asyncio.gather(
            *(
                store.append(request)
                for request in requests
                for store in (first, second)
            )
        )
        assert results[::2] == results[1::2]
        commits = [commit async for commit in first.iter_commits("main")]
        assert len(commits) == 9
        assert [commit.sequence for commit in commits] == list(range(9))
        assert [commit.branch_sequence for commit in commits] == list(range(9))
        assert [commit.parent_hash for commit in commits[1:]] == [
            commit.hash for commit in commits[:-1]
        ]
        assert len((await first.materialize("main")).agent_runs) == 8
        assert await second.materialize("main") == await first.materialize("main")

        branches = await asyncio.gather(
            first.create_branch(branch_id="fork", from_hash=root.hash),
            second.create_branch(branch_id="fork", from_hash=root.hash),
            return_exceptions=True,
        )
        assert branches.count(None) == 1
        assert sum(isinstance(result, CommitConflictError) for result in branches) == 1
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.anyio()
async def test_retries_a_busy_journal_mode_change(tmp_path, monkeypatch):
    execute = aiosqlite.Connection.execute
    attempts = 0

    @asynccontextmanager
    async def busy_once(connection, sql, *args):
        nonlocal attempts
        if sql == "PRAGMA journal_mode = WAL":
            attempts += 1
            if attempts == 1:
                raise aiosqlite.OperationalError("database is locked")
        async with execute(connection, sql, *args) as cursor:
            yield cursor

    monkeypatch.setattr(aiosqlite.Connection, "execute", busy_once)
    async with SQLiteCommitStore(tmp_path / "busy.sqlite3", "execution") as store:
        root = await store.append(start_request())
        assert await store.head("main") == root
    assert attempts == 2


@pytest.mark.anyio()
@pytest.mark.parametrize("cancel", [False, True], ids=["error", "cancellation"])
async def test_interrupted_append_rolls_back_all_writes(tmp_path, monkeypatch, cancel):
    path = tmp_path / "atomic.sqlite3"
    snapshots = tmp_path / "snapshots"
    async with (
        SQLiteCommitStore(
            path, "execution", snapshot_interval=1, state_snapshot_root=snapshots
        ) as store,
        SQLiteCommitStore(path, "execution") as reader,
    ):
        root = await store.append(start_request())
        before = {p.name: p.read_bytes() for p in snapshots.iterdir()}
        request = agent_request(root)
        appended = asyncio.Event()
        release = asyncio.Event()
        append_commit = store._append_commit

        async def interrupt_after_writes(*args):
            await append_commit(*args)
            appended.set()
            await release.wait()
            raise RuntimeError("interrupted before commit")

        monkeypatch.setattr(store, "_append_commit", interrupt_after_writes)
        task = asyncio.create_task(store.append(request))
        try:
            await asyncio.wait_for(appended.wait(), timeout=5)
            # WAL readers see the last committed state during an active write.
            assert await reader.head("main") == root
            if cancel:
                task.cancel()
            else:
                release.set()
            with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        assert await store.head("main") == root
        assert [commit async for commit in store.iter_commits()] == [root]
        assert await store.snapshot_count() == 1
        assert {p.name: p.read_bytes() for p in snapshots.iterdir()} == before
        monkeypatch.setattr(store, "_append_commit", append_commit)
        committed = await store.append(request)
        assert committed.sequence == committed.branch_sequence == 1
        assert await store.append(request) == committed
        assert await store.snapshot_count() == 2
        assert (await reader.materialize("main")).through_commit_hash == committed.hash


@pytest.mark.anyio()
async def test_async_cleanup_closes_stores_and_is_repeatable(tmp_path):
    path = tmp_path / "lifecycle.sqlite3"
    store = SQLiteCommitStore(path, "execution")
    assert not path.exists()
    async with store:
        root = await store.append(start_request())
    await store.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await store.head("main")
    with pytest.raises(RuntimeError, match="closed"):
        await store.trace_commits(root.author.run_id)
    with pytest.raises(RuntimeError, match="closed"):
        await store.snapshot_count()
    async with SQLiteCommitStore(path, "execution") as reopened:
        assert await reopened.head("main") == root

    unused = SQLiteCommitStore(tmp_path / "unused.sqlite3", "execution")
    await unused.aclose()
    assert not unused.path.exists()


@pytest.mark.anyio()
async def test_shard_cleanup_allows_reopening_the_committed_ledger(tmp_path):
    async with ShardedCommitStore(tmp_path) as store:
        shard = store.for_execution("execution")
        root = await shard.append(start_request())
        await store.close_execution("execution")
        with pytest.raises(RuntimeError, match="closed"):
            await shard.head("main")
        reopened = store.for_execution("execution")
        assert await reopened.head("main") == root
    with pytest.raises(RuntimeError, match="closed"):
        await reopened.head("main")
