"""Transactional SQLite implementation of the authored commit ledger."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from corral.core.commit import Commit, CommitRequest, canonical_json
from corral.core.events import (
    AgentCompleted,
    AgentSpawned,
    AgentStarted,
    AgentStateUpdated,
    AgentTurnRecorded,
    ContextImported,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionStarted,
    ParallelGroupCompleted,
    SubmissionAccepted,
    TaskConfigured,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
)
from corral.core.reducer import EventReducer, ReducerError, SharedStateConflictError
from corral.core.state import ExecutionState
from corral.persistence.base import (
    AuthorPermissionError,
    CommitConflictError,
    CommitIntegrityError,
    CommitNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from corral.core.actors import ActorRef


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_AUTHOR_KINDS: dict[type[Any], frozenset[str]] = {
    ExecutionStarted: frozenset({"runtime"}),
    TaskConfigured: frozenset({"runtime"}),
    AgentStarted: frozenset({"runtime"}),
    AgentSpawned: frozenset({"agent"}),
    AgentTurnRecorded: frozenset({"agent"}),
    AgentStateUpdated: frozenset({"agent"}),
    ToolStarted: frozenset({"runtime"}),
    ToolCompleted: frozenset({"tool"}),
    ToolFailed: frozenset({"tool", "runtime"}),
    ParallelGroupCompleted: frozenset({"runtime"}),
    ContextImported: frozenset({"agent"}),
    AgentCompleted: frozenset({"agent", "runtime"}),
    SubmissionAccepted: frozenset({"runtime"}),
    ExecutionFailed: frozenset({"runtime"}),
    ExecutionCompleted: frozenset({"runtime"}),
}


class _BoundSQLiteCommitStore:
    def __init__(
        self,
        store: SQLiteCommitStore,
        author: ActorRef,
        branch_id: str,
        execution_id: str | None,
    ) -> None:
        self._store = store
        self._author = author
        self._branch_id = branch_id
        self._execution_id = execution_id

    @property
    def author(self) -> ActorRef:
        return self._author

    @property
    def branch_id(self) -> str:
        return self._branch_id

    async def append(self, request: CommitRequest) -> Commit:
        if request.author != self._author:
            raise AuthorPermissionError("a bound commit session cannot change author")
        if request.branch_id != self._branch_id:
            raise AuthorPermissionError("a bound commit session cannot change branch")
        if (
            self._execution_id is not None
            and request.execution_id is not None
            and request.execution_id != self._execution_id
        ):
            raise AuthorPermissionError(
                "a bound commit session cannot change execution"
            )
        return await self._store._append(
            request,
            authenticated_author=self._author,
            bound_execution_id=self._execution_id,
        )


class _ExecutionSQLiteCommitStore:
    """A lightweight view that binds shared database access to one execution."""

    def __init__(self, store: SQLiteCommitStore, execution_id: str) -> None:
        self._store = store
        self.execution_id = execution_id

    def bind(
        self,
        author: ActorRef,
        *,
        branch_id: str = "main",
        execution_id: str | None = None,
    ) -> _BoundSQLiteCommitStore:
        if execution_id is not None and execution_id != self.execution_id:
            raise ValueError("execution-bound store cannot change execution")
        return self._store.bind(
            author, branch_id=branch_id, execution_id=self.execution_id
        )

    async def append(self, request: CommitRequest) -> Commit:
        values = request.model_dump(mode="python")
        values["execution_id"] = self.execution_id
        return await self._store.append(CommitRequest.model_validate(values))

    async def head(self, branch_id: str) -> Commit | None:
        return await self._store.head(branch_id, execution_id=self.execution_id)

    async def get_commit(self, commit_hash: str) -> Commit:
        return await self._store.get_commit(commit_hash, execution_id=self.execution_id)

    async def materialize(
        self, branch_id: str, at_hash: str | None = None
    ) -> ExecutionState:
        return await self._store.materialize(
            branch_id, at_hash, execution_id=self.execution_id
        )

    async def create_branch(
        self,
        *,
        branch_id: str,
        from_hash: str,
        execution_id: str | None = None,
    ) -> None:
        if execution_id is not None and execution_id != self.execution_id:
            raise ValueError("execution-bound store cannot change execution")
        await self._store.create_branch(
            branch_id=branch_id,
            from_hash=from_hash,
            execution_id=self.execution_id,
        )

    async def iter_commits(
        self,
        branch_id: str | None = None,
        *,
        after_sequence: int = -1,
        through_hash: str | None = None,
    ) -> AsyncIterator[Commit]:
        async for commit in self._store.iter_commits(
            branch_id,
            after_sequence=after_sequence,
            through_hash=through_hash,
            execution_id=self.execution_id,
        ):
            yield commit

    async def trace_commits(
        self, run_id: str, *, branch_id: str | None = None
    ) -> tuple[Commit, ...]:
        return await self._store.trace_commits(
            run_id, branch_id=branch_id, execution_id=self.execution_id
        )

    def for_execution(self, execution_id: str) -> _ExecutionSQLiteCommitStore:
        if execution_id != self.execution_id:
            raise ValueError("execution-bound store cannot change execution")
        return self


class SQLiteCommitStore:
    """One SQLite database containing small commits and occasional snapshots."""

    def __init__(
        self,
        path: str | Path,
        execution_id: str | None = None,
        *,
        snapshot_interval: int = 50,
    ) -> None:
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be at least 1")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.execution_id = execution_id
        self.snapshot_interval = snapshot_interval
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._reducer = EventReducer()
        self._projection_cache: dict[tuple[str, str], ExecutionState] = {}
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS commits (
                hash TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                execution_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                branch_sequence INTEGER NOT NULL,
                parent_hash TEXT REFERENCES commits(hash),
                based_on_hash TEXT REFERENCES commits(hash),
                author_json TEXT NOT NULL,
                author_kind TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_run_id TEXT NOT NULL,
                parent_run_id TEXT,
                event_type TEXT NOT NULL,
                event_json TEXT NOT NULL,
                action_id TEXT,
                invocation_id TEXT,
                requested_by_run_id TEXT,
                occurred_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                UNIQUE (execution_id, sequence),
                UNIQUE (execution_id, branch_id, branch_sequence)
            );

            CREATE TABLE IF NOT EXISTS branch_heads (
                execution_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                origin_hash TEXT REFERENCES commits(hash),
                head_hash TEXT REFERENCES commits(hash),
                PRIMARY KEY (execution_id, branch_id)
            );

            CREATE TABLE IF NOT EXISTS idempotency_keys (
                execution_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                commit_hash TEXT NOT NULL REFERENCES commits(hash),
                PRIMARY KEY (execution_id, request_id)
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                execution_id TEXT NOT NULL,
                commit_hash TEXT PRIMARY KEY REFERENCES commits(hash),
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS commits_parent_idx ON commits(parent_hash);
            CREATE INDEX IF NOT EXISTS commits_author_run_idx
                ON commits(execution_id, author_run_id, sequence);
            CREATE INDEX IF NOT EXISTS commits_action_idx
                ON commits(execution_id, action_id, sequence);
            CREATE INDEX IF NOT EXISTS commits_invocation_idx
                ON commits(execution_id, invocation_id, sequence);
            CREATE INDEX IF NOT EXISTS commits_requested_run_idx
                ON commits(execution_id, requested_by_run_id, sequence);
            """
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLiteCommitStore is closed")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> SQLiteCommitStore:  # noqa: PYI034
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

    async def __aenter__(self) -> SQLiteCommitStore:  # noqa: PYI034
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)

    def bind(
        self,
        author: ActorRef,
        *,
        branch_id: str = "main",
        execution_id: str | None = None,
    ) -> _BoundSQLiteCommitStore:
        return _BoundSQLiteCommitStore(
            self,
            author,
            branch_id,
            execution_id or self.execution_id,
        )

    def for_execution(self, execution_id: str) -> _ExecutionSQLiteCommitStore:
        if not execution_id:
            raise ValueError("execution_id cannot be empty")
        if self.execution_id is not None and self.execution_id != execution_id:
            raise ValueError("store is already bound to another execution")
        return _ExecutionSQLiteCommitStore(self, execution_id)

    def _execution(self, requested: str | None, bound: str | None = None) -> str:
        candidates = {value for value in (requested, bound, self.execution_id) if value}
        if len(candidates) > 1:
            raise ValueError("commit execution_id differs from the bound execution")
        if not candidates:
            raise ValueError(
                "execution_id is required on CommitRequest or SQLiteCommitStore"
            )
        return candidates.pop()

    @staticmethod
    def _correlation(event: Any) -> tuple[str | None, str | None, str | None]:
        return (
            getattr(event, "action_id", None),
            getattr(event, "invocation_id", None),
            getattr(event, "requested_by_run_id", None),
        )

    @staticmethod
    def _validate_author(request: CommitRequest, state: ExecutionState | None) -> None:
        allowed = _AUTHOR_KINDS.get(type(request.event))
        if allowed is None or request.author.kind not in allowed:
            raise AuthorPermissionError(
                f"{request.author.kind} actors cannot author {request.event.type}"
            )
        event = request.event
        author = request.author
        if (
            isinstance(
                event,
                AgentTurnRecorded
                | AgentStateUpdated
                | AgentSpawned
                | ContextImported
                | AgentCompleted,
            )
            and author.kind == "agent"
        ):
            run = None if state is None else state.agent_runs.get(author.run_id)
            if run is None or run.actor_id != author.actor_id:
                raise AuthorPermissionError("agent author is not a registered run")
            if run.parent_run_id != author.parent_run_id:
                raise AuthorPermissionError(
                    "agent parent identity does not match its run"
                )
            if (
                isinstance(event, AgentCompleted)
                and event.agent_run_id != author.run_id
            ):
                raise AuthorPermissionError("an agent cannot complete another run")
        if (
            isinstance(event, AgentSpawned)
            and event.context_cutoff_hash != request.based_on_hash
        ):
            raise CommitConflictError(
                "a child context cutoff must match the commit observed by its parent"
            )
        if isinstance(event, ToolCompleted):
            if event.invocation_id != author.run_id:
                raise AuthorPermissionError("tool author must match invocation_id")
            if state is not None:
                action = state.actions.get(event.action_id)
                if action is None or action.action.name != author.actor_id:
                    raise AuthorPermissionError("tool author does not match the action")
        if (
            isinstance(event, ToolFailed)
            and author.kind == "tool"
            and event.invocation_id != author.run_id
        ):
            raise AuthorPermissionError("tool author must match invocation_id")
        if (
            isinstance(event, ToolFailed)
            and author.kind == "tool"
            and state is not None
        ):
            action = state.actions.get(event.action_id)
            if action is None or action.action.name != author.actor_id:
                raise AuthorPermissionError("tool author does not match the action")

    @staticmethod
    def _row_to_commit(row: sqlite3.Row) -> Commit:
        try:
            return Commit.model_validate(
                {
                    "schema_version": row["schema_version"],
                    "hash": row["hash"],
                    "execution_id": row["execution_id"],
                    "branch_id": row["branch_id"],
                    "sequence": row["sequence"],
                    "branch_sequence": row["branch_sequence"],
                    "parent_hash": row["parent_hash"],
                    "based_on_hash": row["based_on_hash"],
                    "author": json.loads(row["author_json"]),
                    "event": json.loads(row["event_json"]),
                    "occurred_at": row["occurred_at"],
                    "recorded_at": row["recorded_at"],
                }
            )
        except Exception as exc:
            raise CommitIntegrityError(
                f"commit {row['hash']!r} failed schema or hash validation"
            ) from exc

    @staticmethod
    def _history_hashes(
        connection: sqlite3.Connection, head_hash: str | None
    ) -> list[str]:
        hashes: list[str] = []
        current = head_hash
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise CommitIntegrityError("commit parent history contains a cycle")
            seen.add(current)
            row = connection.execute(
                "SELECT hash, parent_hash FROM commits WHERE hash = ?", (current,)
            ).fetchone()
            if row is None:
                raise CommitIntegrityError(
                    f"history references missing commit {current!r}"
                )
            hashes.append(row["hash"])
            current = row["parent_hash"]
        return hashes

    def _materialize_locked(
        self,
        execution_id: str,
        branch_id: str,
        at_hash: str | None,
    ) -> ExecutionState:
        branch = self._connection.execute(
            "SELECT head_hash FROM branch_heads WHERE execution_id = ? AND branch_id = ?",
            (execution_id, branch_id),
        ).fetchone()
        if branch is None:
            raise CommitNotFoundError(f"branch {branch_id!r} was not found")
        selected = at_hash or branch["head_hash"]
        if selected is None:
            raise CommitNotFoundError(f"branch {branch_id!r} has no commits")
        history = self._history_hashes(self._connection, branch["head_hash"])
        if selected not in history:
            raise CommitNotFoundError(
                f"commit {selected!r} does not belong to branch {branch_id!r}"
            )

        chain: list[Commit] = []
        state: ExecutionState | None = None
        current: str | None = selected
        while current is not None:
            snapshot = self._connection.execute(
                "SELECT state_json FROM snapshots WHERE execution_id = ? AND commit_hash = ?",
                (execution_id, current),
            ).fetchone()
            if snapshot is not None:
                try:
                    state = ExecutionState.model_validate_json(snapshot["state_json"])
                except Exception as exc:
                    raise CommitIntegrityError(
                        f"snapshot at {current!r} is invalid"
                    ) from exc
                if state.through_commit_hash != current:
                    raise CommitIntegrityError(
                        "snapshot cutoff hash does not match its key"
                    )
                break
            row = self._connection.execute(
                "SELECT * FROM commits WHERE hash = ?", (current,)
            ).fetchone()
            if row is None:
                raise CommitIntegrityError(f"commit {current!r} was not found")
            commit = self._row_to_commit(row)
            if commit.execution_id != execution_id:
                raise CommitIntegrityError(
                    "branch history crosses execution identities"
                )
            chain.append(commit)
            current = commit.parent_hash

        for commit in reversed(chain):
            state = self._reducer.apply(state, commit)
        if state is None:  # pragma: no cover - selected guarantees data
            raise CommitIntegrityError("materialization produced no projection")
        if state.branch_id != branch_id:
            values = state.model_dump(mode="python")
            values["branch_id"] = branch_id
            state = ExecutionState.model_validate(values)
        return state

    async def append(self, request: CommitRequest) -> Commit:
        if request.author.kind != "runtime":
            raise AuthorPermissionError(
                "agent and tool commits require an author-bound store capability"
            )
        return await self._append(request, authenticated_author=request.author)

    async def _append(
        self,
        request: CommitRequest,
        *,
        authenticated_author: ActorRef,
        bound_execution_id: str | None = None,
    ) -> Commit:
        self._ensure_open()
        if request.author != authenticated_author:
            raise AuthorPermissionError(
                "request author does not match authenticated actor"
            )
        execution_id = self._execution(request.execution_id, bound_execution_id)
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    """SELECT c.* FROM idempotency_keys i
                       JOIN commits c ON c.hash = i.commit_hash
                       WHERE i.execution_id = ? AND i.request_id = ?""",
                    (execution_id, request.request_id),
                ).fetchone()
                if existing is not None:
                    persisted = self._row_to_commit(existing)
                    if persisted.author != authenticated_author:
                        raise AuthorPermissionError(
                            "idempotency key belongs to another authenticated author"
                        )
                    if persisted.branch_id != request.branch_id:
                        raise CommitConflictError(
                            "idempotency key belongs to another branch"
                        )
                    connection.commit()
                    return persisted

                branch = connection.execute(
                    "SELECT head_hash FROM branch_heads WHERE execution_id = ? AND branch_id = ?",
                    (execution_id, request.branch_id),
                ).fetchone()
                if branch is None:
                    any_commit = connection.execute(
                        "SELECT 1 FROM commits WHERE execution_id = ? LIMIT 1",
                        (execution_id,),
                    ).fetchone()
                    if any_commit is not None or not isinstance(
                        request.event, ExecutionStarted
                    ):
                        raise CommitNotFoundError(
                            f"branch {request.branch_id!r} was not explicitly created"
                        )
                    connection.execute(
                        "INSERT INTO branch_heads(execution_id, branch_id, origin_hash, head_hash) VALUES (?, ?, NULL, NULL)",
                        (execution_id, request.branch_id),
                    )
                    head_hash = None
                else:
                    head_hash = branch["head_hash"]

                history = set(self._history_hashes(connection, head_hash))
                if (
                    request.based_on_hash is not None
                    and request.based_on_hash not in history
                ):
                    raise CommitConflictError(
                        "based_on_hash is not part of the selected branch history"
                    )
                if head_hash is None and request.based_on_hash is not None:
                    raise CommitConflictError(
                        "the first commit cannot be based on a hash"
                    )
                if head_hash is not None and request.based_on_hash is None:
                    raise CommitConflictError(
                        "non-initial commits must record based_on_hash"
                    )
                if isinstance(request.event, ContextImported) and any(
                    commit_hash not in history
                    for commit_hash in request.event.source_commit_ids
                ):
                    raise CommitConflictError(
                        "imported source commits must belong to branch history"
                    )
                if isinstance(request.event, ContextImported):
                    source_rows = connection.execute(
                        """SELECT hash FROM commits
                           WHERE execution_id = ? AND (
                               author_run_id = ? OR requested_by_run_id = ? OR
                               json_extract(event_json, '$.agent_run_id') = ? OR
                               json_extract(event_json, '$.child_run_id') = ?
                           )""",
                        (
                            execution_id,
                            request.event.source_run_id,
                            request.event.source_run_id,
                            request.event.source_run_id,
                            request.event.source_run_id,
                        ),
                    ).fetchall()
                    source_hashes = {row["hash"] for row in source_rows}
                    if any(
                        commit_hash not in source_hashes
                        for commit_hash in request.event.source_commit_ids
                    ):
                        raise CommitConflictError(
                            "imported commits must belong to the selected source trace"
                        )
                if (
                    isinstance(request.event, AgentCompleted)
                    and request.event.trace_head is not None
                    and request.event.trace_head not in history
                ):
                    raise CommitConflictError(
                        "an agent trace head must belong to branch history"
                    )

                state = (
                    None
                    if head_hash is None
                    else self._materialize_locked(
                        execution_id, request.branch_id, head_hash
                    )
                )
                self._validate_author(request, state)
                sequence_row = connection.execute(
                    "SELECT MAX(sequence) AS value FROM commits WHERE execution_id = ?",
                    (execution_id,),
                ).fetchone()
                branch_sequence_row = connection.execute(
                    "SELECT MAX(branch_sequence) AS value FROM commits WHERE execution_id = ? AND branch_id = ?",
                    (execution_id, request.branch_id),
                ).fetchone()
                sequence = (
                    0
                    if sequence_row["value"] is None
                    else int(sequence_row["value"]) + 1
                )
                branch_sequence = (
                    0
                    if branch_sequence_row["value"] is None
                    else int(branch_sequence_row["value"]) + 1
                )
                recorded_at = _utc_now()
                occurred_at = request.occurred_at or recorded_at
                commit = Commit.create(
                    execution_id=execution_id,
                    branch_id=request.branch_id,
                    sequence=sequence,
                    branch_sequence=branch_sequence,
                    parent_hash=head_hash,
                    based_on_hash=request.based_on_hash,
                    author=request.author,
                    event=request.event,
                    occurred_at=occurred_at,
                    recorded_at=recorded_at,
                )
                try:
                    projected = self._reducer.apply(state, commit)
                except SharedStateConflictError as exc:
                    raise CommitConflictError(str(exc)) from exc
                except ReducerError as exc:
                    raise CommitConflictError(str(exc)) from exc

                action_id, invocation_id, requested_by_run_id = self._correlation(
                    request.event
                )
                connection.execute(
                    """INSERT INTO commits(
                           hash, schema_version, execution_id, branch_id, sequence,
                           branch_sequence, parent_hash, based_on_hash, author_json,
                           author_kind, author_id, author_run_id, parent_run_id,
                           event_type, event_json, action_id, invocation_id,
                           requested_by_run_id, occurred_at, recorded_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        commit.hash,
                        commit.schema_version,
                        commit.execution_id,
                        commit.branch_id,
                        commit.sequence,
                        commit.branch_sequence,
                        commit.parent_hash,
                        commit.based_on_hash,
                        canonical_json(commit.author.model_dump(mode="json")),
                        commit.author.kind,
                        commit.author.actor_id,
                        commit.author.run_id,
                        commit.author.parent_run_id,
                        commit.event.type,
                        canonical_json(commit.event.model_dump(mode="json")),
                        action_id,
                        invocation_id,
                        requested_by_run_id,
                        commit.occurred_at.isoformat(),
                        commit.recorded_at.isoformat(),
                    ),
                )
                connection.execute(
                    "INSERT INTO idempotency_keys(execution_id, request_id, commit_hash) VALUES (?, ?, ?)",
                    (execution_id, request.request_id, commit.hash),
                )
                connection.execute(
                    "UPDATE branch_heads SET head_hash = ? WHERE execution_id = ? AND branch_id = ?",
                    (commit.hash, execution_id, request.branch_id),
                )
                if commit.sequence % self.snapshot_interval == 0:
                    connection.execute(
                        "INSERT OR REPLACE INTO snapshots(execution_id, commit_hash, state_json, created_at) VALUES (?, ?, ?, ?)",
                        (
                            execution_id,
                            commit.hash,
                            projected.model_dump_json(),
                            recorded_at.isoformat(),
                        ),
                    )
                connection.commit()
                self._projection_cache[(execution_id, request.branch_id)] = projected
                return commit
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise

    async def head(
        self, branch_id: str, *, execution_id: str | None = None
    ) -> Commit | None:
        self._ensure_open()
        execution_id = self._execution(execution_id)
        with self._lock:
            row = self._connection.execute(
                """SELECT c.* FROM branch_heads b LEFT JOIN commits c ON c.hash = b.head_hash
                   WHERE b.execution_id = ? AND b.branch_id = ?""",
                (execution_id, branch_id),
            ).fetchone()
        if row is None or row["hash"] is None:
            return None
        return self._row_to_commit(row)

    async def get_commit(
        self, commit_hash: str, *, execution_id: str | None = None
    ) -> Commit:
        self._ensure_open()
        execution_id = self._execution(execution_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM commits WHERE execution_id = ? AND hash = ?",
                (execution_id, commit_hash),
            ).fetchone()
        if row is None:
            raise CommitNotFoundError(f"commit {commit_hash!r} was not found")
        return self._row_to_commit(row)

    async def materialize(
        self,
        branch_id: str,
        at_hash: str | None = None,
        *,
        execution_id: str | None = None,
    ) -> ExecutionState:
        self._ensure_open()
        execution_id = self._execution(execution_id)
        with self._lock:
            if at_hash is None:
                cached = self._projection_cache.get((execution_id, branch_id))
                head = self._connection.execute(
                    "SELECT head_hash FROM branch_heads WHERE execution_id = ? AND branch_id = ?",
                    (execution_id, branch_id),
                ).fetchone()
                if (
                    cached is not None
                    and head is not None
                    and cached.through_commit_hash == head["head_hash"]
                ):
                    return cached
            state = self._materialize_locked(execution_id, branch_id, at_hash)
            if at_hash is None:
                self._projection_cache[(execution_id, branch_id)] = state
            return state

    async def create_branch(
        self,
        *,
        branch_id: str,
        from_hash: str,
        execution_id: str | None = None,
    ) -> None:
        self._ensure_open()
        selected_execution = self._execution(execution_id)
        if not branch_id:
            raise ValueError("branch_id cannot be empty")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                source = connection.execute(
                    "SELECT execution_id FROM commits WHERE hash = ?", (from_hash,)
                ).fetchone()
                if source is None:
                    raise CommitNotFoundError(f"commit {from_hash!r} was not found")
                if source["execution_id"] != selected_execution:
                    raise CommitConflictError(
                        "branch source belongs to another execution"
                    )
                existing = connection.execute(
                    "SELECT 1 FROM branch_heads WHERE execution_id = ? AND branch_id = ?",
                    (selected_execution, branch_id),
                ).fetchone()
                if existing is not None:
                    raise CommitConflictError(f"branch {branch_id!r} already exists")
                connection.execute(
                    "INSERT INTO branch_heads(execution_id, branch_id, origin_hash, head_hash) VALUES (?, ?, ?, ?)",
                    (selected_execution, branch_id, from_hash, from_hash),
                )
                connection.commit()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise

    async def iter_commits(
        self,
        branch_id: str | None = None,
        *,
        after_sequence: int = -1,
        through_hash: str | None = None,
        execution_id: str | None = None,
    ) -> AsyncIterator[Commit]:
        self._ensure_open()
        execution_id = self._execution(execution_id)
        with self._lock:
            if branch_id is None:
                if through_hash is not None:
                    row = self._connection.execute(
                        "SELECT sequence FROM commits WHERE execution_id = ? AND hash = ?",
                        (execution_id, through_hash),
                    ).fetchone()
                    if row is None:
                        raise CommitNotFoundError(
                            f"commit {through_hash!r} was not found"
                        )
                    maximum = row["sequence"]
                else:
                    maximum = 2**63 - 1
                rows = self._connection.execute(
                    "SELECT * FROM commits WHERE execution_id = ? AND sequence > ? AND sequence <= ? ORDER BY sequence",
                    (execution_id, after_sequence, maximum),
                ).fetchall()
            else:
                branch = self._connection.execute(
                    "SELECT head_hash FROM branch_heads WHERE execution_id = ? AND branch_id = ?",
                    (execution_id, branch_id),
                ).fetchone()
                if branch is None:
                    raise CommitNotFoundError(f"branch {branch_id!r} was not found")
                head_hash = through_hash or branch["head_hash"]
                branch_history = self._history_hashes(
                    self._connection, branch["head_hash"]
                )
                if head_hash not in branch_history:
                    raise CommitNotFoundError(
                        f"commit {head_hash!r} does not belong to branch {branch_id!r}"
                    )
                hashes = self._history_hashes(self._connection, head_hash)
                if not hashes:
                    rows = []
                else:
                    placeholders = ",".join("?" for _ in hashes)
                    rows = self._connection.execute(
                        f"SELECT * FROM commits WHERE hash IN ({placeholders}) AND sequence > ? ORDER BY sequence",
                        (*hashes, after_sequence),
                    ).fetchall()
        for row in rows:
            yield self._row_to_commit(row)

    async def trace_commits(
        self,
        run_id: str,
        *,
        branch_id: str | None = None,
        execution_id: str | None = None,
    ) -> tuple[Commit, ...]:
        """Return commits authored by or directly delivered to one run."""
        execution_id = self._execution(execution_id)
        if branch_id is not None:
            branch_commits = [
                commit
                async for commit in self.iter_commits(
                    branch_id, execution_id=execution_id
                )
            ]
            return tuple(
                commit
                for commit in branch_commits
                if commit.author.run_id == run_id
                or getattr(commit.event, "requested_by_run_id", None) == run_id
                or getattr(commit.event, "agent_run_id", None) == run_id
                or getattr(commit.event, "child_run_id", None) == run_id
            )
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM commits
                   WHERE execution_id = ? AND (
                       author_run_id = ? OR requested_by_run_id = ? OR
                       json_extract(event_json, '$.agent_run_id') = ? OR
                       json_extract(event_json, '$.child_run_id') = ?
                   ) ORDER BY sequence""",
                (execution_id, run_id, run_id, run_id, run_id),
            ).fetchall()
        return tuple(self._row_to_commit(row) for row in rows)

    def snapshot_count(self) -> int:
        """Return the number of projections persisted as replay accelerators."""
        execution_id = self._execution(None)
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS value FROM snapshots WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        return int(row["value"])


__all__ = ["SQLiteCommitStore"]
