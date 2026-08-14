"""Content-addressed SQLite/WAL StateStore implementation."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from corral.core.state import State
from corral.persistence.base import (
    StateIntegrityError,
    StateNotFoundError,
    StateTransitionConflictError,
)


class SQLiteStateStore:
    """Persist complete immutable States as a forkable content-addressed DAG."""

    def __init__(self, path: str | Path) -> None:
        self.path = (
            str(path) if str(path) == ":memory:" else str(Path(path).expanduser())
        )
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS states (
                    state_hash TEXT PRIMARY KEY,
                    state_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 0),
                    parent_revision INTEGER,
                    parent_hash TEXT,
                    schema_version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    FOREIGN KEY (parent_hash) REFERENCES states (state_hash)
                );

                CREATE INDEX IF NOT EXISTS states_parent_hash
                    ON states (parent_hash);

                CREATE TABLE IF NOT EXISTS state_transitions (
                    parent_hash TEXT NOT NULL,
                    transition_id TEXT NOT NULL,
                    child_hash TEXT NOT NULL,
                    PRIMARY KEY (parent_hash, transition_id),
                    FOREIGN KEY (parent_hash) REFERENCES states (state_hash),
                    FOREIGN KEY (child_hash) REFERENCES states (state_hash)
                );
                """
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLiteStateStore is closed")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> SQLiteStateStore:  # noqa: PYI034
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

    async def __aenter__(self) -> SQLiteStateStore:  # noqa: PYI034
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)

    def _begin(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.rollback()

    @staticmethod
    def _state_from_row(row: sqlite3.Row, *, expected_hash: str | None = None) -> State:
        try:
            state = State.from_json(row["state_json"])
        except (TypeError, ValueError) as exc:
            raise StateIntegrityError(
                "persisted State is not valid State JSON"
            ) from exc

        if expected_hash is not None and row["state_hash"] != expected_hash:
            raise StateIntegrityError(
                "stored State hash does not match the requested hash"
            )
        if state.state_hash != row["state_hash"]:
            raise StateIntegrityError(
                f"State {row['state_hash']!r} failed its content-hash check"
            )
        if (
            state.id != row["state_id"]
            or state.revision != row["revision"]
            or state.parent_revision != row["parent_revision"]
            or state.parent_hash != row["parent_hash"]
            or state.schema_version != row["schema_version"]
        ):
            raise StateIntegrityError(
                f"State {row['state_hash']!r} has inconsistent index metadata"
            )
        return state

    def _row_for_hash(self, state_hash: str) -> sqlite3.Row | None:
        return self._connection.execute(
            """
            SELECT state_hash, state_id, revision, parent_revision,
                   parent_hash, schema_version, state_json
            FROM states WHERE state_hash = ?
            """,
            (state_hash,),
        ).fetchone()

    def _validate_parent(self, state: State) -> None:
        if state.parent_hash is None:
            return

        parent_row = self._row_for_hash(state.parent_hash)
        if parent_row is None:
            raise StateNotFoundError(
                f"parent State {state.parent_hash!r} was not found"
            )
        parent = self._state_from_row(parent_row, expected_hash=state.parent_hash)
        try:
            expected = parent.fork(
                messages=state.messages,
                environment=state.environment,
                workspace=state.workspace,
                usage=state.usage,
                runtime=state.runtime,
                dependency_outputs=state.dependency_outputs,
            )
        except (TypeError, ValueError) as exc:
            raise StateIntegrityError(
                f"State {state.state_hash!r} violates its parent invariants"
            ) from exc
        if expected != state:
            raise StateIntegrityError(
                f"State {state.state_hash!r} is not a valid fork of its parent"
            )

    async def save(
        self,
        state: State,
        transition_id: str | None = None,
    ) -> State:
        """Persist a complete State; identical saves and transitions are idempotent."""
        self._ensure_open()
        if transition_id is not None and not transition_id.strip():
            raise ValueError("transition_id cannot be empty")
        if state.parent_hash is None and transition_id is not None:
            raise ValueError("an initial State cannot have a transition_id")

        state_hash = state.state_hash
        with self._lock:
            self._begin()
            try:
                self._validate_parent(state)

                existing = self._row_for_hash(state_hash)
                if existing is None:
                    self._connection.execute(
                        """
                        INSERT INTO states (
                            state_hash, state_id, revision, parent_revision,
                            parent_hash, schema_version, state_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            state_hash,
                            state.id,
                            state.revision,
                            state.parent_revision,
                            state.parent_hash,
                            state.schema_version,
                            state.to_json(),
                        ),
                    )
                    persisted = state
                else:
                    persisted = self._state_from_row(
                        existing,
                        expected_hash=state_hash,
                    )
                    if persisted != state:
                        raise StateIntegrityError(
                            f"content hash collision for State {state_hash!r}"
                        )

                if transition_id is not None:
                    transition = self._connection.execute(
                        """
                        SELECT child_hash FROM state_transitions
                        WHERE parent_hash = ? AND transition_id = ?
                        """,
                        (state.parent_hash, transition_id),
                    ).fetchone()
                    if transition is not None:
                        if transition["child_hash"] != state_hash:
                            raise StateTransitionConflictError(
                                f"transition {transition_id!r} already points to "
                                "a different child State"
                            )
                    else:
                        self._connection.execute(
                            """
                            INSERT INTO state_transitions (
                                parent_hash, transition_id, child_hash
                            ) VALUES (?, ?, ?)
                            """,
                            (state.parent_hash, transition_id, state_hash),
                        )

                self._connection.commit()
                return persisted
            except Exception:
                self._rollback()
                raise

    async def load(self, state_hash: str) -> State:
        self._ensure_open()
        with self._lock:
            row = self._row_for_hash(state_hash)
        if row is None:
            raise StateNotFoundError(f"State {state_hash!r} was not found")
        return self._state_from_row(row, expected_hash=state_hash)

    async def children(self, state_hash: str) -> tuple[State, ...]:
        self._ensure_open()
        with self._lock:
            parent_row = self._row_for_hash(state_hash)
            if parent_row is None:
                raise StateNotFoundError(f"State {state_hash!r} was not found")
            self._state_from_row(parent_row, expected_hash=state_hash)
            rows = self._connection.execute(
                """
                SELECT state_hash, state_id, revision, parent_revision,
                       parent_hash, schema_version, state_json
                FROM states
                WHERE parent_hash = ?
                ORDER BY state_hash
                """,
                (state_hash,),
            ).fetchall()
        return tuple(self._state_from_row(row) for row in rows)
