"""Append-only JSONL StateStore implementation."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

from corral.core.state import State
from corral.persistence.base import (
    StateIntegrityError,
    StateNotFoundError,
    StateTransitionConflictError,
)

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class JSONLStateStore:
    """Persist complete immutable States in an append-only JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._initialize_lock_file()
        self._thread_lock = _thread_lock(self.path)
        self._closed = False

    def _initialize_lock_file(self) -> None:
        with self._lock_path.open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("JSONLStateStore is closed")

    def close(self) -> None:
        with self._thread_lock:
            self._closed = True

    def __enter__(self) -> JSONLStateStore:  # noqa: PYI034
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

    async def __aenter__(self) -> JSONLStateStore:  # noqa: PYI034
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock, self._lock_path.open("r+b") as lock_handle:
            _lock_file(lock_handle)
            try:
                yield
            finally:
                _unlock_file(lock_handle)

    @staticmethod
    def _state_from_record(record: Mapping[str, Any], line_number: int) -> State:
        state_hash = record.get("state_hash")
        if not isinstance(state_hash, str):
            raise StateIntegrityError(
                f"JSONL State record on line {line_number} has no valid state_hash"
            )
        raw_state = record.get("state")
        if not isinstance(raw_state, Mapping):
            raise StateIntegrityError(
                f"JSONL State record on line {line_number} has no State object"
            )
        try:
            state = State.from_dict(raw_state)
        except (TypeError, ValueError) as exc:
            raise StateIntegrityError(
                f"JSONL State record on line {line_number} is not valid State JSON"
            ) from exc
        if state.state_hash != state_hash:
            raise StateIntegrityError(
                f"State {state_hash!r} failed its content-hash check"
            )
        return state

    def _read_index(
        self,
    ) -> tuple[
        dict[str, State],
        dict[tuple[str, str], str],
        dict[str, set[str]],
        dict[str, str],
    ]:
        states: dict[str, State] = {}
        transitions: dict[tuple[str, str], str] = {}
        children: dict[str, set[str]] = {}
        heads: dict[str, str] = {}

        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except (TypeError, ValueError) as exc:
                    raise StateIntegrityError(
                        f"invalid JSONL record on line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise StateIntegrityError(
                        f"JSONL record on line {line_number} is not an object"
                    )

                record_type = record.get("record")
                if record_type == "state":
                    state = self._state_from_record(record, line_number)
                    state_hash = state.state_hash
                    existing = states.get(state_hash)
                    if existing is not None and existing != state:
                        raise StateIntegrityError(
                            f"content hash collision for State {state_hash!r}"
                        )
                    if (
                        state.parent_hash is not None
                        and state.parent_hash not in states
                    ):
                        raise StateIntegrityError(
                            f"State {state_hash!r} refers to a missing parent State"
                        )
                    states[state_hash] = state
                    if state.parent_hash is not None:
                        children.setdefault(state.parent_hash, set()).add(state_hash)
                    continue

                if record_type == "transition":
                    parent_hash = record.get("parent_hash")
                    transition_id = record.get("transition_id")
                    child_hash = record.get("child_hash")
                    if not all(
                        isinstance(value, str)
                        for value in (parent_hash, transition_id, child_hash)
                    ):
                        raise StateIntegrityError(
                            "JSONL transition record on line "
                            f"{line_number} has invalid fields"
                        )
                    if not transition_id.strip():
                        raise StateIntegrityError(
                            "JSONL transition record on line "
                            f"{line_number} has an empty transition_id"
                        )
                    if parent_hash not in states or child_hash not in states:
                        raise StateIntegrityError(
                            "JSONL transition record on line "
                            f"{line_number} refers to a missing State"
                        )
                    if states[child_hash].parent_hash != parent_hash:
                        raise StateIntegrityError(
                            "JSONL transition record on line "
                            f"{line_number} does not match the child's parent"
                        )
                    key = (parent_hash, transition_id)
                    existing_child = transitions.get(key)
                    if existing_child is not None and existing_child != child_hash:
                        raise StateIntegrityError(
                            "JSONL transition record on line "
                            f"{line_number} conflicts with an earlier transition"
                        )
                    transitions[key] = child_hash
                    continue

                if record_type == "head":
                    state_id = record.get("state_id")
                    state_hash = record.get("state_hash")
                    previous_hash = record.get("previous_hash")
                    if not isinstance(state_id, str) or not isinstance(state_hash, str):
                        raise StateIntegrityError(
                            f"JSONL head record on line {line_number} has invalid fields"
                        )
                    if previous_hash is not None and not isinstance(previous_hash, str):
                        raise StateIntegrityError(
                            "JSONL head record on line "
                            f"{line_number} has an invalid previous_hash"
                        )
                    state = states.get(state_hash)
                    if state is None or state.id != state_id:
                        raise StateIntegrityError(
                            "JSONL head record on line "
                            f"{line_number} refers to an invalid State"
                        )
                    current = heads.get(state_id)
                    if current != previous_hash:
                        raise StateIntegrityError(
                            "JSONL head record on line "
                            f"{line_number} does not continue the recorded head"
                        )
                    if previous_hash is not None and state.parent_hash != previous_hash:
                        raise StateIntegrityError(
                            "JSONL head record on line "
                            f"{line_number} does not advance to a direct child"
                        )
                    heads[state_id] = state_hash
                    continue

                raise StateIntegrityError(
                    f"JSONL record on line {line_number} has an unknown record type"
                )

        return states, transitions, children, heads

    @staticmethod
    def _validate_parent(state: State, states: Mapping[str, State]) -> None:
        if state.parent_hash is None:
            return

        parent = states.get(state.parent_hash)
        if parent is None:
            raise StateNotFoundError(
                f"parent State {state.parent_hash!r} was not found"
            )
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

    def _append(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        payload = "".join(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    async def save(
        self,
        state: State,
        transition_id: str | None = None,
        *,
        advance_head: bool = False,
    ) -> State:
        """Append a State snapshot, transition, and optional atomic head move."""
        self._ensure_open()
        if transition_id is not None and not transition_id.strip():
            raise ValueError("transition_id cannot be empty")
        if state.parent_hash is None and transition_id is not None:
            raise ValueError("an initial State cannot have a transition_id")

        state_hash = state.state_hash
        with self._locked():
            states, transitions, _, heads = self._read_index()
            self._validate_parent(state, states)

            if state.revision == 0:
                existing_initial = next(
                    (
                        candidate
                        for candidate in states.values()
                        if candidate.id == state.id and candidate.revision == 0
                    ),
                    None,
                )
                if existing_initial is not None and existing_initial != state:
                    raise StateTransitionConflictError(
                        f"execution {state.id!r} already has a different initial State"
                    )

            records: list[dict[str, Any]] = []
            persisted = states.get(state_hash)
            if persisted is None:
                records.append(
                    {
                        "record": "state",
                        "state_hash": state_hash,
                        "state": state.model_dump(mode="json"),
                    }
                )
                persisted = state
            elif persisted != state:
                raise StateIntegrityError(
                    f"content hash collision for State {state_hash!r}"
                )

            if transition_id is not None:
                key = (state.parent_hash, transition_id)
                existing_child = transitions.get(key)
                if existing_child is not None and existing_child != state_hash:
                    raise StateTransitionConflictError(
                        f"transition {transition_id!r} already points to "
                        "a different child State"
                    )
                if existing_child is None:
                    records.append(
                        {
                            "record": "transition",
                            "parent_hash": state.parent_hash,
                            "transition_id": transition_id,
                            "child_hash": state_hash,
                        }
                    )

            if advance_head:
                current_head = heads.get(state.id)
                if current_head == state_hash:
                    pass
                elif state.parent_hash is None:
                    # Re-saving the root of an execution must never rewind a
                    # head that has already advanced to a later checkpoint.
                    if current_head is None:
                        records.append(
                            {
                                "record": "head",
                                "state_id": state.id,
                                "previous_hash": None,
                                "state_hash": state_hash,
                            }
                        )
                elif current_head != state.parent_hash:
                    raise StateTransitionConflictError(
                        f"execution {state.id!r} head is not the State parent"
                    )
                else:
                    records.append(
                        {
                            "record": "head",
                            "state_id": state.id,
                            "previous_hash": current_head,
                            "state_hash": state_hash,
                        }
                    )

            self._append(records)
            return persisted

    async def load(self, state_hash: str) -> State:
        self._ensure_open()
        with self._locked():
            states, _, _, _ = self._read_index()
        state = states.get(state_hash)
        if state is None:
            raise StateNotFoundError(f"State {state_hash!r} was not found")
        return state

    async def load_initial(self, state_id: str) -> State | None:
        self._ensure_open()
        if not state_id:
            raise ValueError("state_id cannot be empty")
        with self._locked():
            states, _, _, _ = self._read_index()
        matches = [
            state
            for state in states.values()
            if state.id == state_id and state.revision == 0
        ]
        if len(matches) > 1:
            raise StateIntegrityError(
                f"execution {state_id!r} has multiple initial States"
            )
        return matches[0] if matches else None

    async def load_head(self, state_id: str) -> State | None:
        """Load the State referenced by the execution's append-only head log."""
        self._ensure_open()
        if not state_id:
            raise ValueError("state_id cannot be empty")
        with self._locked():
            states, _, _, heads = self._read_index()
        state_hash = heads.get(state_id)
        return None if state_hash is None else states[state_hash]

    async def children(self, state_hash: str) -> tuple[State, ...]:
        self._ensure_open()
        with self._locked():
            states, _, children, _ = self._read_index()
        if state_hash not in states:
            raise StateNotFoundError(f"State {state_hash!r} was not found")
        return tuple(
            states[child_hash] for child_hash in sorted(children.get(state_hash, set()))
        )

    async def load_transition(
        self,
        parent_hash: str,
        transition_id: str,
    ) -> State | None:
        """Load the child selected by an idempotent parent transition."""
        self._ensure_open()
        if not transition_id.strip():
            raise ValueError("transition_id cannot be empty")
        with self._locked():
            states, transitions, _, _ = self._read_index()
        if parent_hash not in states:
            raise StateNotFoundError(f"State {parent_hash!r} was not found")
        child_hash = transitions.get((parent_hash, transition_id))
        return None if child_hash is None else states[child_hash]
