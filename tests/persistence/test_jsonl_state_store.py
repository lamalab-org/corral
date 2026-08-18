import asyncio
import json

import pytest

from corral.core.state import State
from corral.persistence import (
    JSONLStateStore,
    StateIntegrityError,
    StateNotFoundError,
    StateTransitionConflictError,
)


def run(coro):
    return asyncio.run(coro)


def make_state(state_id: str = "state-1", **kwargs) -> State:
    return State(id=state_id, **kwargs)


def test_save_and_load_complete_state_by_hash(tmp_path):
    path = tmp_path / "states.jsonl"
    state = make_state(environment={"value": 1})

    with JSONLStateStore(path) as store:
        saved = run(store.save(state))

        assert saved == state
        assert run(store.save(state)) == state
        assert run(store.load(state.state_hash)) == state
        assert run(store.children(state.state_hash)) == ()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {
            "record": "state",
            "state_hash": state.state_hash,
            "state": state.model_dump(mode="json"),
        }
    ]


def test_generated_state_ids_do_not_collide_in_shared_store(tmp_path):
    first = State()
    second = State()

    with JSONLStateStore(tmp_path / "states.jsonl") as store:
        run(store.save(first))
        run(store.save(second))

        assert first.id != second.id
        assert run(store.load(first.state_hash)) == first
        assert run(store.load(second.state_hash)) == second


def test_load_initial_uses_execution_identity_and_rejects_a_second_root(tmp_path):
    initial = make_state(environment={"value": 1})
    conflicting = make_state(environment={"value": 2})

    with JSONLStateStore(tmp_path / "states.jsonl") as store:
        assert run(store.load_initial(initial.id)) is None
        run(store.save(initial))
        assert run(store.load_initial(initial.id)) == initial

        with pytest.raises(StateTransitionConflictError, match="initial State"):
            run(store.save(conflicting))


def test_save_linked_fork_and_idempotent_transition_retry(tmp_path):
    parent = make_state()
    child = parent.fork(
        environment={"result": 42},
        messages=({"role": "tool", "content": "42"},),
    )

    with JSONLStateStore(tmp_path / "states.jsonl") as store:
        run(store.save(parent))
        saved = run(store.save(child, transition_id="tool-result-1"))
        retried = run(store.save(child, transition_id="tool-result-1"))

        assert saved == retried == child
        assert run(store.load(child.state_hash)) == child
        assert run(store.children(parent.state_hash)) == (child,)
        assert run(store.load_transition(parent.state_hash, "tool-result-1")) == child
        assert run(store.load_transition(parent.state_hash, "missing")) is None


def test_append_only_head_tracks_the_canonical_state_branch(tmp_path):
    root = make_state()
    canonical = root.fork(environment={"branch": "canonical"})
    speculative = root.fork(environment={"branch": "speculative"})

    with JSONLStateStore(tmp_path / "states.jsonl") as store:
        run(store.save(root, advance_head=True))
        run(
            store.save(
                canonical,
                transition_id="canonical",
                advance_head=True,
            )
        )
        run(store.save(speculative, transition_id="speculative"))

        assert run(store.load_head(root.id)) == canonical

        # Re-saving the root is idempotent and must not rewind the ref.
        run(store.save(root, advance_head=True))
        assert run(store.load_head(root.id)) == canonical

        with pytest.raises(StateTransitionConflictError, match="head"):
            run(
                store.save(
                    speculative,
                    transition_id="speculative",
                    advance_head=True,
                )
            )


def test_multiple_sibling_forks_are_persisted(tmp_path):
    parent = make_state()
    first = parent.fork(environment={"winner": "first"})
    second = parent.fork(environment={"winner": "second"})
    path = tmp_path / "states.jsonl"

    with (
        JSONLStateStore(path) as first_store,
        JSONLStateStore(path) as second_store,
    ):
        run(first_store.save(parent))
        run(first_store.save(first, transition_id="first"))
        run(second_store.save(second, transition_id="second"))

        children = run(first_store.children(parent.state_hash))
        assert {child.state_hash for child in children} == {
            first.state_hash,
            second.state_hash,
        }


def test_transition_retry_with_different_child_fails(tmp_path):
    parent = make_state()
    first = parent.fork(environment={"result": "first"})
    changed_retry = parent.fork(environment={"result": "different"})

    with JSONLStateStore(tmp_path / "states.jsonl") as store:
        run(store.save(parent))
        run(store.save(first, transition_id="same-transition"))

        with pytest.raises(StateTransitionConflictError, match="different child"):
            run(store.save(changed_retry, transition_id="same-transition"))


def test_save_rejects_missing_parent(tmp_path):
    child = make_state(
        revision=1,
        parent_revision=0,
        parent_hash="a" * 64,
    )

    with (
        JSONLStateStore(tmp_path / "states.jsonl") as store,
        pytest.raises(StateNotFoundError, match="parent State"),
    ):
        run(store.save(child))


def test_save_rejects_child_that_was_not_forked_from_parent(tmp_path):
    parent = make_state(metadata={"model": {"name": "original"}})
    invalid_child = State(
        id=parent.id,
        revision=1,
        parent_revision=0,
        parent_hash=parent.state_hash,
        metadata={"model": {"name": "changed"}},
    )

    with JSONLStateStore(tmp_path / "states.jsonl") as store:
        run(store.save(parent))
        with pytest.raises(StateIntegrityError, match="valid fork"):
            run(store.save(invalid_child))


def test_store_detects_tampered_state_json(tmp_path):
    path = tmp_path / "states.jsonl"
    state = make_state(environment={"untampered": True})

    with JSONLStateStore(path) as store:
        run(store.save(state))

    record = json.loads(path.read_text())
    record["state"]["environment"] = {"tampered": True}
    path.write_text(json.dumps(record) + "\n")

    with (
        JSONLStateStore(path) as store,
        pytest.raises(StateIntegrityError, match="content-hash"),
    ):
        run(store.load(state.state_hash))


def test_store_rejects_malformed_jsonl(tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text('{"record":"state"')

    with (
        JSONLStateStore(path) as store,
        pytest.raises(StateIntegrityError, match="line 1"),
    ):
        run(store.load("missing"))


def test_missing_state_errors_are_explicit(tmp_path):
    with JSONLStateStore(tmp_path / "states.jsonl") as store:
        with pytest.raises(StateNotFoundError):
            run(store.load("missing"))
        with pytest.raises(StateNotFoundError):
            run(store.children("missing"))
