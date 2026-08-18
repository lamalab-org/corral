import copy
import hashlib
import json
import pickle
from uuid import UUID

import pytest
from pydantic import ValidationError

from corral.core.state import (
    State,
    checkpoint_state,
)


def make_state(**kwargs) -> State:
    return State(id="state-1", **kwargs)


def test_state_hash_is_stable_across_serialization_and_mapping_order():
    left = make_state(environment={"b": 2, "a": {"d": 4, "c": [1, 2]}})
    right = make_state(environment={"a": {"c": [1, 2], "d": 4}, "b": 2})

    restored = State.from_json(left.to_json())

    assert left.state_hash == right.state_hash
    assert restored == left
    assert restored.state_hash == left.state_hash
    assert json.loads(left.to_json())["schema_version"] == 2


def test_state_mints_globally_unique_ids():
    first = State()
    second = State()

    assert first.id != second.id
    assert str(UUID(first.id)) == first.id
    assert str(UUID(second.id)) == second.id


def test_checkpoint_state_squashes_in_memory_forks_into_one_direct_child():
    parent = make_state(messages=({"role": "user", "content": "start"},))
    working = parent.fork(
        messages=(*parent.messages, {"role": "assistant", "content": "thinking"})
    )
    working = working.fork(environment={"temperature": 298.15})

    checkpoint = checkpoint_state(parent, working)

    assert checkpoint.revision == parent.revision + 1
    assert checkpoint.parent_hash == parent.state_hash
    assert checkpoint.messages == working.messages
    assert checkpoint.environment == working.environment


def test_state_metadata_describes_the_execution_without_benchmark_identity():
    state = make_state(
        metadata={
            "model": {"provider": "openai", "name": "test-model"},
            "scaffold": {"name": "react"},
            "environment": {"name": "wetlab"},
            "task": {"prompt": "measure the sample"},
            "extra": {"temperature": 0.1},
        }
    )

    assert state.metadata.model["name"] == "test-model"
    assert state.metadata.scaffold == {"name": "react"}
    assert "benchmark_run_id" not in state.to_json()


def test_artifacts_are_owned_by_the_workspace():
    state = make_state(workspace={"files": {}, "artifacts": {}})
    digest = hashlib.sha256(b"report").hexdigest()
    workspace = state.workspace.fork(
        files={
            "report.pdf": {
                "path": "report.pdf",
                "sha256": digest,
                "size": 6,
                "blob_ref": f"sha256:{digest}",
            }
        },
        artifacts={
            "final_report": {
                "path": "report.pdf",
                "kind": "report",
                "metadata": {},
            }
        },
    )
    child = state.fork(workspace=workspace)

    assert child.workspace.artifacts["final_report"].path == "report.pdf"
    assert "artifacts" not in type(child).model_fields
    with pytest.raises(ValidationError, match="artifacts"):
        make_state(artifacts={"final_report": {"path": "report.pdf"}})


def test_state_is_deeply_immutable():
    state = make_state(
        environment={"inventory": {"sample": [1, 2]}},
        messages=({"role": "user", "content": "measure"},),
    )

    with pytest.raises(ValidationError):
        state.revision = 3
    with pytest.raises(TypeError, match=r"State\.fork"):
        state.environment["inventory"]["sample"].append(3)
    with pytest.raises(TypeError, match=r"State\.fork"):
        state.model_copy(update={"revision": 3})


def test_immutable_state_supports_copy_and_pickle_round_trips():
    state = make_state(
        environment={"inventory": {"sample": [1, 2]}},
        messages=({"role": "user", "content": "measure"},),
    )

    copies = (
        copy.copy(state),
        copy.deepcopy(state),
        state.model_copy(deep=True),
        pickle.loads(pickle.dumps(state)),
    )

    for restored in copies:
        assert restored == state
        assert restored.state_hash == state.state_hash
        with pytest.raises(TypeError, match=r"State\.fork"):
            restored.environment["inventory"]["sample"].append(3)


@pytest.mark.parametrize(
    "invalid_hash",
    [
        "+" + "a" * 63,
        " " + "a" * 63,
        "A" * 64,
        "g" * 64,
    ],
)
def test_hash_fields_require_exact_lowercase_sha256_hex(invalid_hash):
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        make_state(
            revision=1,
            parent_revision=0,
            parent_hash=invalid_hash,
        )


def test_fork_creates_one_linked_revision_and_preserves_base():
    state = make_state(environment={"counter": 1})
    child = state.fork(
        environment={"counter": 2, "nested": {"value": "new"}},
        messages=({"role": "assistant", "content": "working"},),
        usage={"llm_calls": 1, "input_tokens": 12, "agent_steps": 1},
        runtime={"status": "running"},
    )

    assert child.revision == 1
    assert child.parent_revision == 0
    assert child.parent_hash == state.state_hash
    assert child.id == state.id
    assert child.metadata is state.metadata
    assert child.environment == {"counter": 2, "nested": {"value": "new"}}
    assert child.usage.llm_calls == 1
    assert len(child.messages) == 1
    assert state.environment == {"counter": 1}
    assert state.messages == ()


def test_same_parent_can_create_distinct_sibling_forks():
    state = make_state()

    first = state.fork(environment={"winner": "first"})
    second = state.fork(environment={"winner": "second"})

    assert first.parent_hash == state.state_hash
    assert second.parent_hash == state.state_hash
    assert first.revision == second.revision == 1
    assert first.state_hash != second.state_hash


def test_messages_are_append_only_when_forking():
    state = make_state(messages=({"role": "user", "content": "original"},))

    with pytest.raises(ValueError, match="append-only"):
        state.fork(messages=({"role": "user", "content": "rewritten"},))


def test_usage_is_monotonic_when_forking():
    state = make_state(usage={"input_tokens": 12, "llm_calls": 1})

    with pytest.raises(ValueError, match="cannot decrease"):
        state.fork(usage={"input_tokens": 0, "llm_calls": 0})


def test_revision_chain_metadata_is_validated():
    with pytest.raises(ValidationError, match="revision 0 cannot have a parent"):
        make_state(parent_revision=0, parent_hash="a" * 64)

    with pytest.raises(ValidationError, match="immediately precede"):
        make_state(revision=2, parent_revision=0, parent_hash="a" * 64)


def test_previous_state_shapes_are_rejected_instead_of_migrated():
    with pytest.raises(ValidationError, match="schema_version"):
        State.from_dict({"schema_version": 1, "id": "old-state"})

    with pytest.raises(ValidationError):
        State.from_dict(
            {
                "schema_version": 2,
                "identity": {"task_id": "task-1", "trial_id": "trial-1"},
                "runtime_spec": {},
            }
        )


def test_non_json_runtime_data_is_rejected():
    with pytest.raises(ValidationError):
        make_state(environment={"live_client": object()})
