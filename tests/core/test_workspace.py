import hashlib

import pytest
from pydantic import ValidationError

from corral.core import Artifact, FileRef, State, WorkspaceState


def file_ref(path: str, content: bytes = b"content") -> FileRef:
    digest = hashlib.sha256(content).hexdigest()
    return FileRef(
        path=path,
        sha256=digest,
        size=len(content),
        blob_ref=f"sha256:{digest}",
    )


def test_workspace_manifest_is_typed_serializable_and_deeply_immutable():
    workspace = WorkspaceState(
        id="workspace-1",
        files={"results/data.csv": file_ref("results/data.csv")},
        artifacts={
            "dataset": Artifact(
                path="results/data.csv",
                kind="table",
                metadata={"columns": ["a", "b"]},
            )
        },
    )
    state = State(id="state-1", workspace=workspace)
    restored = State.from_json(state.to_json())

    assert restored.workspace == workspace
    assert restored.state_hash == state.state_hash
    with pytest.raises(TypeError, match=r"WorkspaceState\.fork"):
        workspace.artifacts["dataset"].metadata["columns"].append("c")


@pytest.mark.parametrize(
    "path",
    ["", ".", "../secret", "nested/../secret", "/absolute", r"windows\path"],
)
def test_workspace_rejects_non_portable_or_escaping_paths(path):
    with pytest.raises(ValidationError):
        file_ref(path)


def test_file_ref_requires_blob_address_to_match_content_hash():
    reference = file_ref("data.txt")

    with pytest.raises(ValidationError, match="must match sha256"):
        FileRef(
            path=reference.path,
            sha256=reference.sha256,
            size=reference.size,
            blob_ref=f"sha256:{'a' * 64}",
        )


def test_workspace_rejects_missing_artifact_file_and_file_parent_collision():
    with pytest.raises(ValidationError, match="references missing file"):
        WorkspaceState(artifacts={"report": {"path": "report.pdf", "kind": "report"}})

    with pytest.raises(ValidationError, match="nested below file"):
        WorkspaceState(
            files={
                "results": file_ref("results"),
                "results/data.csv": file_ref("results/data.csv"),
            }
        )


def test_state_accepts_only_next_revision_of_same_workspace():
    state = State(id="state-1")
    next_workspace = state.workspace.fork(files={"data.txt": file_ref("data.txt")})

    child = state.fork(workspace=next_workspace)

    assert child.workspace.id == state.workspace.id
    assert child.workspace.revision == state.workspace.revision + 1
    with pytest.raises(ValueError, match="workspace identity"):
        state.fork(workspace=WorkspaceState())
    with pytest.raises(ValueError, match="next WorkspaceState revision"):
        state.fork(workspace=next_workspace.fork())
