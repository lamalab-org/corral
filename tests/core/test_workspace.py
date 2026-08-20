import hashlib

import pytest
from pydantic import ValidationError

from corral.core import Artifact, FileRef, WorkspaceState


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
    restored = WorkspaceState.model_validate_json(workspace.model_dump_json())

    assert restored == workspace
    with pytest.raises(TypeError, match="immutable"):
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


def test_workspace_fork_preserves_identity_and_increments_revision():
    workspace = WorkspaceState(id="workspace")
    next_workspace = workspace.fork(files={"data.txt": file_ref("data.txt")})

    assert next_workspace.id == workspace.id
    assert next_workspace.revision == workspace.revision + 1
