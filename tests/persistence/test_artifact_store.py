import asyncio
import hashlib

import pytest

from corral.persistence import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    LocalArtifactStore,
)


def run(coro):
    return asyncio.run(coro)


def test_local_artifact_store_is_content_addressed_and_idempotent(tmp_path):
    store = LocalArtifactStore(tmp_path / "objects")
    content = b"portable bytes"
    digest = hashlib.sha256(content).hexdigest()

    first = run(store.put_bytes(content))
    second = run(store.put_bytes(content))

    assert first == second
    assert first.blob_ref == f"sha256:{digest}"
    assert first.size == len(content)
    assert (tmp_path / "objects" / digest[:2] / digest).read_bytes() == content
    assert run(store.get_bytes(first.blob_ref)) == content
    assert run(store.contains(first.blob_ref)) is True


def test_put_file_and_materialize_stream_regular_files(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789" * 1000)
    store = LocalArtifactStore(tmp_path / "objects")

    stored = run(store.put_file(source))
    destination = tmp_path / "elsewhere" / "restored.bin"
    run(store.materialize(stored.blob_ref, destination))

    assert destination.read_bytes() == source.read_bytes()
    destination.write_bytes(b"changed checkout")
    assert run(store.get_bytes(stored.blob_ref)) == source.read_bytes()


def test_artifact_store_reports_missing_and_corrupt_blobs(tmp_path):
    store = LocalArtifactStore(tmp_path / "objects")

    with pytest.raises(ArtifactNotFoundError):
        run(store.get_bytes(f"sha256:{'a' * 64}"))

    stored = run(store.put_bytes(b"original"))
    digest = stored.sha256
    (tmp_path / "objects" / digest[:2] / digest).write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="content-hash"):
        run(store.get_bytes(stored.blob_ref))
    with pytest.raises(ArtifactIntegrityError, match="content-hash"):
        run(store.contains(stored.blob_ref))
