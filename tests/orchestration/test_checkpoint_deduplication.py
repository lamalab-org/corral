import os
import shutil
from pathlib import Path

import pytest

from corral.orchestration import launchers
from corral.persistence import ShardedCommitStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("interrupt_copy", [False, True])
async def test_checkpoint_export_preserves_shared_artifacts(
    monkeypatch, tmp_path, interrupt_copy
):
    store = ShardedCommitStore(tmp_path / "runs")
    shard = store.execution_dir("benchmark:task-a:1")
    artifact = shard / "artifacts" / "blob"
    artifact.write_bytes(b"saved checkpoint")
    other_run = tmp_path / "other-run-blob"
    os.link(artifact, other_run)
    pending = shard / "checkpoint-pending.json"
    pending.write_text('{"container_id": "container"}')

    async def fake_command(*arguments, **kwargs):
        if arguments[1] == "cp":
            staged = Path(arguments[-1]) / "artifacts"
            staged.mkdir()
            (staged / "blob").write_bytes(b"exported checkpoint")
        return 0, ""

    monkeypatch.setattr(launchers, "_command", fake_command)
    if interrupt_copy:
        def interrupted_copy(source, destination, **kwargs):
            Path(destination).write_bytes(b"partial")
            raise OSError("interrupted export")

        monkeypatch.setattr(shutil, "copy2", interrupted_copy)

    launcher = launchers.DockerTaskLauncher(store)
    if interrupt_copy:
        with pytest.raises(launchers.DockerInfrastructureError):
            await launcher._export_checkpoint(shard, "container")
        assert artifact.read_bytes() == b"saved checkpoint"
        assert pending.exists()
    else:
        await launcher._export_checkpoint(shard, "container")
        assert artifact.read_bytes() == b"exported checkpoint"
        assert not pending.exists()
    assert other_run.read_bytes() == b"saved checkpoint"
    assert not list(artifact.parent.glob(".*.checkpoint-*"))
    await store.aclose()
