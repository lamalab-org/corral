"""Exercise AI Scientist branch isolation without Unix filesystem capabilities."""

from pathlib import Path

import anyio
import pytest
from anyio.from_thread import BlockingPortal
from tests.agents import test_session as session_fixtures

from corral.agents.ai_scientist.agent import _BranchSessionRegistry
from corral.runtime import permissions

anyio_backend = session_fixtures.anyio_backend
workspace_session = session_fixtures.workspace_session


@pytest.mark.anyio
async def test_scientist_clones_and_promotes_without_unix_flags(
    workspace_session, monkeypatch
):
    for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "fchmod"):
        monkeypatch.delattr(permissions.os, name, raising=False)
    monkeypatch.setattr(permissions.os, "supports_dir_fd", set())
    monkeypatch.setattr(permissions.os, "supports_fd", set())

    parent = workspace_session
    parent_path = Path(parent.workspace)
    original = (parent_path / "notes.txt").read_bytes()
    payload = b"experiment\r\n\x00\x1a\xff"

    async with BlockingPortal() as portal:
        sessions = _BranchSessionRegistry(parent, portal)
        first = await anyio.to_thread.run_sync(sessions.create_branch)
        first_path = Path(first.workspace)
        (first_path / "notes.txt").write_bytes(payload)
        (first_path / "plot.bin").write_bytes(payload)

        clone = await anyio.to_thread.run_sync(
            sessions.clone_branch, first.execution_id
        )
        clone_path = Path(clone.workspace)
        assert first_path != clone_path
        assert first_path.parent == clone_path.parent
        assert (clone_path / "notes.txt").read_bytes() == payload
        assert (clone_path / "plot.bin").read_bytes() == payload
        assert (parent_path / "notes.txt").read_bytes() == original
        assert not (parent_path / "plot.bin").exists()

        (clone_path / "notes.txt").write_bytes(b"winning result")
        assert (first_path / "notes.txt").read_bytes() == payload
        promotion = await anyio.to_thread.run_sync(
            sessions.promote_branch_artifacts,
            clone.execution_id,
            parent.execution_id,
        )
        assert set(promotion["files"]) == {"notes.txt", "plot.bin"}
        assert (parent_path / "notes.txt").read_bytes() == b"winning result"
        assert (parent_path / "plot.bin").read_bytes() == payload

        sessions.close_branch(clone.execution_id)
        sessions.close_branch(first.execution_id)
