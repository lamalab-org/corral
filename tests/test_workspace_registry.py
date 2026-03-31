import sqlite3
from pathlib import Path

import pytest

from corral.utils.workspace_registry import WorkspaceEntry, WorkspaceRegistry


@pytest.fixture()
def tmp_work_dir(tmp_path):
    """Create a temporary work directory."""
    return str(tmp_path / "workdir")


@pytest.fixture()
def registry(tmp_work_dir):
    """Create a WorkspaceRegistry in a temp dir."""
    return WorkspaceRegistry(tmp_work_dir)


class TestWorkspaceRegistryInit:
    def test_creates_db_file(self, tmp_work_dir):
        WorkspaceRegistry(tmp_work_dir)
        db_path = Path(tmp_work_dir) / WorkspaceRegistry.DB_FILENAME
        assert db_path.exists()

    def test_creates_base_dir(self, tmp_path):
        work_dir = str(tmp_path / "nonexistent" / "deep" / "dir")
        WorkspaceRegistry(work_dir)
        assert Path(work_dir).exists()

    def test_tables_exist(self, registry):
        conn = sqlite3.connect(registry.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "workspaces" in table_names
        assert "files" in table_names
        conn.close()

    def test_idempotent_init(self, tmp_work_dir):
        """Creating registry twice on same dir should not fail."""
        reg1 = WorkspaceRegistry(tmp_work_dir)
        reg1.register_workspace("task1", "0", "uuid-1", "/path/1")
        reg2 = WorkspaceRegistry(tmp_work_dir)
        # Data from reg1 should still be there
        entry = reg2.get_by_id("uuid-1")
        assert entry is not None
        assert entry.task_id == "task1"


class TestRegisterWorkspace:
    def test_register_and_retrieve(self, registry):
        entry = registry.register_workspace("task_a", "0", "uuid-abc", "/some/path")
        assert isinstance(entry, WorkspaceEntry)
        assert entry.workspace_id == "uuid-abc"
        assert entry.task_id == "task_a"
        assert entry.trial_id == "0"
        assert entry.status == "active"
        assert entry.completed_at is None

    def test_get_by_id(self, registry):
        registry.register_workspace("task_a", "0", "uuid-abc", "/some/path")
        result = registry.get_by_id("uuid-abc")
        assert result is not None
        assert result.workspace_id == "uuid-abc"

    def test_get_by_id_not_found(self, registry):
        result = registry.get_by_id("nonexistent")
        assert result is None

    def test_multiple_workspaces(self, registry):
        registry.register_workspace("task_a", "0", "uuid-1", "/path/1")
        registry.register_workspace("task_a", "1", "uuid-2", "/path/2")
        registry.register_workspace("task_b", "0", "uuid-3", "/path/3")

        all_ws = registry.get_workspaces()
        assert len(all_ws) == 3

        task_a_ws = registry.get_workspaces(task_id="task_a")
        assert len(task_a_ws) == 2

        task_b_ws = registry.get_workspaces(task_id="task_b")
        assert len(task_b_ws) == 1


class TestUpdateStatus:
    def test_update_to_completed(self, registry):
        registry.register_workspace("task_a", "0", "uuid-1", "/path/1")
        registry.update_status(
            "uuid-1", "completed", completed_at="2026-01-01T00:00:00Z"
        )

        entry = registry.get_by_id("uuid-1")
        assert entry.status == "completed"
        assert entry.completed_at == "2026-01-01T00:00:00Z"

    def test_filter_by_status(self, registry):
        registry.register_workspace("task_a", "0", "uuid-1", "/path/1")
        registry.register_workspace("task_a", "1", "uuid-2", "/path/2")
        registry.update_status("uuid-1", "completed")

        active = registry.get_workspaces(status="active")
        assert len(active) == 1
        assert active[0].workspace_id == "uuid-2"

        completed = registry.get_workspaces(status="completed")
        assert len(completed) == 1
        assert completed[0].workspace_id == "uuid-1"


class TestFileTracking:
    def test_register_and_find_file(self, registry):
        registry.register_workspace(
            "task_a", "0", "uuid-1", "/work/task_a_trial_0_uuid1"
        )
        registry.register_file(
            "uuid-1", "result.json", "/work/task_a_trial_0_uuid1/result.json"
        )

        found = registry.find_file("result.json")
        assert found == "/work/task_a_trial_0_uuid1/result.json"

    def test_find_file_scoped_to_task(self, registry):
        registry.register_workspace("task_a", "0", "uuid-1", "/path/1")
        registry.register_workspace("task_b", "0", "uuid-2", "/path/2")
        registry.register_file("uuid-1", "output.cif", "/path/1/output.cif")
        registry.register_file("uuid-2", "output.cif", "/path/2/output.cif")

        # Without task scope, returns most recent
        found = registry.find_file("output.cif")
        assert found is not None

        # Scoped to task_a
        found_a = registry.find_file("output.cif", task_id="task_a")
        assert found_a == "/path/1/output.cif"

        # Scoped to task_b
        found_b = registry.find_file("output.cif", task_id="task_b")
        assert found_b == "/path/2/output.cif"

    def test_find_file_not_found(self, registry):
        result = registry.find_file("nonexistent.txt")
        assert result is None

    def test_find_file_returns_most_recent(self, registry):
        registry.register_workspace("task_a", "0", "uuid-1", "/path/1")
        registry.register_workspace("task_a", "1", "uuid-2", "/path/2")
        registry.register_file("uuid-1", "data.json", "/path/1/data.json")
        registry.register_file("uuid-2", "data.json", "/path/2/data.json")

        # Should return the most recently registered one (uuid-2)
        found = registry.find_file("data.json")
        assert found == "/path/2/data.json"
