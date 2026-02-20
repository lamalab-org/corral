"""Tests for resolve_working_dir_path idempotency and normalization."""

from pathlib import Path

from ml.tools import resolve_working_dir_path


class TestResolveWorkingDirPath:
    """Verify resolve_working_dir_path handles edge cases correctly."""

    def test_simple_relative_path(self, tmp_path):
        """A simple relative path is joined with work_dir."""
        result = resolve_working_dir_path("data/file.json", work_dir=str(tmp_path))
        expected = str((tmp_path / "data" / "file.json").resolve())
        assert result == expected

    def test_absolute_path_passthrough(self):
        """An absolute path is returned unchanged."""
        abs_path = "/some/absolute/path/file.json"
        result = resolve_working_dir_path(abs_path, work_dir="/other/dir")
        assert result == abs_path

    def test_no_double_prefix(self, tmp_path):
        """A relative path containing ../ segments is normalized, not double-prefixed."""
        work_dir = str(tmp_path)
        # Simulate what the agent might pass: a path that goes up and
        # back into the work_dir
        tricky_path = f"../{tmp_path.name}/polymorph_data"
        result = resolve_working_dir_path(tricky_path, work_dir=work_dir)
        # Should resolve to a clean path without duplication
        expected = str((tmp_path / "polymorph_data").resolve())
        assert result == expected

    def test_dot_segments_normalized(self, tmp_path):
        """Paths with ./ and ../ segments are properly normalized."""
        result = resolve_working_dir_path(
            "./subdir/../subdir/file.txt", work_dir=str(tmp_path)
        )
        expected = str((tmp_path / "subdir" / "file.txt").resolve())
        assert result == expected

    def test_uses_env_var_fallback(self, tmp_path, monkeypatch):
        """Falls back to CORRAL_WORK_DIR env var when no work_dir given."""
        monkeypatch.setenv("CORRAL_WORK_DIR", str(tmp_path))
        result = resolve_working_dir_path("output.json")
        expected = str((tmp_path / "output.json").resolve())
        assert result == expected

    def test_falls_back_to_cwd(self, monkeypatch):
        """Falls back to cwd when no work_dir and no env var."""
        monkeypatch.delenv("CORRAL_WORK_DIR", raising=False)
        result = resolve_working_dir_path("data.csv", work_dir=None)
        expected = str(Path.cwd() / "data.csv")
        assert result == expected
