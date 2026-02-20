import json

import pytest

from corral.sandbox.config import SandboxConfig
from corral.sandbox.subprocess_sandbox import SubprocessSandbox


class TestSubprocessSandbox:
    def test_simple_execution(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute("result = 10 * 5")
            assert result.success
            assert result.execution_result.get("result") == 50

    def test_print_captured_in_stdout(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute("print('hello sandbox')")
            assert result.success
            assert "hello sandbox" in result.stdout

    def test_state_persistence(self):
        config = SandboxConfig(backend="subprocess", persistent_state=True)
        with SubprocessSandbox(config) as sb:
            r1 = sb.execute("x = 42")
            assert r1.success

            r2 = sb.execute("result = x * 2")
            assert r2.success
            assert r2.execution_result.get("result") == 84

    def test_state_persistence_does_not_include_modules(self):
        """Modules are not picklable and are excluded from state persistence.

        Agents must re-import in each execution or include imports inline.
        """
        config = SandboxConfig(backend="subprocess", persistent_state=True)
        with SubprocessSandbox(config) as sb:
            r1 = sb.execute("import math")
            assert r1.success

            # math module is NOT persisted (modules can't be pickled)
            r2 = sb.execute("result = math.sqrt(144)")
            assert not r2.success

    def test_state_persistence_with_inline_import(self):
        config = SandboxConfig(backend="subprocess", persistent_state=True)
        with SubprocessSandbox(config) as sb:
            r1 = sb.execute("import math\nx = math.pi")
            assert r1.success

            # x (a float) IS persisted; re-import math for new calls
            r2 = sb.execute("import math\nresult = math.floor(x)")
            assert r2.success
            assert r2.execution_result.get("result") == 3

    def test_no_persistence(self):
        config = SandboxConfig(backend="subprocess", persistent_state=False)
        with SubprocessSandbox(config) as sb:
            sb.execute("x = 42")
            r2 = sb.execute("result = x * 2")
            # Should fail because x is not defined
            assert not r2.success

    def test_timeout(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute("import time; time.sleep(100)", timeout=2)
            assert not result.success
            assert result.timed_out

    def test_error_handling(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute("raise ValueError('test error')")
            assert not result.success
            assert "ValueError" in result.stderr

    def test_syntax_error(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute("def foo(")
            assert not result.success

    def test_working_directory_isolation(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute("import os; result = os.getcwd()")
            assert result.success
            cwd = result.execution_result.get("result", "")
            assert "corral_sandbox" in cwd or "/tmp" in cwd or "var" in cwd

    def test_execute_command(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute_command("echo hello")
            assert result.success
            assert "hello" in result.stdout

    def test_execute_command_timeout(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute_command("sleep 100", timeout=2)
            assert not result.success
            assert result.timed_out

    def test_duration_tracked(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute("result = 1 + 1")
            assert result.success
            assert result.duration_seconds is not None
            assert result.duration_seconds >= 0

    def test_not_started_raises(self):
        config = SandboxConfig(backend="subprocess")
        sb = SubprocessSandbox(config)
        with pytest.raises(RuntimeError) as exc_info:
            sb.execute("x = 1")
        msg = str(exc_info.value).lower()
        assert "not started" in msg or "sandbox is not started" in msg

    def test_file_upload_download(self, tmp_path):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            # Create a local file
            local_file = tmp_path / "input.txt"
            local_file.write_text("test data")

            # Upload into sandbox
            sb.upload_file(str(local_file), "input.txt")

            # Read it from within the sandbox
            result = sb.execute("with open('input.txt') as f: result = f.read()")
            assert result.success
            assert result.execution_result.get("result") == "test data"

            # Write a file inside sandbox and download it
            sb.execute("with open('output.txt', 'w') as f: f.write('from sandbox')")
            download_path = tmp_path / "downloaded.txt"
            sb.download_file("output.txt", str(download_path))
            assert download_path.read_text() == "from sandbox"

    def test_env_variables(self):
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute(
                "import os; result = os.environ.get('MY_VAR')",
                env={"MY_VAR": "hello"},
            )
            assert result.success
            assert result.execution_result.get("result") == "hello"

    def test_to_tool_result_backward_compat(self):
        """Verify the output format matches what agents expect."""
        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            result = sb.execute("result = 42")
            tool_output = result.to_tool_result()
            parsed = json.loads(tool_output)
            assert "success" in parsed
            assert "stdout" in parsed
            assert "stderr" in parsed
            assert "return_code" in parsed
            assert "execution_result" in parsed


class TestSetWorkDir:
    """Tests for SubprocessSandbox.set_work_dir()."""

    def test_set_work_dir_changes_cwd(self, tmp_path):
        """After set_work_dir, code executes in the new directory."""
        new_dir = tmp_path / "trial_dir"
        new_dir.mkdir()

        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            sb.set_work_dir(str(new_dir))
            result = sb.execute("import os; result = os.getcwd()")
            assert result.success
            assert result.execution_result.get("result") == str(new_dir)

    def test_set_work_dir_clears_state(self, tmp_path):
        """Persistent state does not leak across set_work_dir calls."""
        dir_a = tmp_path / "trial_a"
        dir_b = tmp_path / "trial_b"

        config = SandboxConfig(backend="subprocess", persistent_state=True)
        with SubprocessSandbox(config) as sb:
            sb.set_work_dir(str(dir_a))
            r1 = sb.execute("x = 99")
            assert r1.success

            r2 = sb.execute("result = x")
            assert r2.success
            assert r2.execution_result.get("result") == 99

            # Switch to a new trial directory — state should be fresh
            sb.set_work_dir(str(dir_b))
            r3 = sb.execute("result = x")
            # x should not be defined in the new trial
            assert not r3.success

    def test_stop_preserves_external_work_dir(self, tmp_path):
        """stop() does not delete a directory provided via set_work_dir."""
        external_dir = tmp_path / "env_managed"
        external_dir.mkdir()
        marker = external_dir / "keep_me.txt"
        marker.write_text("important")

        config = SandboxConfig(backend="subprocess")
        sb = SubprocessSandbox(config)
        sb.start()
        sb.set_work_dir(str(external_dir))
        sb.stop()

        assert external_dir.exists()
        assert marker.read_text() == "important"

    def test_set_work_dir_creates_directory(self, tmp_path):
        """set_work_dir creates the directory if it doesn't exist yet."""
        new_dir = tmp_path / "nonexistent" / "trial"

        config = SandboxConfig(backend="subprocess")
        with SubprocessSandbox(config) as sb:
            sb.set_work_dir(str(new_dir))
            assert new_dir.exists()

    def test_stop_removes_own_temp_dir(self):
        """stop() still removes the sandbox's own temporary directory."""
        config = SandboxConfig(backend="subprocess")
        sb = SubprocessSandbox(config)
        sb.start()
        work_dir = sb._work_dir
        assert work_dir.exists()
        sb.stop()
        assert not work_dir.exists()
