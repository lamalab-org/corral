import shutil

import pytest

from corral.sandbox.config import NetworkPolicy, ResourceLimits, SandboxConfig
from corral.sandbox.docker_sandbox import DockerSandbox

pytestmark = pytest.mark.skipif(
    not shutil.which("docker"), reason="Docker not available"
)


class TestDockerSandbox:
    def test_simple_execution(self):
        config = SandboxConfig(backend="docker", docker_image="python:3.11-slim")
        with DockerSandbox(config) as sb:
            result = sb.execute("result = 2 + 2")
            assert result.success
            assert result.execution_result.get("result") == 4

    def test_state_persistence(self):
        config = SandboxConfig(
            backend="docker",
            docker_image="python:3.11-slim",
            persistent_state=True,
        )
        with DockerSandbox(config) as sb:
            r1 = sb.execute("x = 42")
            assert r1.success

            r2 = sb.execute("result = x * 2")
            assert r2.success
            assert r2.execution_result.get("result") == 84

    def test_timeout(self):
        config = SandboxConfig(backend="docker", docker_image="python:3.11-slim")
        with DockerSandbox(config) as sb:
            result = sb.execute("import time; time.sleep(100)", timeout=3)
            assert not result.success
            assert result.timed_out

    def test_error_handling(self):
        config = SandboxConfig(backend="docker", docker_image="python:3.11-slim")
        with DockerSandbox(config) as sb:
            result = sb.execute("raise ValueError('test error')")
            assert not result.success
            assert "ValueError" in result.stderr

    def test_execute_command(self):
        config = SandboxConfig(backend="docker", docker_image="python:3.11-slim")
        with DockerSandbox(config) as sb:
            result = sb.execute_command("echo hello from docker")
            assert result.success
            assert "hello from docker" in result.stdout

    def test_package_installation(self):
        config = SandboxConfig(
            backend="docker",
            docker_image="python:3.11-slim",
            python_packages=["requests"],
        )
        with DockerSandbox(config) as sb:
            result = sb.execute("import requests; result = requests.__version__")
            assert result.success
            assert result.execution_result.get("result") is not None

    def test_network_isolation(self):
        config = SandboxConfig(
            backend="docker",
            docker_image="python:3.11-slim",
            network=NetworkPolicy(allow_network=False),
        )
        with DockerSandbox(config) as sb:
            result = sb.execute(
                "import urllib.request; urllib.request.urlopen('https://httpbin.org/get')"
            )
            assert not result.success

    def test_memory_limit(self):
        config = SandboxConfig(
            backend="docker",
            docker_image="python:3.11-slim",
            resources=ResourceLimits(memory_mb=64),
        )
        with DockerSandbox(config) as sb:
            # Allocate more than 64MB
            result = sb.execute("x = bytearray(200 * 1024 * 1024)")
            assert not result.success

    def test_file_upload_download(self, tmp_path):
        config = SandboxConfig(backend="docker", docker_image="python:3.11-slim")
        with DockerSandbox(config) as sb:
            local_file = tmp_path / "input.txt"
            local_file.write_text("docker test data")

            sb.upload_file(str(local_file), "/workspace/input.txt")

            result = sb.execute(
                "with open('/workspace/input.txt') as f: result = f.read()"
            )
            assert result.success
            assert result.execution_result.get("result") == "docker test data"

            sb.execute(
                "with open('/workspace/output.txt', 'w') as f: f.write('from docker')"
            )
            download_path = tmp_path / "downloaded.txt"
            sb.download_file("/workspace/output.txt", str(download_path))
            assert download_path.read_text() == "from docker"

    def test_container_cleanup(self):
        config = SandboxConfig(backend="docker", docker_image="python:3.11-slim")
        sb = DockerSandbox(config)
        sb.start()
        container_name = sb._container_name
        sb.stop()

        # Verify container is removed
        import subprocess

        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "-q"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.stdout.strip() == ""
