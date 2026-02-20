from corral.sandbox.config import NetworkPolicy, ResourceLimits, SandboxConfig


class TestResourceLimits:
    def test_defaults(self):
        limits = ResourceLimits()
        assert limits.cpu_count == 2
        assert limits.memory_mb == 2048
        assert limits.timeout_seconds == 300
        assert limits.max_output_bytes == 60 * 1024

    def test_custom_values(self):
        limits = ResourceLimits(cpu_count=4, memory_mb=8192, timeout_seconds=600)
        assert limits.cpu_count == 4
        assert limits.memory_mb == 8192
        assert limits.timeout_seconds == 600


class TestNetworkPolicy:
    def test_defaults_deny(self):
        policy = NetworkPolicy()
        assert policy.allow_network is False
        assert policy.allowed_hosts == []

    def test_allow_with_hosts(self):
        policy = NetworkPolicy(
            allow_network=True, allowed_hosts=["pypi.org", "example.com"]
        )
        assert policy.allow_network is True
        assert len(policy.allowed_hosts) == 2


class TestSandboxConfig:
    def test_defaults(self):
        config = SandboxConfig()
        assert config.backend == "docker"
        assert config.docker_image == "python:3.11-slim"
        assert config.persistent_state is True
        assert config.python_packages == []
        assert config.network.allow_network is False

    def test_subprocess_backend(self):
        config = SandboxConfig(backend="subprocess")
        assert config.backend == "subprocess"

    def test_with_packages(self):
        config = SandboxConfig(python_packages=["numpy", "scipy"])
        assert config.python_packages == ["numpy", "scipy"]

    def test_nested_models(self):
        config = SandboxConfig(
            resources=ResourceLimits(memory_mb=4096),
            network=NetworkPolicy(allow_network=True),
        )
        assert config.resources.memory_mb == 4096
        assert config.network.allow_network is True
