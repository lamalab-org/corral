from pydantic import BaseModel, Field


class ResourceLimits(BaseModel):
    """Resource constraints for the sandbox."""

    cpu_count: int = 2
    memory_mb: int = 2048  # TODO: ml task might require more
    storage_mb: int = 4096
    timeout_seconds: int = 300
    max_output_bytes: int = 60 * 1024  # needs to be defined in the code_tools.py file


class NetworkPolicy(BaseModel):
    """Network access policy for the sandbox."""

    allow_network: bool = False
    allowed_hosts: list[str] = Field(default_factory=list)


class SandboxConfig(BaseModel):
    """Configuration for a sandbox instance."""

    backend: str = "docker"

    # Docker-specific
    docker_image: str = "python:3.11-slim"  # TODO: 3.12 or 3.12
    dockerfile: str | None = None

    # Environment
    working_dir: str = "/workspace"
    python_packages: list[str] = Field(default_factory=list)
    pip_install_timeout: int = 120

    # Resource limits
    resources: ResourceLimits = Field(default_factory=ResourceLimits)

    # Network
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)

    # State persistence
    persistent_state: bool = True
    state_dir: str | None = None
