from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Callable, Optional

from corral_tools.health_checks import (
    EnvVarHealthCheck,
    HealthCheck,
    HealthCheckMixin,
    PackageHealthCheck,
)
from corral_tools.pymatgen.functions import (
    create_pymatgen_structure_from_cif,
    create_slab_from_structure,
    get_structure_from_mp,
)
from corral_tools.registry import ToolRegistry
from dotenv import load_dotenv

from corral.base import Tool

load_dotenv("../.env")

# Tool registry instance for pymatgen tools
pymatgen_registry = ToolRegistry()


# Health check for pymatgen tools
class PymatgenToolHealthCheck(HealthCheck):
    def __init__(self, packages: Sequence[str], env_vars: Sequence[str]):
        self.packages = packages
        self.env_vars = env_vars

    def is_healthy(self) -> tuple[bool, str | None]:
        # Check for required Python packages
        missing_packages = [pkg for pkg in self.packages if not find_spec(pkg)]
        if missing_packages:
            return False, f"Missing packages: {', '.join(missing_packages)}"

        # Check for required environment variables
        missing_env_vars = [var for var in self.env_vars if not os.getenv(var)]
        if missing_env_vars:
            return (
                False,
                f"Missing environment variables: {', '.join(missing_env_vars)}",
            )

        return True, None


# Base class for reusable pymatgen tools
class PymatgenTool(Tool, HealthCheckMixin):
    def __init__(
        self, name: str, description: str, func: Callable, registry: ToolRegistry
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.func = func
        self.registry = registry
        self.registry.register(self, environment="pymatgen")
        slef.packageds
        # can take packages and envs as well
        self.add_health_check(
            PymatgenToolHealthCheck(
                packages=["pymatgen", "mp_api"], env_vars=["MP_API_KEY"]
            )
        )

    def run(self, **kwargs) -> Optional[str]:
        return self.func(**kwargs)


# Maybe register based on the config file with some defaults
# Register tools
PymatgenTool(
    name="get_structure_from_mp",
    description="Get pymatgen structure from MP API given material id and save it as a CIF file.",
    func=get_structure_from_mp,
    registry=pymatgen_registry,
    packages=["mp_api"],
)

PymatgenTool(
    name="create_pymatgen_structure_from_cif",
    description="Create pymatgen structure from CIF file and save it as a pickle file.",
    func=create_pymatgen_structure_from_cif,
    registry=pymatgen_registry,
)

PymatgenTool(
    name="create_slab_from_structure",
    description="Create a slab from a structure and save it as a CIF file.",
    func=create_slab_from_structure,
    registry=pymatgen_registry,
)
