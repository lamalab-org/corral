"""External reference responses for synthetic workflow tests."""

import pytest
from ase.build import bulk


@pytest.fixture
def offline_mp149_reference(monkeypatch):
    """Replace only the external API response; keep all comparison checks live."""
    monkeypatch.setattr(
        "corral_md.workflow_scoring.lammps_checks._mp149_live_reference",
        lambda: bulk("Si", "diamond", a=5.43, cubic=True),
    )
