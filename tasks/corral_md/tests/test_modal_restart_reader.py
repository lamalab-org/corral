"""The trusted worker converts restart bytes without running agent inputs."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write


@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[1] / "modal_app/verification_worker.py"
    spec = importlib.util.spec_from_file_location("restart_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(root):
    binary = b"LAMMPS binary restart bytes, isolated as data"
    digest = hashlib.sha256(binary).hexdigest()
    name = f"checkpoint-{digest}.restart"
    (root / name).write_bytes(binary)
    return {"operation": "lammps_restart", "parameters": {
        "checkpoint": name,
        "checkpoint_sha256": digest,
    }}


def _converted_data(root, count=216):
    atoms = bulk("Si", "diamond", a=5.468728, cubic=True).repeat((3, 3, 3))
    atoms = atoms[:count]
    atoms.set_velocities(np.full((len(atoms), 3), 0.01))
    path = root / "fixture.data"
    write(
        path, atoms, format="lammps-data", atom_style="full",
        units="metal", masses=True, velocities=True,
    )
    lines = path.read_text().splitlines()
    lines[0] = (
        "LAMMPS data file via write_data, version 22 Jul 2025, "
        "timestep = 500000, units = metal"
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def _dump_header(count=216, boundaries="pp pp pp"):
    return (
        f"ITEM: TIMESTEP\n500000\nITEM: NUMBER OF ATOMS\n{count}\n"
        f"ITEM: BOX BOUNDS {boundaries}\n"
    )


def test_restart_conversion_uses_fixed_lammps_command_and_returns_state(
    worker, tmp_path, monkeypatch
):
    job = _job(tmp_path)
    fixture = _converted_data(tmp_path)
    calls = []

    def convert(command, **kwargs):
        calls.append((command, kwargs))
        if "-restart2data" in command:
            Path(command[-2]).write_bytes(fixture.read_bytes())
        else:
            Path(command[command.index("custom") + 1]).write_text(_dump_header())
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(worker.subprocess, "run", convert)
    result = worker.calculate(job, {}, tmp_path)
    assert result["checkpoint_sha256"] == job["parameters"]["checkpoint_sha256"]
    assert result["state"]["timestep"] == 500000
    assert len(result["state"]["numbers"]) == 216
    assert set(result["state"]["numbers"]) == {14}
    assert len(set(result["state"]["atom_ids"])) == 216
    assert np.asarray(result["state"]["momenta"]).shape == (216, 3)
    command, options = calls[0]
    assert command[:5] == [
        "/usr/local/bin/lmp", "-log", "none", "-screen", "none"
    ]
    assert command[5] == "-restart2data"
    assert command[-1] == "nocoeff"
    assert Path(command[6]).parent == tmp_path
    assert options["timeout"] == 120
    assert options["check"] is False
    dump_command, dump_options = calls[1]
    assert dump_command[5] == "-restart2dump"
    assert dump_command[7:9] == ["all", "custom"]
    assert dump_options["timeout"] == 120
    assert result["state"]["pbc"] == [True, True, True]


def test_restart_reader_rejects_wrong_digest_and_path_before_lammps(
    worker, tmp_path, monkeypatch
):
    job = _job(tmp_path)
    monkeypatch.setattr(
        worker.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("LAMMPS must not run"),
    )
    job["parameters"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        worker.calculate(job, {}, tmp_path)
    job["parameters"]["checkpoint"] = "../other.restart"
    with pytest.raises(ValueError, match="filename"):
        worker.calculate(job, {}, tmp_path)


def test_restart_reader_rejects_invalid_converted_state(worker, tmp_path, monkeypatch):
    job = _job(tmp_path)
    fixture = _converted_data(tmp_path, count=8)

    def convert(command, **_kwargs):
        if "-restart2data" in command:
            Path(command[-2]).write_bytes(fixture.read_bytes())
        else:
            Path(command[command.index("custom") + 1]).write_text(
                _dump_header(count=8)
            )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(worker.subprocess, "run", convert)
    with pytest.raises(ValueError, match="216-atom Si"):
        worker.calculate(job, {}, tmp_path)


def test_restart_reader_rejects_nonperiodic_binary(worker, tmp_path, monkeypatch):
    job = _job(tmp_path)
    fixture = _converted_data(tmp_path)

    def convert(command, **_kwargs):
        if "-restart2data" in command:
            Path(command[-2]).write_bytes(fixture.read_bytes())
        else:
            Path(command[command.index("custom") + 1]).write_text(
                _dump_header(boundaries="ff pp pp")
            )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(worker.subprocess, "run", convert)
    with pytest.raises(ValueError, match="not periodic"):
        worker.calculate(job, {}, tmp_path)
