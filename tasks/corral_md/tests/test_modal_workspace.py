from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

import pytest
from corral_md.env import (
    MolecularDynamicsEnvironment,
    _md_file_tools,
    _md_task_prompt,
)
from corral_md.modal_workspace import run_lammps_in_modal
from corral_md.score import check_log, check_numerical

from corral.core.environment import Toolset
from corral.core.task import TaskDefinition


class _EntryType(Enum):
    FILE = 1


@dataclass(frozen=True)
class _Entry:
    path: str
    size: int
    type: _EntryType = _EntryType.FILE


class _Upload:
    def __init__(self, volume: _Volume) -> None:
        self.volume = volume

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def put_directory(self, local_path: str, remote_path: str) -> None:
        root = Path(local_path)
        prefix = remote_path.rstrip("/")
        for source in root.rglob("*"):
            if source.is_file():
                relative = source.relative_to(root).as_posix()
                self.volume.files[f"{prefix}/{relative}"] = source.read_bytes()


class _Volume:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.removed: list[str] = []

    def batch_upload(self, *, force: bool = False) -> _Upload:
        assert force is True
        return _Upload(self)

    def listdir(self, path: str, *, recursive: bool = False) -> list[_Entry]:
        assert recursive is True
        prefix = path.rstrip("/") + "/"
        return [
            _Entry(path=name, size=len(data))
            for name, data in self.files.items()
            if name.startswith(prefix)
        ]

    def read_file(self, path: str):
        data = self.files[path]
        midpoint = len(data) // 2
        yield data[:midpoint]
        yield data[midpoint:]

    def remove_file(self, path: str, *, recursive: bool = False) -> None:
        assert recursive is True
        prefix = path.rstrip("/") + "/"
        self.files = {
            name: data
            for name, data in self.files.items()
            if not name.startswith(prefix)
        }
        self.removed.append(path)


class _RemoteFunction:
    def __init__(self, volume: _Volume, *, error: Exception | None = None) -> None:
        self.volume = volume
        self.error = error
        self.calls: list[tuple[str, str, str, str]] = []

    def remote(
        self,
        input_file: str,
        log_file: str,
        local_workspace: str,
        remote_workspace: str,
    ) -> None:
        self.calls.append((input_file, log_file, local_workspace, remote_workspace))
        volume_workspace = remote_workspace.removeprefix("/results")
        self.volume.files[f"{volume_workspace}/{log_file}"] = b"LAMMPS log\n"
        self.volume.files[f"{volume_workspace}/trajectory.dump"] = b"atoms\n"
        if self.error is not None:
            raise self.error


def test_modal_lammps_round_trip_replaces_local_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "run.in").write_text("read_data structure.data\n")
    (workspace / "structure.data").write_text("atoms\n")

    volume = _Volume()
    function = _RemoteFunction(volume)
    log_path, downloaded = run_lammps_in_modal(
        workspace,
        "run.in",
        volume=volume,
        remote_function=function,
        job_id="job-1",
    )

    assert log_path == workspace / "run.log"
    assert downloaded == 4
    assert (workspace / "run.in").read_text() == "read_data structure.data\n"
    assert (workspace / "run.log").read_text() == "LAMMPS log\n"
    assert (workspace / "trajectory.dump").read_text() == "atoms\n"
    assert function.calls == [
        (
            "/results/corral/jobs/job-1/run.in",
            "run.log",
            str(workspace.resolve()),
            "/results/corral/jobs/job-1",
        )
    ]
    assert volume.removed == ["/corral/jobs/job-1"]
    assert volume.files == {}


def test_failed_modal_run_still_downloads_diagnostics(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "run.in").write_text("bad command\n")
    volume = _Volume()
    function = _RemoteFunction(volume, error=RuntimeError("LAMMPS failed"))

    with pytest.raises(RuntimeError, match="diagnostics were synchronized"):
        run_lammps_in_modal(
            workspace,
            "run.in",
            volume=volume,
            remote_function=function,
            job_id="job-2",
        )

    assert (workspace / "run.log").read_text() == "LAMMPS log\n"
    assert (workspace / "trajectory.dump").read_text() == "atoms\n"
    assert volume.files == {}


def test_modal_lammps_rejects_input_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.in"
    outside.write_text("run 1\n")

    with pytest.raises(ValueError, match="inside the current workspace"):
        run_lammps_in_modal(
            workspace,
            str(outside),
            volume=_Volume(),
            remote_function=_RemoteFunction(_Volume()),
            job_id="job-3",
        )


def test_md_io_tools_are_bound_to_local_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = _md_file_tools(str(workspace))

    tools["write_file"].execute(path="input/run.in", content="run 10\n")

    assert (workspace / "input" / "run.in").read_text() == "run 10\n"
    assert tools["read_file"].execute(path="input/run.in") == "run 10\n"
    assert tools["run_lammps"].name == "run_lammps"


def test_md_prompt_requires_workspace_relative_paths(tmp_path: Path) -> None:
    task = TaskDefinition(
        name="md",
        description="md",
        tools=[],
        scoring_fn=lambda _answer: 1.0,
        submission_format={},
        prompt_fn=_md_task_prompt,
        resolve_answer=False,
    )
    environment = MolecularDynamicsEnvironment(
        "md",
        task,
        base_work_dir=str(tmp_path),
        task_execution_id="execution",
        toolset=Toolset(workspace_factory=None),
    )

    started = environment.initial_event(execution_id="execution")
    prompt = started.task["prompt"]

    assert "Always pass workspace-relative POSIX paths" in prompt
    assert str(environment.workspace_path) not in prompt


def test_md_domain_tool_paths_cannot_target_a_sibling_workspace(
    tmp_path: Path,
) -> None:
    task = TaskDefinition(
        name="md",
        description="md",
        tools=[],
        scoring_fn=lambda _answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    environment = MolecularDynamicsEnvironment(
        "md",
        task,
        base_work_dir=str(tmp_path),
        task_execution_id="execution",
        toolset=Toolset(workspace_factory=None),
    )
    workspace = Path(environment.workspace_path or "")
    sibling = tmp_path / "sibling"
    sibling.mkdir()

    parsed = environment.preprocess_arguments(
        "convert_structure_to_lammps_data",
        {"structure_path": "input.cif", "output_file": "output.data"},
    )
    assert Path(parsed["structure_path"]).parent == workspace
    assert Path(parsed["output_file"]).parent == workspace

    for path in ("../sibling/secret.cif", str(sibling / "secret.cif")):
        with pytest.raises(ValueError, match="workspace"):
            environment.preprocess_arguments(
                "convert_structure_to_lammps_data",
                {"structure_path": path, "output_file": "output.data"},
            )


def test_md_file_scorers_read_local_outputs(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("Step Temp Press\n0 295 1\n1 305 1\n")
    restart = tmp_path / "run.restart"
    restart.write_bytes(b"restart")

    score_log = check_log(variable="Temp", target=300, tolerance=0.05, window=2)
    assert (
        score_log(json.dumps({"log_file": str(log), "restart_file": str(restart)}))
        == 1.0
    )

    score_energy = check_numerical(target=-10.0, tolerance=0.01)
    assert (
        score_energy(
            json.dumps(
                {
                    "BULK ENERGY": -10.0,
                    "path to relaxed structure": str(restart),
                }
            )
        )
        == 1.0
    )
