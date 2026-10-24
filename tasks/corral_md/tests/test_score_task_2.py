"""Score final logs, including failures that self-reported summaries can hide."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from corral_md import env
from corral_md.scoring import check_level1_workflow, check_level2_workflow
from corral_md.scoring.task_2 import Task2Scorer
from corral_md.submission import resolve_submission

from corral.core.environment import Toolset
from corral.core.state import ExecutionState, RuntimeState
from corral.evaluation import TaskScorer

SETUP = """LAMMPS (22 Jul 2025)
units real
atom_style charge
boundary p p p
read_data /workspace/resources/md-structures-melt-liq4000/liq4000.dat
Reading data file ...
  192 atoms
include /workspace/resources/md-potentials-BKS-pot/pot.mod
pair_style hybrid/overlay buck/coul/long 8.0 12.0
pair_coeff * * buck/coul/long 0.0 1.0 0.0
pair_coeff 1 3 buck/coul/long 101093.3472 0.243838 707.96963
pair_coeff 2 3 buck/coul/long 316001.3219145 0.193817 1260.9930729
pair_coeff 3 3 buck/coul/long 42541.49842 0.343645 4441.068122
kspace_style pppm 1.0e-4
timestep 1
thermo 1000
thermo_style custom step time temp press density atoms
fix cooling all npt temp 4000 300 100 iso 0 0 10000
"""
HOLD = "unfix cooling\nfix hold all npt temp 300 300 100 iso 0 0 5000\n"


def calculation(
    root,
    *,
    resolved=True,
    interval=1000,
    noise=0,
    split=False,
    drift=1e-5,
    temperature_offset=0,
):
    rng = np.random.default_rng(74)
    rows = {}
    for step in range(0, 470001, interval):
        if step <= 370000:
            target = 4000 - step / 100
            # A known breakpoint in measured temperature, independent of the
            # evaluator's estimator and the thermostat target.
            temperature = target + temperature_offset * np.sin(np.pi * step / 370000)
            density = 2.6 - 4e-5 * (temperature - 300)
            if resolved:
                density -= 7e-5 * max(temperature - 1400, 0)
            density += noise * rng.normal()
        else:
            temperature = 300 + 2 * np.sin(2 * np.pi * (step / 1000 - 370) / 10)
            density = 2.6 + drift * (step / 1000 - 370)
        rows[step] = f"{step} {step} {temperature:.12g} 0 {density:.12g} 192\n"

    def run(start, end, command):
        return (
            command
            + "\nSetting up Verlet run ...\nStep Time Temp Press Density Atoms\n"
            + "".join(row for step, row in rows.items() if start <= step <= end)
            + f"Loop time of 1 on 1 procs for {end - start} steps with 192 atoms\n"
        )

    cooling = (
        run(0, 185000, "run 185000 start 0 stop 370000")
        + run(185000, 370000, "run 370000 upto start 0 stop 370000")
        if split
        else run(0, 370000, "run 370000")
    )
    log = root / "run.log"
    log.write_text(
        SETUP
        + cooling
        + HOLD
        + run(370000, 470000, "run 100000")
        + "Total wall time: 0:00:01\n"
    )
    answer = {
        "artifacts": {"raw_logs": ["run.log"]},
        "results": {
            "hold_temperature_K": 300,
            "hold_density_g_cm3": 2.6 + 60 * drift,
            "hold_density_drift_g_cm3_ps": drift,
            "transition": "resolved" if resolved else "unresolved",
            "cooling_tg_K": 1400 if resolved else None,
        },
    }
    path = root / "answer.json"
    path.write_text(json.dumps(answer))
    return path, answer, log


def score(path, *, level=2):
    return Task2Scorer(level=level).evaluate(path)


def change_pressure(log, function):
    lines = log.read_text().splitlines()
    for i, line in enumerate(lines):
        fields = line.split()
        if len(fields) == 6 and fields[0].isdigit():
            fields[3] = str(function(int(fields[0])))
            lines[i] = " ".join(fields)
    log.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("change", ["wrong_target", "drift", "noise"])
def test_measured_hold_pressure_must_settle_near_zero(tmp_path, level, change):
    path, answer, log = calculation(tmp_path)
    answer["results"] = {} if level == 1 else {"cooling_tg_K": 1400}
    path.write_text(json.dumps(answer))

    def pressure(step):
        if change == "wrong_target":
            return 1e9
        if change == "drift":
            return (step - 430000) * 10
        return (-1) ** (step // 1000) * 1e6

    change_pressure(log, pressure)
    result = Task2Scorer(level=level).evaluate_submission(path)
    assert result.score == 0
    assert "pressure" in result.feedback


def test_pressure_is_checked_in_hold_and_reported_in_lammps_real_units(tmp_path):
    path, answer, log = calculation(tmp_path)
    # Cooling is nonstationary; its pressures must not be treated as hold data.
    change_pressure(log, lambda step: 1e6 if step < 390000 else 200)
    answer["results"] = {"hold_pressure_atm": 200}
    path.write_text(json.dumps(answer))
    report = Task2Scorer(level=1).evaluate(path)
    assert report["score"] == 1, report
    assert report["computed"]["hold_pressure_atm"] == pytest.approx(200)
    answer["results"]["hold_pressure_atm"] = 200 * 1.01325e-4
    path.write_text(json.dumps(answer))
    assert Task2Scorer(level=1)(path) == 0


def test_sampled_pressure_fluctuations_are_allowed(tmp_path):
    path, _, log = calculation(tmp_path)
    rng = np.random.default_rng(42)
    pressures = dict(zip(range(0, 470001, 1000), rng.normal(0, 3000, 471), strict=True))
    change_pressure(log, pressures.__getitem__)
    assert Task2Scorer(level=1)(path) == 1


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("level", [1, 2])
def test_complete_log_passes_without_other_documents(tmp_path, split, level):
    path, answer, _ = calculation(tmp_path, split=split)
    if level == 1:
        answer.pop("results")
    else:
        answer["results"] = {"cooling_tg_K": answer["results"]["cooling_tg_K"]}
    path.write_text(json.dumps(answer))
    report = score(path, level=level)
    assert report["score"] == 1, report
    assert report["computed"]["hold_temperature_K"] == pytest.approx(300)
    assert report["computed"]["hold_density_g_cm3"] == pytest.approx(2.6006)
    assert report["computed"]["hold_density_drift_g_cm3_ps"] == pytest.approx(1e-5)
    assert ("cooling_tg_K" in report["computed"]) is (level == 2)
    assert {p.name for p in tmp_path.iterdir()} == {"run.log", "answer.json"}
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("transition", "tg"),
    [("unresolved", None), ("resolved", 1400), ("unresolved", 1400)],
)
def test_unresolved_data_fail_even_with_a_reported_tg(tmp_path, transition, tg):
    path, answer, _ = calculation(tmp_path, resolved=False, noise=0.0005)
    answer["results"].update(transition=transition, cooling_tg_K=tg)
    path.write_text(json.dumps(answer))
    submission = resolve_submission(path.read_text(), workspace=tmp_path)
    result = Task2Scorer(level=2).evaluate_submission(submission)
    report = result.metadata["workflow_evaluation"]
    assert result.score == 0, report
    if tg is None:
        assert (
            "results.cooling_tg_K must be a finite numerical result" in result.feedback
        )
    else:
        assert "requires a resolved Tg" in result.feedback
        assert report["computed"]["transition"] == "unresolved"
        assert report["computed"]["cooling_tg_K"] is None
    assert report["pending_checks"] == []


def test_level1_preparation_does_not_require_a_resolved_tg(tmp_path):
    path, answer, _ = calculation(tmp_path, resolved=False, noise=0.0005)
    del answer["results"]["transition"]
    del answer["results"]["cooling_tg_K"]
    path.write_text(json.dumps(answer))
    report = score(path, level=1)
    assert report["score"] == 1, report


def test_missing_tg_fails_even_with_a_resolved_transition(tmp_path):
    path, answer, _ = calculation(tmp_path)
    del answer["results"]["cooling_tg_K"]
    path.write_text(json.dumps(answer))
    result = Task2Scorer(level=2).evaluate_submission(path)
    assert result.score == 0
    assert "Missing requested submission field: results.cooling_tg_K" in result.feedback


def test_rotated_log_retains_protocol_without_repeating_initial_units(tmp_path):
    path, _, log = calculation(tmp_path)
    text = (
        log.read_text().replace("units real\n", "").replace("atom_style charge\n", "")
    )
    lines = text.splitlines()
    # A PPPM mesh initialization changes pressure at the shared run boundary,
    # without changing the retained temperature, density or elapsed step.
    boundary = [i for i, line in enumerate(lines) if line.startswith("370000 ")]
    assert len(boundary) == 2
    fields = lines[boundary[1]].split()
    fields[3] = "7"
    lines[boundary[1]] = " ".join(fields)
    log.write_text("\n".join(lines) + "\n")
    assert score(path)["score"] == 1


def test_measured_temperature_and_cutoff_determine_fit(tmp_path):
    path, _, log = calculation(tmp_path, temperature_offset=120)
    lines = log.read_text().splitlines()
    for i, line in enumerate(lines):
        fields = line.split()
        if len(fields) == 6 and fields[0].isdigit() and float(fields[2]) > 2600:
            fields[4] = str(8 + int(fields[0]) / 100000)
            lines[i] = " ".join(fields)
    log.write_text("\n".join(lines))
    report = score(path)
    assert report["score"] == 1, report
    assert report["computed"]["cooling_tg_K"] == pytest.approx(1400, abs=50)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("hold_temperature_K", 350),
        ("hold_density_g_cm3", 3.0),
        ("hold_density_drift_g_cm3_ps", 0.01),
        ("pressure_se_atm", 100),
        ("hold_pressure_drift_atm_ps", 100),
        ("cooling_tg_K", 2000),
        ("transition", "unresolved"),
        ("cooling_tg_K", None),
        ("cooling_tg_K", True),
        ("cooling_tg_K", "1400"),
        ("cooling_tg_K", float("nan")),
        ("cooling_tg_K", float("inf")),
        ("hold_temperature_K", True),
        ("hold_density_g_cm3", float("nan")),
    ],
)
def test_altered_answers_fail(tmp_path, key, value):
    path, answer, _ = calculation(tmp_path)
    answer["results"][key] = value
    path.write_text(json.dumps(answer))
    assert score(path)["score"] == 0


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("run 100000\n", "run 90000\n"),
        ("timestep 1\n", "timestep 2\n"),
        ("iso 0 0 10000", "iso 0 0 5000"),
        ("iso 0 0 5000", "iso 1 1 5000"),
        ("temp 4000 300 100", "temp 2600 300 100"),
        ("temp 300 300 100", "temp 300 300 200"),
        ("101093.3472", "101093.0"),
        ("pppm 1.0e-4", "ewald 1.0e-4"),
        ("boundary p p p", "boundary p f p"),
        ("units real", "units metal"),
        ("unfix cooling\n", ""),
        ("unfix cooling\n", "unfix cooling\nvelocity all create 300 42\n"),
        ("unfix cooling\n", "unfix cooling\nreset_timestep 370000\n"),
        ("unfix cooling\n", "unfix cooling\nset atom 1 charge 0\n"),
        (
            "unfix cooling\n",
            "unfix cooling\nread_data /workspace/resources/md-structures-melt-liq4000/liq4000.dat\n",
        ),
        ("run 100000\n", "run 100000 every 1000 'velocity all zero linear'\n"),
        ("timestep 1\n", "timestep 1\necho none\n"),
        ("Step Time Temp Press Density Atoms", "Step Time Temp Press Volume Atoms"),
        ("100000 steps with 192 atoms", "99999 steps with 192 atoms"),
        ("100000 steps with 192 atoms", "100000 steps with 191 atoms"),
    ],
)
def test_log_protocol_contradictions_fail(tmp_path, old, new):
    path, _, log = calculation(tmp_path)
    text = log.read_text()
    assert old in text
    log.write_text(text.replace(old, new))
    assert score(path)["score"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "missing_cooling",
        "reheating",
        "duplicate",
        "boundary",
        "nonfinite",
        "time_units",
        "wrong_source",
    ],
)
def test_bad_log_evidence_fails(tmp_path, mutation):
    path, answer, log = calculation(tmp_path)
    text = log.read_text()
    if mutation == "truncated":
        text = text[: text.rfind("Loop time")]
    elif mutation == "missing_cooling":
        text = text[text.index(HOLD) :]
    elif mutation == "reheating":
        text += "unfix hold\nfix heating all npt temp 300 1000 100 iso 0 0 5000\nrun 70000\n"
    elif mutation == "duplicate":
        answer["artifacts"]["raw_logs"] *= 2
    elif mutation == "boundary":
        text = text.replace(
            "370000 370000 300 0 2.6 192", "370000 370000 310 0 2.6 192", 1
        )
    elif mutation == "nonfinite":
        text = text.replace(
            "370000 370000 300 0 2.6 192", "370000 370000 nan 0 2.6 192"
        )
    elif mutation == "time_units":
        text = text.replace("370000 370000 300", "370000 370 300")
    else:
        (tmp_path / "wrong.dat").write_text("wrong system")
        text = text.replace(
            "/workspace/resources/md-structures-melt-liq4000/liq4000.dat", "wrong.dat"
        )
    log.write_text(text)
    path.write_text(json.dumps(answer))
    assert score(path)["score"] == 0


@pytest.mark.parametrize(("interval", "noise"), [(10000, 0), (1000, 0.3)])
def test_inadequate_data_cannot_pass_as_unresolved(tmp_path, interval, noise):
    path, _, _ = calculation(tmp_path, resolved=False, interval=interval, noise=noise)
    report = score(path)
    assert report["score"] == 0, report


def test_continuing_hold_drift_fails_even_when_correctly_reported(tmp_path):
    path, _, _ = calculation(tmp_path, drift=0.005)
    assert score(path, level=1)["score"] == 0


def test_level1_also_checks_measured_cooling_temperature(tmp_path):
    path, _, log = calculation(tmp_path)
    lines = log.read_text().splitlines()
    for i, line in enumerate(lines):
        fields = line.split()
        if len(fields) == 6 and fields[0].isdigit() and int(fields[0]) < 370000:
            fields[2] = "300"
            lines[i] = " ".join(fields)
    log.write_text("\n".join(lines))
    assert score(path, level=1)["score"] == 0


def test_local_source_copy_is_checked_without_an_extra_manifest_link(tmp_path):
    from zipfile import ZipFile

    path, _, log = calculation(tmp_path)
    with ZipFile(env.PACKAGE_DATA_ROOT.parent / "structures.zip") as archive:
        (tmp_path / "liq4000.dat").write_bytes(
            archive.read("structures/melt/liq4000.dat")
        )
    log.write_text(
        log.read_text().replace(
            "/workspace/resources/md-structures-melt-liq4000/liq4000.dat", "liq4000.dat"
        )
    )
    assert score(path)["score"] == 1
    (tmp_path / "liq4000.dat").write_text("changed")
    assert score(path)["score"] == 0


def test_ordered_logs_and_workspace_resolution(tmp_path):
    _path, answer, log = calculation(tmp_path)
    cooling, hold = log.read_text().split(HOLD)
    log.write_text(cooling)
    (tmp_path / "hold.log").write_text(HOLD + hold)
    answer["artifacts"]["raw_logs"] = ["/workspace/run.log", "/workspace/hold.log"]
    resolved = resolve_submission(json.dumps(answer), tmp_path)
    assert score(resolved)["score"] == 1
    answer["artifacts"]["raw_logs"].reverse()
    assert score(resolve_submission(json.dumps(answer), tmp_path))["score"] == 0


def test_log_paths_are_confined(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    _path, answer, log = calculation(tmp_path)
    answer["artifacts"]["raw_logs"] = [str(log)]
    resolved = resolve_submission(json.dumps(answer), root)
    assert score(resolved)["score"] == 0


@pytest.mark.parametrize("level", [1, 2])
def test_scoring_cli_reads_saved_task2_logs(tmp_path, level):
    answer_path, _, _ = calculation(tmp_path)
    report_path = tmp_path / "score.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "corral_md.scoring",
            "2",
            str(answer_path),
            "--level",
            str(level),
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["score"] == 1
    assert report["level"] == level
    assert report["mode"] == "saved_logs"


@pytest.mark.parametrize("level", [1, 2])
def test_task_routes_to_log_scorer_without_examples_or_validator(tmp_path, level):
    factory = check_level1_workflow if level == 1 else check_level2_workflow
    for backend in ("offline", "local", "modal"):
        assert isinstance(factory(2, verification_backend=backend), Task2Scorer)
    _path, _, _ = calculation(tmp_path)
    task_id = f"level_{level}_task_2"
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_{level}/tasks_json/task_2.json"
    )[task_id]
    current = env.MolecularDynamicsEnvironment(
        task_id,
        replace(task, tools=[]),
        base_work_dir=str(tmp_path),
        workspace_path=str(tmp_path),
        toolset=Toolset(workspace_factory=None),
    )
    current._ensure_seed_directories()
    assert not (tmp_path / "submission_examples").exists()
    assert "validate_submission" not in current.tools
    state = ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="task2",
        branch_id="main",
        runtime=RuntimeState(status="submitted"),
        submission=_path.read_text(),
    )
    prompt = current.get_task_prompt(state)
    assert "blank example" not in prompt
    assert task.submission_format in prompt
    assert '"raw_logs"' in task.submission_format
    result = TaskScorer(task, workspace=tmp_path).evaluate(state)
    assert result.score == 1
    assert result.metadata["workflow_evaluation"]["mode"] == "saved_logs"
    assert Task2Scorer(level=level).evaluate_submission("{}").score == 0


@pytest.mark.parametrize("level", [1, 2])
def test_native_read_data_timing_is_not_another_command(tmp_path, level):
    path, _, log = calculation(tmp_path)
    log.write_text(
        log.read_text().replace(
            "  192 atoms\n", "  192 atoms\n  read_data CPU = 0.002 seconds\n"
        )
    )
    assert score(path, level=level)["score"] == 1
