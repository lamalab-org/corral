"""Independent recomputation, score gating, and untrusted checkpoint isolation."""

import copy
import importlib.util
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read
from corral_md.score import WorkflowScorer, check_level2_workflow
from corral_md.workflow_scoring.common import Evidence, Rubric
from corral_md.workflow_scoring.verification import (
    ModalVerifier,
    Plan,
    apply_verification,
    build_plan,
    compare,
    file_ref,
    geometry,
)


def module(name):
    path = Path(__file__).resolve().parents[1] / "modal_app" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def fixture_module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


worker = fixture_module("test_modal_worker_protocol").worker
_submission = fixture_module("test_workflow_numerical_policy")._submission


def response(plan, fingerprint):
    return {
        "evidence_sha256": fingerprint,
        "challenge": plan.challenge,
        "jobs": [
            {
                "id": job["id"],
                "status": "complete",
                "value": copy.deepcopy(plan.expected[job["id"]]["values"]),
            }
            for job in plan.jobs
        ],
    }


@pytest.mark.parametrize("number", range(3, 11))
def test_current_evidence_builds_checks_without_scripts_or_expected_answers(
    tmp_path, number
):
    plan = build_plan(Evidence(_submission(number, tmp_path)), number, "nonce")
    assert plan.jobs
    assert not [n for n in plan.notes if n["status"] == "unverified"], plan.notes
    assert all("expected" not in job and "value" not in job for job in plan.jobs)
    assert all(
        "energy" not in frame and "forces" not in frame and "info" not in frame
        for job in plan.jobs
        for frame in job.get("frames", [])
    )
    assert all(path.suffix != ".py" for path in plan.files.values())


def test_sampling_is_evaluator_controlled(tmp_path):
    evidence = Evidence(_submission(10, tmp_path))
    first = build_plan(evidence, 10, "first")
    assert first.jobs == build_plan(evidence, 10, "first").jobs
    assert first.jobs != build_plan(evidence, 10, "second").jobs
    assert all(len(job["sample_indices"]) == 3 for job in first.jobs)


def test_palladium_recalculation_is_split_into_bounded_jobs():
    plan = Plan(Evidence({"results": {"value": 1}}), 4, "nonce")
    frames = [bulk("Pd", "fcc", a=3.89) for _ in range(101)]
    plan.frames(
        "palladium_forces",
        frames,
        count=len(frames),
        chunk_size=50,
        values=[{"energy": float(index)} for index in range(len(frames))],
    )
    jobs = [job for job in plan.jobs if job["id"].startswith("palladium_forces")]
    assert len(jobs) == 3
    assert all(1 <= len(job["frames"]) <= 50 for job in jobs)
    assert sorted(i for job in jobs for i in job["sample_indices"]) == list(
        range(len(frames))
    )


def test_verifier_bounds_parallelism_and_stops_launching_after_failure():
    evidence = Evidence({"results": {"value": 1}})
    plan = Plan(evidence, 4, "nonce")
    for index in range(4):
        plan.add(
            f"job_{index}",
            "mace",
            {"frames": [{"index": index}]},
            {"frames": [{"energy": 0.0}]},
            targets=["raw_energy_force_records"],
            tolerance={"energy": [0, 0]},
        )

    lock = threading.Lock()
    active = 0
    maximum_active = 0
    launched = []

    def transport(subplan, fingerprint):
        nonlocal active, maximum_active
        job = subplan.jobs[0]
        with lock:
            launched.append(job["id"])
            active += 1
            maximum_active = max(maximum_active, active)
        value = copy.deepcopy(subplan.expected[job["id"]]["values"])
        if job["id"] == "job_0":
            value["frames"][0]["energy"] = 1.0
        with lock:
            active -= 1
        return {
            "evidence_sha256": fingerprint,
            "challenge": subplan.challenge,
            "jobs": [{"id": job["id"], "status": "complete", "value": value}],
        }

    verifier = ModalVerifier(transport=transport, max_parallel_calculations=1)
    checks, _, aborted = verifier._evaluate_plan(plan, evidence.fingerprint())
    by_id = {check["id"]: check for check in checks}
    assert aborted is True
    assert maximum_active == 1
    assert launched == ["job_0"]
    assert by_id["job_0"]["status"] == "failed"
    assert by_id["job_1"]["status"] == "skipped"
    assert by_id["job_2"]["status"] == "skipped"
    assert by_id["job_3"]["status"] == "skipped"


def test_verifier_runs_independent_jobs_with_bounded_parallelism():
    evidence = Evidence({"results": {"value": 1}})
    plan = Plan(evidence, 4, "nonce")
    for index in range(4):
        plan.add(
            f"job_{index}",
            "mace",
            {"frames": [{"index": index}]},
            {"frames": [{"energy": 0.0}]},
            tolerance={"energy": [0, 0]},
        )

    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def transport(subplan, fingerprint):
        nonlocal active, maximum_active
        job = subplan.jobs[0]
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return {
            "evidence_sha256": fingerprint,
            "challenge": subplan.challenge,
            "jobs": [
                {
                    "id": job["id"],
                    "status": "complete",
                    "value": copy.deepcopy(subplan.expected[job["id"]]["values"]),
                }
            ],
        }

    verifier = ModalVerifier(transport=transport, max_parallel_calculations=2)
    checks, _, aborted = verifier._evaluate_plan(plan, evidence.fingerprint())
    assert aborted is False
    assert maximum_active == 2
    assert all(check["status"] == "passed" for check in checks)


def test_verifier_waits_for_declared_dependencies():
    evidence = Evidence({"results": {"value": 1}})
    plan = Plan(evidence, 4, "nonce")
    expected = {"frames": [{"energy": 0.0}]}
    inputs = {"frames": [{"index": 0}]}
    tolerance = {"energy": [0, 0]}
    plan.add("first", "mace", inputs, expected, tolerance=tolerance)
    plan.add(
        "dependent",
        "mace",
        inputs,
        expected,
        tolerance=tolerance,
        depends_on=["first"],
    )
    plan.add("independent", "mace", inputs, expected, tolerance=tolerance)
    first_finished = False

    def transport(subplan, fingerprint):
        nonlocal first_finished
        job = subplan.jobs[0]
        if job["id"] == "dependent":
            assert first_finished
        if job["id"] == "first":
            first_finished = True
        return {
            "evidence_sha256": fingerprint,
            "challenge": subplan.challenge,
            "jobs": [
                {
                    "id": job["id"],
                    "status": "complete",
                    "value": copy.deepcopy(subplan.expected[job["id"]]["values"]),
                }
            ],
        }

    verifier = ModalVerifier(transport=transport, max_parallel_calculations=3)
    checks, _, aborted = verifier._evaluate_plan(plan, evidence.fingerprint())
    assert aborted is False
    assert all(check["status"] == "passed" for check in checks)


def test_fixed_task3_and_task5_inputs_use_release_ground_truth(tmp_path):
    task3_root = tmp_path / "task3"
    task5_root = tmp_path / "task5"
    task3_root.mkdir()
    task5_root.mkdir()
    task3 = build_plan(Evidence(_submission(3, task3_root)), 3, "nonce")
    operations = {job["id"]: job["operation"] for job in task3.jobs}
    assert operations["teacher_dimer_labels"] == "reference"
    assert operations["student_before"] == "reference"
    assert operations["student_after"] == "mace"
    assert operations["trained_bulk_model"] == "mace"

    task5 = build_plan(Evidence(_submission(5, task5_root)), 5, "nonce")
    assert [job["id"] for job in task5.jobs] == ["silicon_reference_hessians"]
    assert task5.jobs[0]["operation"] == "reference"
    assert len(task5.jobs[0]["frames"]) == 25


def test_nonreference_calculator_settings_keep_live_verification(tmp_path):
    for number, setting_name in ((3, "teacher_model_settings"), (5, "model_settings")):
        root = tmp_path / f"task{number}"
        root.mkdir()
        path = _submission(number, root)
        manifest = json.loads(path.read_text())
        settings_path = Path(manifest["settings"])
        settings = json.loads(settings_path.read_text())
        settings[setting_name] = {"default_dtype": "float32"}
        settings_path.write_text(json.dumps(settings))
        plan = build_plan(Evidence(path), number, "nonce")
        if number == 3:
            teacher = next(j for j in plan.jobs if j["id"] == "teacher_dimer_labels")
            assert teacher["operation"] == "mace"
        else:
            assert plan.jobs
            assert all(job["operation"] == "mace" for job in plan.jobs)


def test_ground_truth_resolves_dimer_orientation_and_task5_atom_order():
    ground_truth = module("ground_truth")
    assets = json.loads(
        (Path(__file__).resolve().parents[1] / "modal_app/assets.json").read_text()
    )
    jobs = ground_truth.calculation_jobs()

    def dimer_rows(offset):
        return [
            {
                "energy": float(index + offset),
                "forces": [[float(index + 1), 0, 0], [-float(index + 1), 0, 0]],
            }
            for index in range(45)
        ]

    base_hessian = np.arange(36, dtype=float).reshape(6, 6)
    base_hessian = (base_hessian + base_hessian.T) / 2
    hessians = [(base_hessian + 100 * index).tolist() for index in range(25)]
    response_value = {
        "jobs": [
            {
                "id": "task3_teacher",
                "status": "complete",
                "value": {"frames": dimer_rows(0)},
            },
            {
                "id": "task3_student",
                "status": "complete",
                "value": {"frames": dimer_rows(100)},
            },
            {
                "id": "task5_teacher",
                "status": "complete",
                "value": {"frames": [{"hessian": hessian} for hessian in hessians]},
            },
        ]
    }
    document = ground_truth.build_document(response_value, "release-1", assets)

    dimer = copy.deepcopy(jobs[0])
    dimer.update(operation="reference", reference={"name": "task3_teacher_dimers"})
    separation = ground_truth.TASK3_SEPARATIONS[0]
    dimer["frames"][0]["positions"] = [[0, 0, 0], [0, separation, 0]]
    result = ground_truth.resolve_reference(dimer, document, "release-1", assets)
    assert result["frames"][0]["energy"] == 0
    assert np.asarray(result["frames"][0]["forces"]) == pytest.approx(
        np.asarray([[0, 1, 0], [0, -1, 0]])
    )

    strained = copy.deepcopy(jobs[2])
    strained.update(
        operation="reference",
        reference={
            "name": "task5_strained_hessians",
            "strain_pairs": ground_truth.TASK5_STRAINS.tolist(),
        },
        postprocess=[
            {"symmetrization": "none", "acoustic_sum_rule": "none"} for _ in range(25)
        ],
    )
    strained["frames"][0]["positions"].reverse()
    result = ground_truth.resolve_reference(strained, document, "release-1", assets)
    permutation = np.asarray([3, 4, 5, 0, 1, 2])
    expected = np.asarray(hessians[0])[np.ix_(permutation, permutation)]
    assert np.asarray(result["frames"][0]["hessian"]) == pytest.approx(expected)


def test_runtime_uses_ground_truth_without_starting_a_sandbox(worker, monkeypatch):
    w, _ = worker
    runtime = module("verification_runtime").VerificationRuntime(w)
    folder = w.RUNS.parent / "verifications/reference-check/input"
    folder.mkdir(parents=True)
    job = {
        "id": "fixed",
        "operation": "reference",
        "parameters": {},
        "reference": {"name": "fixed-test-data"},
        "fallback_operation": "mace",
    }
    (folder / "request.json").write_text(
        json.dumps(
            {
                "jobs": [job],
                "challenge": "nonce",
                "evidence_sha256": "hash",
            }
        )
    )
    monkeypatch.setattr(
        runtime, "_resolve_reference", lambda *_: {"frames": [{"energy": -1.0}]}
    )

    def no_sandbox(*_args, **_kwargs):
        raise AssertionError("ground truth must not start a calculation sandbox")

    monkeypatch.setattr(runtime, "sandbox", no_sandbox)
    result = runtime.calculations("reference-check", "release-1", w._manifest(folder))
    assert result["jobs"] == [
        {
            "id": "fixed",
            "status": "complete",
            "value": {"frames": [{"energy": -1.0}]},
            "source": "ground_truth",
            "detail": "Compared with release-owned ground truth",
        }
    ]
    assert result["cache"] == {
        "hits": 0,
        "misses": 0,
        "ground_truth": ["fixed"],
    }


def test_runtime_falls_back_to_live_calculation_without_ground_truth(
    worker, monkeypatch
):
    w, _ = worker
    backend = module("verification_runtime")
    runtime = backend.VerificationRuntime(w)
    folder = w.RUNS.parent / "verifications/fallback-check/input"
    folder.mkdir(parents=True)
    job = {
        "id": "fixed",
        "operation": "reference",
        "parameters": {},
        "reference": {"name": "fixed-test-data"},
        "fallback_operation": "mace",
    }
    (folder / "request.json").write_text(
        json.dumps(
            {
                "jobs": [job],
                "challenge": "nonce",
                "evidence_sha256": "hash",
            }
        )
    )

    def unavailable(*_args):
        raise backend.ReferenceUnavailable("not published")

    def sandbox(payload, *_args, **_kwargs):
        live = payload["jobs"][0]
        assert live["operation"] == "mace"
        assert "reference" not in live
        assert "fallback_operation" not in live
        output = folder.parent / "fallback-output"
        output.mkdir()
        (output / "result.json").write_text(
            json.dumps(
                {
                    "jobs": [
                        {
                            "id": "fixed",
                            "status": "complete",
                            "value": {"frames": [{"energy": -1.0}]},
                        }
                    ]
                }
            )
        )
        yield output, {"sandbox_id": "sb-fallback"}

    monkeypatch.setattr(runtime, "_resolve_reference", unavailable)
    monkeypatch.setattr(runtime, "sandbox", sandbox)
    result = runtime.calculations("fallback-check", "release-1", w._manifest(folder))
    assert result["jobs"][0]["status"] == "complete"
    assert result["cache"] == {"hits": 0, "misses": 1, "ground_truth": []}


def test_public_asset_links_are_not_exported_from_sandbox(worker, tmp_path):
    w, _ = worker
    root = tmp_path / "collected"
    root.mkdir()

    class Volume:
        def iterdir(self, *_args, **_kwargs):
            return [
                SimpleNamespace(path="models", type=w.FileEntryType.SYMLINK),
                SimpleNamespace(path="result.json", type=w.FileEntryType.FILE),
            ]

        def read_file(self, name):
            assert name == "/result.json"
            yield b'{"energy": -1}'

    w._collect_workspace(Volume(), root)
    assert list(root.iterdir()) == [root / "result.json"]
    assert json.loads((root / "result.json").read_text()) == {"energy": -1}


def test_palladium_analytical_derivative_is_recalculated(tmp_path):
    path = _submission(4, tmp_path)
    fixture_module(
        "test_workflow_task_4"
    ).test_task4_analytical_hessian_is_read_as_data(path)
    plan = build_plan(Evidence(path), 4, "analytical")
    job = next(j for j in plan.jobs if j["id"] == "palladium_hessian")
    assert job["operation"] == "mace"
    assert job["properties"] == ["hessian"]
    assert np.asarray(
        plan.expected[job["id"]]["values"]["frames"][0]["hessian"]
    ).shape == (375, 375)
    assert "hessian" not in job["frames"][0]


def test_mutation_during_artifact_grading_is_rejected(tmp_path, monkeypatch):
    path = _submission(4, tmp_path)
    grader = WorkflowScorer(4, verifier=ModalVerifier(transport=response))
    original = grader.module.evaluate

    def changed(evidence, rubric):
        original(evidence, rubric)
        artifact = evidence.artifact("structures")
        artifact.write_text(artifact.read_text() + " ")

    monkeypatch.setattr(grader.module, "evaluate", changed)
    with pytest.raises(RuntimeError, match="changed during grading"):
        grader.evaluate(path)


def test_disagreement_reduces_only_affected_existing_points(tmp_path):
    path = _submission(4, tmp_path)
    original = check_level2_workflow(4)(path)

    def wrong(plan, fingerprint):
        value = response(plan, fingerprint)
        value["jobs"][0]["value"]["frames"][0]["forces"][0][0] += 5
        return value

    report = WorkflowScorer(4, verifier=ModalVerifier(transport=wrong)).evaluate(path)
    assert report["mode"] == "artifact_and_modal"
    assert report["score"] < original
    assert sum(c["points"] for c in report["checks"]) == 100
    assert (
        next(c for c in report["checks"] if c["name"] == "fcc_supercell_and_mapping")[
            "status"
        ]
        == "passed"
    )
    assert report["verification"]["checks"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "defect", ["outage", "wrong_digest", "wrong_nonce", "missing_job", "duplicate_job"]
)
def test_outage_or_wrong_response_never_awards_verification(tmp_path, defect):
    path = _submission(4, tmp_path)

    def broken(plan, fingerprint):
        value = response(plan, fingerprint)
        if defect == "outage":
            raise RuntimeError("Modal unavailable")
        if defect == "wrong_digest":
            value["evidence_sha256"] = "wrong"
        elif defect == "wrong_nonce":
            value["challenge"] = "wrong"
        elif defect == "missing_job":
            value["jobs"] = []
        elif defect == "duplicate_job":
            value["jobs"] *= 2
        return value

    report = WorkflowScorer(4, verifier=ModalVerifier(transport=broken)).evaluate(path)
    assert report["score"] is None
    assert report["status"] == "pending_review"
    assert report["possible_points"] == 100


def test_mutation_during_verification_is_rejected(tmp_path):
    path = _submission(4, tmp_path)

    def mutate(plan, fingerprint):
        artifact = plan.evidence.artifact("structures")
        artifact.write_text(artifact.read_text() + " ")
        return response(plan, fingerprint)

    with pytest.raises(RuntimeError, match="changed during"):
        ModalVerifier(transport=mutate).evaluate(Evidence(path), 4)


def test_invalid_calculation_outputs_cannot_pass():
    expected = {"frames": [{"energy": -4.0, "forces": [[0.0, 0.0, 0.0]]}]}
    tolerance = {"energy": [0, 0.01], "forces": [0, 0.02]}
    assert compare(expected, expected, tolerance)
    for actual in (
        {},
        {"frames": []},
        {"frames": [{"energy": float("nan"), "forces": [[0.0, 0.0, 0.0]]}]},
        {"frames": [{"energy": -4.0, "forces": [0.0, 0.0, 0.0]}]},
    ):
        assert not compare(expected, actual, tolerance)


def test_worker_recalculates_instead_of_reading_stored_values(monkeypatch):
    w = module("verification_worker")
    monkeypatch.setattr(w, "mace_calculator", lambda *args, **kwargs: EMT())
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.calc = EMT()
    frame = geometry(atoms)
    frame["energy"] = -123456
    result = w.calculate(
        {
            "operation": "mace",
            "frames": [frame],
            "properties": ["energy", "forces", "stress"],
        },
        {},
        Path("/unused"),
    )
    assert result["frames"][0]["energy"] == pytest.approx(atoms.get_potential_energy())
    assert np.asarray(result["frames"][0]["forces"]) == pytest.approx(
        atoms.get_forces()
    )


@pytest.mark.parametrize("count", [1, 2])
def test_soap_preserves_one_row_per_input_frame(monkeypatch, count):
    import sys
    from types import ModuleType

    descriptor = ModuleType("dscribe.descriptors")
    values = np.arange(count * 147, dtype=float).reshape(count, 147)
    descriptor.SOAP = lambda **kwargs: SimpleNamespace(
        create=lambda frames, n_jobs: values[0] if len(frames) == 1 else values
    )
    monkeypatch.setitem(sys.modules, "dscribe", ModuleType("dscribe"))
    monkeypatch.setitem(sys.modules, "dscribe.descriptors", descriptor)
    result = module("verification_worker").calculate(
        {
            "operation": "soap",
            "parameters": {"average": "inner"},
            "frames": [geometry(bulk("Si", "diamond", a=5.43))] * count,
        },
        {},
        Path("/unused"),
    )
    assert np.asarray(result["features"]).shape == (count, 147)
    assert np.array_equal(result["features"], values)


def test_runtime_isolates_checkpoints_and_replays_only_identical_input(
    worker, monkeypatch
):
    w, _ = worker
    runtime = module("verification_runtime").VerificationRuntime(w)
    folder = w.RUNS.parent / "verifications/check-1/input"
    folder.mkdir(parents=True)
    checkpoint = folder / "checkpoint.pkl"
    checkpoint.write_bytes(b"opaque checkpoint")
    params = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_ref(checkpoint)["sha256"],
    }
    jobs = [
        {
            "id": "teacher",
            "operation": "mace",
            "parameters": {},
            "sample_indices": [0],
        },
        {"id": "sw", "operation": "sw", "parameters": {}, "sample_indices": [0]},
        {
            "id": "pipeline",
            "operation": "pipeline",
            "parameters": params,
            "sample_indices": [0],
        },
        {
            "id": "student",
            "operation": "mace",
            "parameters": {"model": "submitted", **params},
            "sample_indices": [0],
        },
    ]
    (folder / "request.json").write_text(
        json.dumps({"jobs": jobs, "challenge": "nonce", "evidence_sha256": "hash"})
    )
    calls = []

    def sandbox(payload, files=None, **options):
        calls.append((payload, files, options))
        output = folder.parent / f"output-{len(calls)}"
        output.mkdir()
        (output / "result.json").write_text(
            json.dumps(
                {
                    "jobs": [
                        {"id": j["id"], "status": "complete", "value": {}}
                        for j in payload["jobs"]
                    ]
                }
            )
        )
        yield output, {"sandbox_id": f"sb-{len(calls)}"}

    monkeypatch.setattr(runtime, "sandbox", sandbox)
    refs = w._manifest(folder)
    result = runtime.calculations("check-1", "release-1", refs)
    assert len(calls) == 4
    assert [j["id"] for j in calls[0][0]["jobs"]] == ["teacher"]
    assert not calls[0][1]
    assert [j["id"] for j in calls[1][0]["jobs"]] == ["sw"]
    assert not calls[1][1]
    assert calls[1][2]["gpu"] is False
    assert calls[2][2]["gpu"] is False
    assert calls[3][2]["gpu"] is True
    assert runtime.calculations("check-1", "release-1", refs) == result
    assert len(calls) == 4
    second = w.RUNS.parent / "verifications/check-2/input"
    shutil.copytree(folder, second)
    second_request = json.loads((second / "request.json").read_text())
    for job in second_request["jobs"]:
        job["sample_indices"] = [99]
    (second / "request.json").write_text(json.dumps(second_request))
    cached = runtime.calculations("check-2", "release-1", w._manifest(second))
    assert len(calls) == 4
    assert cached["cache"] == {"hits": 4, "misses": 0, "ground_truth": []}
    with pytest.raises(ValueError, match="reused"):
        runtime.calculations("check-1", "release-1", {})


@pytest.fixture
def short_config():
    return {
        "random_seed": 17,
        "stages": [
            {
                "id": f"stage_{i}",
                "target_temperature_K": t,
                "equilibration_steps": 2,
                "production_steps": 2,
                "sample_interval": 1,
                "equilibration_interval": 1,
            }
            for i, t in enumerate((300, 400, 500, 600, 700, 800, 900, 300))
        ],
    }


def test_controlled_runner_records_actual_steps_and_continuity(tmp_path, short_config):
    runner = module("trusted_md")
    observed = runner.run(short_config, tmp_path, calculator=EMT())
    assert observed["total_steps"] == 32
    frames = read(tmp_path / "production.traj", ":")
    assert len(frames) == 24
    boundary = read(tmp_path / "boundaries.traj", ":")
    previous = observed["initial_state_sha256"]
    for i, stage in enumerate(observed["stages"]):
        assert stage["start_state_sha256"] == previous
        assert stage["start_state_sha256"] == runner.state_digest(boundary[2 * i])
        assert stage["end_state_sha256"] == runner.state_digest(boundary[2 * i + 1])
        assert stage["production_start_step"] == 4 * i + 2
        assert stage["end_step"] == 4 * i + 4
        assert (
            stage["equilibration_end_state_sha256"]
            == stage["production_start_state_sha256"]
        )
        previous = stage["end_state_sha256"]
    assert len((tmp_path / "equilibration_trace.csv").read_text().splitlines()) == 17


@pytest.mark.parametrize(
    "mutation", ["code", "wrong_cycle", "repeat_id", "zero_step", "invalid_sampling"]
)
def test_invalid_protocols_are_rejected(short_config, mutation):
    runner = module("trusted_md")
    if mutation == "code":
        short_config["script"] = "print('forged')"
    elif mutation == "wrong_cycle":
        short_config["stages"][-1]["target_temperature_K"] = 900
    elif mutation == "repeat_id":
        short_config["stages"][-1]["id"] = "stage_0"
    elif mutation == "zero_step":
        short_config["timestep_fs"] = 0
    else:
        short_config["stages"][0]["sample_interval"] = 3
    with pytest.raises(ValueError):
        runner.validate_config(short_config)


def test_backend_rejects_agent_attestations_and_altered_hashes(
    worker, monkeypatch, short_config
):
    w, _ = worker
    runtime = module("verification_runtime").VerificationRuntime(w)
    w._prepare("run-1", "release-1")
    working = w.RUNS / "run-1/workspace"
    (working / "config.json").write_text(json.dumps(short_config))

    def dynamics(config, destination, *, run_id, action_id):
        output = destination / "output/verified_md/action-1"
        observed = module("trusted_md").run(config, output, calculator=EMT())
        return {
            **observed,
            "run_id": run_id,
            "action_id": action_id,
            "release_id": "release-1",
            "model_sha256": "a" * 64,
            "prefix": "output/verified_md/action-1",
            "files": {
                "output/verified_md/action-1/" + name: ref
                for name, ref in w._manifest(output).items()
            },
        }

    monkeypatch.setattr(
        w, "_verification_runtime", lambda: SimpleNamespace(dynamics=dynamics)
    )
    result = w._execute(
        "verified_md",
        "run-1",
        "action-1",
        "config.json",
        "/local",
        [],
        "release-1",
        "base",
        w._manifest(working),
    )
    refs = result["trusted_md"]["files"]
    assert (
        runtime.provenance("run-1", "action-1", "release-1", refs)["status"] == "passed"
    )
    changed = copy.deepcopy(refs)
    changed[next(iter(changed))]["sha256"] = "0" * 64
    assert (
        runtime.provenance("run-1", "action-1", "release-1", changed)["status"]
        == "failed"
    )
    assert (
        runtime.provenance("run-1", "missing", "release-1", refs)["status"]
        == "unverified"
    )
    state = w._prepare("run-1", "release-1")
    assert state["files"] == result["files"]
    result["kind"] = "python"
    w._write(w.RUNS / "run-1/actions/action-1.json", result)
    assert (
        runtime.provenance("run-1", "action-1", "release-1", refs)["status"]
        == "unverified"
    )


def test_modal_factory_is_lazy_and_offline_is_explicit(monkeypatch):
    monkeypatch.setenv("CORRAL_MD_VERIFICATION", "modal")
    assert check_level2_workflow(10).verifier.require_provenance
    assert check_level2_workflow(10, verification_backend="offline").verifier is None


def test_verification_cannot_upgrade_failed_artifacts():
    rubric = Rubric(4)
    rubric.check("raw_energy_force_records", 10, False)
    apply_verification(
        rubric,
        {
            "checks": [
                {
                    "id": "model",
                    "status": "passed",
                    "targets": ["raw_energy_force_records"],
                }
            ]
        },
    )
    assert rubric.checks[0]["status"] == "failed"


@pytest.mark.parametrize(
    "jobs,loopback",
    [
        ([{"operation": "mace"}], False),
        ([{"operation": "sw"}], True),
        (
            [
                {
                    "operation": "lammps_restart",
                    "parameters": {
                        "checkpoint": "state.restart",
                        "checkpoint_sha256": "0" * 64,
                    },
                }
            ],
            True,
        ),
        (
            [
                {
                    "operation": "lammps_restart",
                    "parameters": {
                        "checkpoint": "state.restart",
                        "checkpoint_sha256": "0" * 64,
                    },
                },
                {"operation": "mace"},
            ],
            False,
        ),
        ([{"operation": "pipeline"}], False),
        ([{"operation": "sw", "parameters": {"checkpoint": "untrusted.pkl"}}], False),
        ([{"operation": "sw"}, {"operation": "mace"}], False),
    ],
)
def test_verification_sandbox_has_no_agent_workspace_or_controller_mounts(
    worker, monkeypatch, tmp_path, jobs, loopback
):
    from io import StringIO

    w, _ = worker
    runtime_module = module("verification_runtime")
    runtime = runtime_module.VerificationRuntime(w)
    volumes = []
    options_seen = []

    class Volume:
        def __init__(self):
            self.files = {}
            volumes.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read_only(self):
            return ("readonly", self)

        def batch_upload(self):
            return self

        def put_file(self, source, path):
            self.files[path] = (
                source.read() if hasattr(source, "read") else Path(source).read_bytes()
            )

        def iterdir(self, *_, **kwargs):
            return [
                SimpleNamespace(path=key, type=w.FileEntryType.FILE)
                for key in self.files
            ]

        def read_file(self, key):
            yield self.files[key]

    class Sandbox:
        object_id = "sb-isolated"
        stdout = StringIO("")
        stderr = StringIO("")
        returncode = 0

        def wait(self):
            volumes[1].files["/result.json"] = b'{"jobs": []}'

        def terminate(self, *, wait):
            assert wait

    def create(*command, **options):
        options_seen.append((command, options))
        return Sandbox()

    monkeypatch.setattr(runtime_module.modal.Volume, "ephemeral", Volume)
    monkeypatch.setattr(runtime_module.modal.Sandbox, "create", create)
    for output, metadata in runtime.sandbox({"jobs": jobs}):
        assert (output / "result.json").is_file()
        assert metadata["sandbox_id"] == "sb-isolated"
    command, options = options_seen[0]
    if loopback:
        assert options["cidr_allowlist"] == ["127.0.0.0/8"]
        assert not options.get("block_network")
    else:
        assert options["block_network"] is True
        assert "cidr_allowlist" not in options
    assert options["secrets"] == []
    assert command[:3] == ("python", "-I", "-c")
    assert set(options["volumes"]) == {
        "/evidence",
        "/output",
        "/workspace/models",
        "/workspace/potentials",
    }
    assert options["volumes"]["/evidence"] == ("readonly", volumes[0])
    assert list(volumes[0].files) == ["/request.json"]


def test_provenance_evaluator_uses_controller_record_not_submitted_copy(
    worker, monkeypatch, short_config
):
    import modal
    from corral_md.workflow_scoring.verification import _check_provenance

    w, _ = worker
    output = w.RUNS.parent / "local-output"
    observed = module("trusted_md").run(short_config, output, calculator=EMT())
    record = {
        **observed,
        "run_id": "run-1",
        "action_id": "action-1",
        "release_id": "release-1",
        "model_sha256": "a" * 64,
    }
    manifest = {
        "run_id": "run-1",
        "action_id": "action-1",
        "settings": str(output / "settings.json"),
        "artifacts": {
            key: str(output / filename)
            for key, filename in {
                "initial_state": "initial.traj",
                "boundary_states": "boundaries.traj",
                "production_trajectory": "production.traj",
                "thermal_trace": "thermal_trace.csv",
            }.items()
        },
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest))
    calls = []

    def remote(*args):
        calls.append(args)
        return {"status": "passed", "record": record}

    monkeypatch.setattr(
        modal.Function,
        "from_name",
        lambda app, function: SimpleNamespace(remote=remote),
    )
    verifier = ModalVerifier(
        release_id="release-1",
        volume_name="simulations",
        run_id="run-1",
    )
    assert _check_provenance(verifier, Evidence(path))["status"] == "passed"
    assert calls[0][:3] == ("run-1", "action-1", "release-1")
    assert len(calls[0][3]) == 4
    manifest["run_id"] = "other-run"
    path.write_text(json.dumps(manifest))
    assert _check_provenance(verifier, Evidence(path))["status"] == "failed"
    manifest["run_id"] = "run-1"
    path.write_text(json.dumps(manifest))
    settings = json.loads((output / "settings.json").read_text())
    settings["stages"][0]["end_time_fs"] += 1
    (output / "settings.json").write_text(json.dumps(settings))
    assert _check_provenance(verifier, Evidence(path))["status"] == "failed"
