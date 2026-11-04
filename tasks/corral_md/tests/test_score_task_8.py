"""Compact task-8 recordings checked against independent synthetic calculations."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from corral_md.scoring import task_8 as score_task_8
from corral_md.scoring.evaluator import (
    EvaluationError,
    WorkflowScorer,
    check_level1_workflow,
    check_level2_workflow,
)
from corral_md.scoring.verification import ModalVerifier


def features(frames, parameters):
    """Simple geometry-only oracle, separate from the exported model state."""
    return (
        np.array(
            [
                [a.positions[:, 0].mean(), np.var(a.positions), a.cell.volume / 1000]
                for a in frames
            ]
        )
        * parameters["r_cut"]
    )


def energy(atoms):
    return float(
        -300 + np.sin(atoms.positions[:, 0].mean() * 10) + np.var(atoms.positions)
    )


class Teacher(Calculator):
    implemented_properties = ("energy",)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": energy(atoms)}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    model = tmp_path / "teacher.model"
    model.write_bytes(b"synthetic teacher")
    monkeypatch.setattr(
        score_task_8,
        "PINNED_TEACHER_SHA256",
        hashlib.sha256(model.read_bytes()).hexdigest(),
    )
    calculators, descriptors = (
        ModuleType("mace.calculators"),
        ModuleType("dscribe.descriptors"),
    )
    calculators.MACECalculator = Teacher

    class SOAP:
        def __init__(self, **parameters):
            self.parameters = parameters

        def create(self, frames):
            return features(frames, self.parameters)

    descriptors.SOAP = SOAP
    for name, module in (
        ("mace", ModuleType("mace")),
        ("mace.calculators", calculators),
        ("dscribe", ModuleType("dscribe")),
        ("dscribe.descriptors", descriptors),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return model


def transport(plan, fingerprint):
    jobs = []
    for job in plan.jobs:
        frames = [Atoms(**row) for row in job["frames"]]
        value = (
            {"frames": [{"energy": energy(a)} for a in frames]}
            if job["operation"] == "mace"
            else {"features": features(frames, job["parameters"]).tolist()}
        )
        jobs.append({"id": job["id"], "status": "complete", "value": value})
    return {
        "evidence_sha256": fingerprint,
        "challenge": plan.challenge,
        "jobs": jobs,
        "release_id": "synthetic",
    }


def scorer(level=2, backend=transport):
    return WorkflowScorer(8, level=level, verifier=ModalVerifier(transport=backend))


def _metrics(y, prediction):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
    }


def _fit(x, y, config):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    kind = config.get("preprocessing", "StandardScaler")
    options = config.get("preprocessing_options", {})
    state = {"preprocessing": kind}
    if kind == "identity":
        if options:
            raise ValueError("Identity preprocessing has no options")
        transform = None
        z = x
    else:
        classes = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "PCA": PCA,
        }
        transform = classes[kind](**options).fit(x)
        z = transform.transform(x)
        if kind == "StandardScaler":
            state.update(
                mean=(
                    transform.mean_ if transform.with_mean else np.zeros(x.shape[1])
                ).tolist(),
                scale=(
                    transform.scale_ if transform.with_std else np.ones(x.shape[1])
                ).tolist(),
                with_mean=transform.with_mean,
                with_std=transform.with_std,
            )
        elif kind == "MinMaxScaler":
            state.update(
                scale=transform.scale_.tolist(),
                offset=transform.min_.tolist(),
                data_min=transform.data_min_.tolist(),
                data_max=transform.data_max_.tolist(),
                feature_range=list(transform.feature_range),
                clip=transform.clip,
            )
        else:
            state.update(
                mean=transform.mean_.tolist(),
                components=transform.components_.tolist(),
                explained_variance=transform.explained_variance_.tolist(),
                whiten=transform.whiten,
                svd_solver=transform._fit_svd_solver,
            )
    ridge = Ridge(
        alpha=config["alpha"],
        fit_intercept=config.get("fit_intercept", True),
        positive=config.get("positive", False),
    ).fit(z, y)
    state.update(
        alpha=float(ridge.alpha),
        coef=ridge.coef_.tolist(),
        intercept=float(ridge.intercept_),
        fit_intercept=ridge.fit_intercept,
        positive=ridge.positive,
    )

    def predict(features):
        return ridge.predict(
            transform.transform(features) if transform is not None else features
        )

    return state, predict


def make_submission(
    tmp_path, model, *, level=2, preprocessing="StandardScaler", alpha=0.01
):
    """Construct saved evidence independently of any agent-side helper."""
    from dscribe.descriptors import SOAP

    records = [
        {
            "event": "start",
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "default_dtype": "float64",
            "units": {"energy": "eV", "length": "Angstrom"},
        }
    ]

    def dataset(name, seed):
        rng = np.random.default_rng(seed)
        ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
        frames, rows, labels = [], [], []
        for i in range(100):
            strain = (-0.02 + 0.01 * (i // 20)) if name == "strained_test" else 0
            sigma = 0.05 if name == "strained_test" else 0.01 + 0.001 * i
            atoms = ideal.copy()
            atoms.set_cell(ideal.cell * (1 + strain), scale_atoms=True)
            atoms.positions += rng.normal(0, sigma, (64, 3))
            atoms.calc = Teacher()
            value = float(atoms.get_potential_energy())
            rows.append(
                {
                    "row_id": f"{name}:{i}",
                    "positions": atoms.positions.tolist(),
                    "cell": atoms.cell.array.tolist(),
                    "energy_eV": value,
                }
            )
            frames.append(atoms)
            labels.append(value)
        records.append(
            {
                "event": "dataset",
                "name": name,
                "seed": seed,
                "rng": "numpy.default_rng",
                "rows": rows,
            }
        )
        return frames, np.asarray(labels)

    def finish(results):
        records.append({"event": "end"})
        write_log(tmp_path, records)
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps({"artifacts": {"run_log": "run.jsonl"}, "results": results})
        )
        return path, None

    frames, labels = dataset("train", 17)
    if level == 1:
        return finish({})
    config = {
        "soap": {
            "species": ["Si"],
            "periodic": True,
            "average": "inner",
            "r_cut": 4,
            "n_max": 2,
            "l_max": 1,
        },
        "alpha": alpha,
        "preprocessing": preprocessing,
    }
    if preprocessing == "PCA":
        config["preprocessing_options"] = {"n_components": 2, "svd_solver": "full"}
    candidates = [config, {**config, "alpha": alpha * 10}]
    splits = [
        {
            "train_indices": list(range(parity, 100, 2)),
            "test_indices": list(range(1 - parity, 100, 2)),
        }
        for parity in (0, 1)
    ]
    subsets = [list(range(0, 100, 5)), list(range(0, 100, 2)), list(range(100))]
    scores = []
    for candidate in candidates:
        x = np.asarray(SOAP(**candidate["soap"]).create(frames))
        folds = []
        for split in splits:
            train, test = split["train_indices"], split["test_indices"]
            state, predict = _fit(x[train], labels[train], candidate)
            folds.append(
                {
                    **split,
                    "model": state,
                    "metrics": _metrics(labels[test], predict(x[test])),
                }
            )
        records.append(
            {
                "event": "candidate",
                "config": candidate,
                "features": x.tolist(),
                "folds": folds,
            }
        )
        scores.append(np.mean([fold["metrics"]["rmse"] for fold in folds]))
    selected = int(np.argmin(scores))
    config = candidates[selected]
    descriptor = SOAP(**config["soap"])
    x = np.asarray(descriptor.create(frames))
    final, predict = _fit(x, labels, config)
    curves, predictors = [], []
    for indices in subsets:
        state, predictor = _fit(x[indices], labels[indices], config)
        curves.append({"train_indices": indices, "model": state})
        predictors.append(predictor)
    records.append(
        {
            "event": "freeze",
            "selected_candidate": selected,
            "selection_metric": "rmse",
            "test_seeds": [18, 19],
            "final": final,
            "learning_curve": curves,
        }
    )
    results = {}
    for name, seed in (("id_test", 18), ("strained_test", 19)):
        frames, labels = dataset(name, seed)
        x = np.asarray(descriptor.create(frames))
        records.append({"event": "test_features", "name": name, "features": x.tolist()})
        prediction = predict(x)
        if name == "id_test":
            results[name] = _metrics(labels, prediction)
            results["learning_curve"] = [
                {"size": len(indices), **_metrics(labels, predictor(x))}
                for indices, predictor in zip(subsets, predictors, strict=True)
            ]
        else:
            results["strained_pooled"] = _metrics(labels, prediction)
            results["strain_groups"] = [
                {
                    "strain": strain,
                    **_metrics(
                        labels[20 * i : 20 * (i + 1)], prediction[20 * i : 20 * (i + 1)]
                    ),
                }
                for i, strain in enumerate((-0.02, -0.01, 0, 0.01, 0.02))
            ]
    return finish(results)


def read_log(root):
    return [json.loads(line) for line in (root / "run.jsonl").read_text().splitlines()]


def write_log(root, rows):
    (root / "run.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "energy",
        "seed",
        "digest",
        "missing_energy",
        "species",
        "periodicity",
        "order",
    ],
)
def test_native_dataset_frames_use_existing_science_and_verifier(
    tmp_path, runtime, defect
):
    path, _ = make_submission(tmp_path, runtime, level=1)
    log = read_log(tmp_path)
    frames = []
    for row in log[1]["rows"]:
        atoms = Atoms("Si64", positions=row["positions"], cell=row["cell"], pbc=True)
        if defect == "species":
            atoms.numbers[0] = 13
        if defect == "periodicity":
            atoms.pbc = False
        atoms.calc = SinglePointCalculator(
            atoms, energy=row["energy_eV"] + (1 if defect == "energy" else 0)
        )
        frames.append(atoms)
    if defect == "order":
        frames.reverse()
    frames[0].info.update(
        seed=18 if defect == "seed" else log[1]["seed"],
        model_sha256="wrong" if defect == "digest" else log[0]["model_sha256"],
    )
    if defect == "missing_energy":
        frames[-1].calc = None
    with Trajectory(tmp_path / "dataset.traj", "w") as writer:
        for atoms in frames:
            writer.write(atoms)
    path.write_text(json.dumps({"artifacts": {"run_log": "dataset.traj"}}))
    seen = []

    def independent(plan, fingerprint):
        seen.extend(frame for job in plan.jobs for frame in job["frames"])
        return transport(plan, fingerprint)

    report = scorer(1, independent).evaluate(path)
    assert report["score"] == (1 if defect is None else 0), report
    if defect in (None, "energy"):
        assert len(seen) == 100


@pytest.mark.parametrize("seed", [17, 2**80 + 17])
@pytest.mark.parametrize("rng_name", [None, "numpy.random.default_rng"])
@pytest.mark.usefixtures("runtime")
def test_native_dataset_accepts_default_rng_seeds_without_checkpoint_digest(
    tmp_path, seed, rng_name
):
    rng = np.random.default_rng(seed)
    ideal = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    with Trajectory(tmp_path / "dataset.traj", "w") as writer:
        for index in range(100):
            atoms = ideal.copy()
            atoms.positions += rng.normal(0, 0.01 + 0.001 * index, (64, 3))
            atoms.calc = SinglePointCalculator(atoms, energy=energy(atoms))
            if index == 0:
                atoms.info.update(seed=seed, default_dtype="float64")
                if rng_name is not None:
                    atoms.info["rng"] = rng_name
            writer.write(atoms)
    manifest = tmp_path / "answer.json"
    manifest.write_text(json.dumps({"artifacts": {"run_log": "dataset.traj"}}))
    report = scorer(level=1).evaluate(manifest)
    assert report["score"] == 1, report
    assert report["verification"]["status"] == "passed"


@pytest.mark.parametrize("level", [1, 2])
def test_minimal_contract_and_independent_plan(tmp_path, runtime, level):
    path, _ = make_submission(tmp_path, runtime, level=level)
    seen = []

    def independent(plan, fingerprint):
        seen.append(plan)
        assert all("expected" not in job and "value" not in job for job in plan.jobs)
        return transport(plan, fingerprint)

    report = scorer(level, independent).evaluate(path)
    assert report["score"] == 1, report
    assert report["possible_points"] == 100
    assert "pending_checks" not in report
    assert sum(
        len(job["frames"]) for job in seen[0].jobs if job["operation"] == "mace"
    ) == (100 if level == 1 else 300)
    assert set(json.loads(path.read_text())) == {"artifacts", "results"}
    assert set(json.loads(path.read_text())["artifacts"]) == {"run_log"}
    assert ("id_test" in report["calculated_results"]) is (level == 2)


def test_log_only_keeps_strain_and_learning_curve_analysis(tmp_path, runtime):
    path, _ = make_submission(tmp_path, runtime)
    path = Path(path)
    manifest = json.loads(path.read_text())
    manifest.pop("results")
    path.write_text(json.dumps(manifest))
    report = scorer().evaluate(path)
    assert report["score"] == 1, report
    assert len(report["calculated_results"]["strain_groups"]) == 5
    assert len(report["calculated_results"]["learning_curve"]) == 3
    manifest["results"] = {"id_test": {"rmse": 999}}
    path.write_text(json.dumps(manifest))
    assert scorer().evaluate(path)["score"] == 0


@pytest.mark.parametrize("kind", ["identity", "StandardScaler", "MinMaxScaler", "PCA"])
def test_supported_training_only_preprocessing(tmp_path, runtime, kind):
    path, _ = make_submission(tmp_path, runtime, preprocessing=kind)
    assert scorer().evaluate(path)["score"] == 1


def test_self_consistent_poor_predictions_fail_reference_quality(tmp_path, runtime):
    path, _ = make_submission(tmp_path, runtime, alpha=1e12)
    report = scorer().evaluate(path)
    assert report["score"] == 0, report
    assert report["calculated_results"]["id_test"]["r2"] < 0.98
    assert report["calculated_results"]["strained_pooled"]["r2"] < 0
    assert report["verification"]["status"] == "passed"
    assert report["checks"][-1]["name"] == "physical_reference_r2"
    assert report["checks"][-1]["status"] == "failed"


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "wrong_teacher",
        "units",
        "seed",
        "duplicate_seed",
        "relaxed",
        "cell",
        "row_id",
        "missing_structure",
        "test_before_freeze",
        "fit_after_freeze",
        "overlap",
        "out_of_bounds",
        "duplicate_indices",
        "empty_fold",
        "wrong_selection",
        "scaler_leak",
        "coefficient",
        "curve_coefficient",
        "curve_subset",
        "one_size",
        "soap_species",
        "soap_local",
        "cv_metric",
        "nan",
    ],
)
def test_invalid_calculations_fail(tmp_path, runtime, mutation):
    path, _ = make_submission(tmp_path, runtime)
    manifest = json.loads(path.read_text())
    manifest.pop("results")
    path.write_text(json.dumps(manifest))
    rows = read_log(tmp_path)
    candidate = rows[2]
    frozen = next(row for row in rows if row["event"] == "freeze")
    if mutation == "truncated":
        rows.pop()
    elif mutation == "wrong_teacher":
        rows[0]["model_sha256"] = "0" * 64
    elif mutation == "units":
        rows[0]["units"]["energy"] = "eV/atom"
    elif mutation == "seed":
        rows[1]["seed"] += 1
    elif mutation == "duplicate_seed":
        frozen["test_seeds"][0] = rows[1]["seed"]
    elif mutation in {"relaxed", "cell"}:
        rows[1]["rows"][0]["positions" if mutation == "relaxed" else "cell"][0][0] += (
            0.01
        )
    elif mutation == "row_id":
        rows[1]["rows"][0]["row_id"] = "train:1"
    elif mutation == "missing_structure":
        rows[1]["rows"].pop()
    elif mutation == "test_before_freeze":
        index = rows.index(frozen)
        rows[index], rows[index + 1] = rows[index + 1], rows[index]
    elif mutation == "fit_after_freeze":
        rows.insert(-1, copy.deepcopy(candidate))
    elif mutation in {"overlap", "out_of_bounds", "duplicate_indices", "empty_fold"}:
        fold = candidate["folds"][0]
        if mutation == "empty_fold":
            fold["train_indices"] = []
        else:
            fold["train_indices"][0] = {
                "overlap": 1,
                "out_of_bounds": 100,
                "duplicate_indices": 2,
            }[mutation]
    elif mutation == "wrong_selection":
        frozen["selected_candidate"] = 1 - frozen["selected_candidate"]
    elif mutation == "scaler_leak":
        candidate["folds"][0]["model"]["mean"] = np.mean(
            candidate["features"], axis=0
        ).tolist()
    elif mutation == "coefficient":
        frozen["final"]["coef"][0] += 10
    elif mutation == "curve_coefficient":
        frozen["learning_curve"][0]["model"]["coef"][0] += 10
    elif mutation == "curve_subset":
        frozen["learning_curve"][0]["train_indices"] = list(range(20))
    elif mutation == "one_size":
        frozen["learning_curve"] = frozen["learning_curve"][:1]
    elif mutation == "soap_species":
        candidate["config"]["soap"]["species"] = ["Cu"]
    elif mutation == "soap_local":
        candidate["config"]["soap"]["average"] = "off"
    elif mutation == "cv_metric":
        candidate["folds"][0]["metrics"]["rmse"] += 0.1
    else:
        rows[1]["rows"][0]["energy_eV"] = float("nan")
    write_log(tmp_path, rows)
    assert scorer().evaluate(path)["score"] == 0, mutation


@pytest.mark.parametrize(
    "group", ["id_test", "strained_pooled", "strain_groups", "learning_curve"]
)
@pytest.mark.parametrize("metric", ["mae", "rmse", "r2"])
def test_every_reported_metric_is_checked(tmp_path, runtime, group, metric):
    path, _ = make_submission(tmp_path, runtime)
    manifest = json.loads(path.read_text())
    claim = manifest["results"][group]
    if isinstance(claim, list):
        claim = claim[0]
    claim[metric] += 0.1
    path.write_text(json.dumps(manifest))
    report = scorer().evaluate(path)
    assert report["score"] == 0
    assert (
        next(
            check for check in report["checks"] if check["name"] == "reported_metrics"
        )["status"]
        == "failed"
    )


@pytest.mark.parametrize("forged", ["labels", "descriptors"])
def test_self_consistent_fabrications_fail_independent_recalculation(
    tmp_path, runtime, monkeypatch, forged
):
    if forged == "labels":
        original = Teacher.calculate

        def altered(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.results["energy"] += 1

        monkeypatch.setattr(Teacher, "calculate", altered)
    else:
        descriptor = sys.modules["dscribe.descriptors"].SOAP
        original = descriptor.create
        monkeypatch.setattr(
            descriptor, "create", lambda self, frames: original(self, frames) * 2
        )
    path, _ = make_submission(tmp_path, runtime)
    report = scorer().evaluate(path)
    assert report["score"] == 0, report
    independent = next(
        check
        for check in report["checks"]
        if check["name"] == "independent_teacher_and_soap"
    )
    assert independent["status"] == "failed"


def test_independent_check_is_required_and_bound_to_evidence(tmp_path, runtime):
    path, _ = make_submission(tmp_path, runtime, level=1)
    offline = WorkflowScorer(8, level=1)
    with pytest.raises(EvaluationError):
        offline.evaluate(path)
    with pytest.raises(EvaluationError):
        offline(path)

    def mismatched(plan, fingerprint):
        return {**transport(plan, fingerprint), "challenge": "wrong"}

    assert scorer(1, mismatched).evaluate(path)["score"] == 0
    assert check_level1_workflow(8).verifier is not None
    assert check_level2_workflow(8).verifier is not None


def test_teacher_pin_matches_released_asset():
    from corral_md.scoring.level1_trusted import PINNED_TEACHER_SHA256

    assets = json.loads(
        (Path(__file__).parents[1] / "modal_app/assets.json").read_text()
    )
    assert assets["models"]["teacher.model"]["sha256"] == PINNED_TEACHER_SHA256


@pytest.mark.parametrize("kind", ["identity", "StandardScaler", "MinMaxScaler", "PCA"])
def test_real_soap_pipeline_roundtrip(tmp_path, runtime, monkeypatch, kind):
    # DScribe belongs to the calculation image, so keep it optional in unit-only installs.
    monkeypatch.delitem(sys.modules, "dscribe")
    monkeypatch.delitem(sys.modules, "dscribe.descriptors")
    soap = pytest.importorskip("dscribe.descriptors").SOAP

    def independent(plan, fingerprint):
        response = transport(plan, fingerprint)
        for request, job in zip(plan.jobs, response["jobs"], strict=True):
            if request["operation"] == "soap":
                frames = [Atoms(**row) for row in request["frames"]]
                job["value"] = {
                    "features": soap(**request["parameters"]).create(frames).tolist()
                }
        return response

    path, _ = make_submission(tmp_path, runtime, preprocessing=kind)
    report = scorer(backend=independent).evaluate(path)
    # The synthetic teacher depends on absolute coordinates, which invariant
    # SOAP cannot reliably predict. Reconstruction remains valid, but the
    # independently measured poor predictive quality must now fail the task.
    assert all(check["status"] == "passed" for check in report["checks"][:-1]), report
    assert report["checks"][-1]["name"] == "physical_reference_r2"
    assert report["calculated_results"]["id_test"]["r2"] < 0.98
    assert report["score"] == 0, report


def test_dataset_rows_need_no_start_dataset_end_wrapper(tmp_path, runtime):
    path, _ = make_submission(tmp_path, runtime, level=1)
    original = read_log(tmp_path)
    header, dataset = original[:2]
    rows = [
        {
            **row,
            "row_id": i,
            "seed": dataset["seed"],
            "rng": dataset["rng"],
            "model_sha256": header["model_sha256"],
            "default_dtype": header["default_dtype"],
        }
        for i, row in enumerate(dataset["rows"])
    ]
    write_log(tmp_path, rows)
    report = scorer(level=1).evaluate(path)
    assert report["score"] == 1, report


def test_validation_metrics_can_be_derived_from_retained_fits(tmp_path, runtime):
    path, _ = make_submission(tmp_path, runtime)
    rows = read_log(tmp_path)
    for row in rows:
        if row.get("event") == "candidate":
            for fold in row["folds"]:
                fold["metrics"].pop("rmse")
    write_log(tmp_path, rows)
    assert scorer().evaluate(path)["score"] == 1


@pytest.mark.parametrize("level", [1, 2])
def test_dataset_event_labels_are_not_a_required_field_name(tmp_path, runtime, level):
    path, _ = make_submission(tmp_path, runtime, level=level)
    rows = read_log(tmp_path)
    for row in rows:
        row["type"] = row.pop("event")
    rows[0]["type"] = "metadata"
    write_log(tmp_path, rows)
    assert scorer(level=level).evaluate(path)["score"] == 1


@pytest.mark.parametrize("level", [1, 2])
def test_dataset_identity_uses_independent_checks_and_record_order(
    tmp_path, runtime, level
):
    path, _ = make_submission(tmp_path, runtime, level=level)
    rows = read_log(tmp_path)
    for key in ("model_sha256", "units", "default_dtype"):
        rows[0].pop(key)
    for row in rows:
        if row.get("event") == "dataset":
            row.pop("rng")
            for structure in row["rows"]:
                structure.pop("row_id")
    write_log(tmp_path, rows)
    assert scorer(level=level).evaluate(path)["score"] == 1
