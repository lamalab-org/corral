"""Shared numerical helpers, independent imports, and public resource contract."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from stargazer.audit import DEFAULT_DATA_ROOT, reference_submission
from stargazer.fit import validate_fit
from stargazer.models import load_task
from stargazer.protocol import public_resources
from stargazer.public_rv import PublicRV

ARRAY_FIELDS = {"planet_model_ms", "model_ms", "residuals_ms"}


def test_helper_arrays_match_all_historical_and_reference_candidates():
    records = json.loads(
        (Path(__file__).parent / "fixtures/run2_candidates.json").read_text()
    )
    for path in sorted((DEFAULT_DATA_ROOT / "synthetic").glob("*.json")):
        task = load_task(path)
        records.append(
            {
                "task_id": task.task_id,
                "candidate": reference_submission(
                    task, json.loads(path.read_text())
                ).canonical_payload(),
            }
        )
    assert len(records) == 92
    for row in records:
        task = load_task(DEFAULT_DATA_ROOT / "synthetic" / f"{row['task_id']}.json")
        context = task.public_fit_context()
        helper = PublicRV(context)
        candidate = row["candidate"]
        result = helper.diagnostics(
            candidate["planets"],
            candidate.get("noise_jitter_ms"),
            rv_offset_ms=candidate.get("rv_offset_ms"),
        )
        public = validate_fit(context, candidate)
        assert {k: v for k, v in result.items() if k not in ARRAY_FIELDS} == public
        assert result["valid"]
        signal, components = helper.predict(candidate["planets"], per_planet=True)
        np.testing.assert_allclose(
            signal, result["planet_model_ms"], rtol=1e-12, atol=1e-10
        )
        np.testing.assert_allclose(
            components.sum(axis=0), signal, rtol=1e-12, atol=1e-10
        )
        offsets = np.array(
            [
                result["gamma_per_instrument_ms"][label]
                for label in context.observations.instruments
            ]
        )
        np.testing.assert_allclose(
            signal + offsets, result["model_ms"], rtol=1e-12, atol=1e-10
        )
        np.testing.assert_array_equal(
            np.asarray(context.observations.rvs_ms) - result["model_ms"],
            result["residuals_ms"],
        )
        assert (
            np.sqrt(np.mean(result["residuals_ms"] ** 2)) == result["residuals"]["rms"]
        )
        assert np.mean(np.abs(result["residuals_ms"])) == result["residuals"]["mae"]


def test_original_epoch_and_candidate_indices(simple_task, exact_submission):
    context = simple_task.public_fit_context()
    helper = PublicRV(context)
    planets = exact_submission["planets"] * 2
    times = np.asarray(context.observations.times_days)
    full = helper.predict(planets)
    partial, components = helper.predict(
        planets, times_days=times[11:], per_planet=True
    )
    np.testing.assert_array_equal(full[11:], partial)
    assert components.shape == (2, len(times) - 11)
    np.testing.assert_allclose(components.sum(axis=0), partial)
    empty, parts = helper.predict([], times_days=[], per_planet=True)
    assert empty.shape == (0,)
    assert parts.shape == (0, 0)
    with pytest.raises(ValueError, match="one-dimensional"):
        helper.predict(planets, times_days=[[1, 2]])


def test_multiple_instruments_clipping_aliases_and_omitted_jitter(simple_task):
    obs = simple_task.observations
    context = replace(
        simple_task.public_fit_context(),
        observations=replace(
            obs,
            instruments=tuple(
                "A" if i % 2 else "B" for i in range(len(obs.times_days))
            ),
        ),
    )
    planets = [
        {
            "period_days": 13.1,
            "semi_amplitude_ms": 240,
            "eccentricity": 0.95,
            "omega_rad": 0.9,
            "phase_frac": 0.23,
        }
    ]
    for jitter in (None, 0, 0.1, 2.0):
        result = PublicRV(context).diagnostics(planets, jitter, rv_offset_ms=1.0)
        assert {
            k: v for k, v in result.items() if k not in ARRAY_FIELDS
        } == validate_fit(
            context,
            {"planets": planets, "noise_jitter_ms": jitter, "rv_offset_ms": 1.0},
        )
        assert set(result["gamma_per_instrument_ms"]) == {"A", "B"}


def test_public_module_has_no_private_imports_and_source_runs_standalone(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stargazer.public_rv; "
            "assert not {'stargazer.models','stargazer.score','stargazer.env','stargazer.bank'} & sys.modules.keys()",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    resources = public_resources()
    assert set(resources) == {"public_rv.py", "analysis-guide.md"}
    (tmp_path / "public_rv.py").write_text(resources["public_rv.py"])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from public_rv import PublicRV, PublicFitContext, Observations; "
            "h=PublicRV(PublicFitContext(Observations((0.,1.),(0.,0.),(1.,1.),('a','a')),1.)); "
            "assert h.predict([]).tolist() == [0.,0.]",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_guide_example_runs_without_task_data():
    example = (
        public_resources()["analysis-guide.md"].split("```python\n")[1].split("```")[0]
    )
    namespace = {"np": np}
    exec(example, namespace)
    assert namespace["d"]["valid"]
    assert namespace["d"]["bic"] < namespace["example"].diagnostics([])["bic"]
    assert abs(namespace["fit"].x[0] - 19.3) < 0.1
    assert json.loads(namespace["answer_json"]) == namespace["d"]["candidate"]


def test_reference_substitution_and_helper_tampering(simple_task, exact_submission):
    alternate = replace(
        simple_task, truth_planets=(), metadata={"reference": "PRIVATE"}
    )
    a, b = (PublicRV(task.public_fit_context()) for task in (simple_task, alternate))
    for planets in ([], exact_submission["planets"], [{"P_days": -1}]):
        left, right = (h.diagnostics(planets) for h in (a, b))
        for key in ARRAY_FIELDS & left.keys():
            np.testing.assert_array_equal(left.pop(key), right.pop(key))
        assert left == right
    first = a.diagnostics(exact_submission["planets"])
    first["model_ms"][:] = 999
    first["residuals_ms"][:] = 999
    first["candidate"]["planets"][0]["P_days"] = 999
    assert a.diagnostics(exact_submission["planets"])["residuals"]["rms"] == 0
    assert a.predict.__func__.__closure__ is None
    assert (
        not {"StargazerTask", "load_task", "evaluate_submission"}
        & a.predict.__func__.__globals__.keys()
    )
