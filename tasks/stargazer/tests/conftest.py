from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from stargazer.models import (
    InstrumentParams,
    Observations,
    PlanetParams,
    StargazerTask,
    simulate_keplerian_rv,
)


@pytest.fixture
def simple_task() -> StargazerTask:
    times = np.linspace(0.0, 120.0, 180)
    truth = (
        PlanetParams(
            P_days=17.25,
            m_sin_i_mjup=0.42,
            e=0.18,
            omega_rad=0.7,
            l_rad=2.1,
        ),
    )
    velocities = simulate_keplerian_rv(truth, times, 0.9) + 3.5
    return StargazerTask(
        task_id="test_single",
        source="synthetic",
        truth_difficulty=1,
        star_mass_sun=0.9,
        truth_planets=truth,
        observations=Observations(
            times_days=tuple(times.tolist()),
            rvs_ms=tuple(velocities.tolist()),
            sigmas_ms=tuple(np.ones(times.size).tolist()),
            instruments=tuple("instA" for _ in times),
        ),
        instruments=(InstrumentParams(label="instA"),),
        metadata={"rv_semantics": "rv_only"},
    )


@pytest.fixture
def exact_submission(simple_task: StargazerTask) -> dict:
    planet = asdict(simple_task.truth_planets[0])
    planet.pop("m_true_mjup")
    return {"planets": [planet], "noise_jitter_ms": 0.0}


@pytest.fixture
def exact_submission_json(exact_submission: dict) -> str:
    return json.dumps(exact_submission)
