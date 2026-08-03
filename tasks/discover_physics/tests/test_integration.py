"""End-to-end tests against the real hidden simulator (physchool/scienceagent).

Slower than the rest of the suite (each call runs an actual FieldSampler/NBody
integration), so kept to one representative world rather than all 11.
"""

import json

from discover_physics.env import _expose_world_config, load_tasks_from_json
from discover_physics.score import check_physics_law
from discover_physics.tools import create_tools
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parents[1] / "environments" / "level_1" / "tasks_json"


class _FakeEnv:
    def __init__(self, task):
        self.current_task = task
        self.hidden_args = {}


class TestRunExperimentAgainstRealSimulator:
    def test_gravity_experiment_returns_a_trajectory(self):
        tasks = load_tasks_from_json(TASKS_DIR, work_dir="/tmp/dp_integration_test")
        task = tasks["gravity_seed0_noise0.0"]
        env = _FakeEnv(task)
        _expose_world_config(env)

        run_experiment = create_tools()["run_experiment"]
        experiments = json.dumps(
            [
                {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [3.0, 0.0],
                    "velocity2": [0.0, 0.0],
                    "measurement_times": [0.5, 1.0, 2.0],
                }
            ]
        )
        result = json.loads(
            run_experiment.execute(
                experiments=experiments, world_config=env.hidden_args["world_config"]
            )
        )
        assert len(result) == 1
        case = result[0]
        assert case["measurement_times"] == [0.5, 1.0, 2.0]
        assert len(case["pos2"]) == 3
        # An attractive law: particle 2 should have moved measurably closer
        # to the fixed source at particle 1 (origin) over 2 time units.
        start_dist = 3.0
        end_dist = (case["pos2"][-1][0] ** 2 + case["pos2"][-1][1] ** 2) ** 0.5
        assert end_dist < start_dist


class TestScoringAgainstRealSimulator:
    def test_static_law_scores_low(self):
        static_law = (
            "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
            "    return pos2, velocity2\n"
        )
        score = check_physics_law(
            static_law, world="gravity", engine="field", noise_std=0.0, noise_seed=0
        )
        assert 0.0 <= score < 0.5

    def test_score_is_bounded(self):
        static_law = (
            "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
            "    return pos2, velocity2\n"
        )
        score = check_physics_law(
            static_law, world="gravity", engine="field", noise_std=0.0, noise_seed=0
        )
        assert 0.0 <= score <= 1.0
