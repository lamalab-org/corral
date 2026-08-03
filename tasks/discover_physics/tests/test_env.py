from pathlib import Path

from discover_physics.env import _expose_world_config, load_tasks_from_json
from discover_physics.worlds import ALL_WORLDS

TASKS_DIR = Path(__file__).resolve().parents[1] / "environments" / "level_1" / "tasks_json"


class _FakeEnv:
    def __init__(self, task):
        self.current_task = task
        self.hidden_args = {}


class TestLoadTasksFromJson:
    def test_loads_all_44_generated_tasks(self):
        tasks = load_tasks_from_json(TASKS_DIR, work_dir="/tmp/dp_test_env")
        assert len(tasks) == 11 * 2 * 2

    def test_every_world_represented(self):
        tasks = load_tasks_from_json(TASKS_DIR, work_dir="/tmp/dp_test_env")
        worlds_seen = {t.scoring_inputs["world"] for t in tasks.values()}
        assert worlds_seen == set(ALL_WORLDS)

    def test_task_has_run_experiment_tool_only(self):
        tasks = load_tasks_from_json(TASKS_DIR, work_dir="/tmp/dp_test_env")
        for task in tasks.values():
            assert task.tools == ["run_experiment"]

    def test_answer_resolution_disabled(self):
        # Submissions are raw Python source, not a file path/JSON value.
        tasks = load_tasks_from_json(TASKS_DIR, work_dir="/tmp/dp_test_env")
        for task in tasks.values():
            assert task.resolve_answer is False

    def test_world_config_not_shown_in_default_prompt(self):
        tasks = load_tasks_from_json(TASKS_DIR, work_dir="/tmp/dp_test_env")
        task = tasks["gravity_seed0_noise0.0"]
        # initial_input (which the default prompt renders verbatim) must stay
        # empty -- world_config only reaches tools via hidden_args.
        assert {k: v for k, v in task.initial_input.items() if k != "work_dir"} == {}


class TestHiddenArgsWiring:
    def test_setup_fn_exposes_world_config(self):
        tasks = load_tasks_from_json(TASKS_DIR, work_dir="/tmp/dp_test_env")
        task = tasks["yukawa_seed1_noise0.05"]
        env = _FakeEnv(task)
        _expose_world_config(env)
        assert env.hidden_args == {
            "world_config": {
                "world": "yukawa",
                "engine": "field",
                "noise_std": 0.05,
                "noise_seed": 1,
            }
        }
