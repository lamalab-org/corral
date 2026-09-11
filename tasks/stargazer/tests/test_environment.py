from __future__ import annotations

import json
from dataclasses import asdict

from stargazer.env import create_environments


def test_official_levels_have_fixed_reference_valid_synthetic_banks(tmp_path):
    levels = {
        level: create_environments(level=level, work_dir=tmp_path / f"level_{level}")
        for level in (1, 2, 3)
    }

    assert {level: len(environments) for level, environments in levels.items()} == {
        1: 20,
        2: 20,
        3: 20,
    }
    task_ids = [task_id for environments in levels.values() for task_id in environments]
    assert len(task_ids) == len(set(task_ids)) == 60

    expected_source_counts = {level: {"synthetic": 20} for level in (1, 2, 3)}
    expected_difficulty_counts = {
        1: {1: 10, 2: 10},
        2: {3: 5, 4: 5, 5: 5, 6: 5},
        3: {7: 4, 8: 7, 9: 4, 10: 5},
    }
    expected_evaluation_budgets = {1: {2}, 2: {4}, 3: {9}}
    for level, environments in levels.items():
        source_counts: dict[str, int] = {}
        difficulty_counts: dict[int, int] = {}
        for environment in environments.values():
            task = environment.current_task.scoring_inputs["benchmark_task"]
            source_counts[task.source] = source_counts.get(task.source, 0) + 1
            difficulty_counts[task.truth_difficulty] = (
                difficulty_counts.get(task.truth_difficulty, 0) + 1
            )
        assert source_counts == expected_source_counts[level]
        assert difficulty_counts == expected_difficulty_counts[level]
        assert {
            environment.current_task.scoring_inputs["max_evaluations"]
            for environment in environments.values()
        } == expected_evaluation_budgets[level]


def test_real_data_is_a_separate_challenge_split(tmp_path):
    environments = create_environments(level="real", work_dir=tmp_path / "real")

    assert len(environments) == 20
    assert {
        environment.current_task.scoring_inputs["benchmark_task"].source
        for environment in environments.values()
    } == {"real"}


def test_evaluation_sessions_are_isolated_between_trials(tmp_path):
    environments = create_environments(level=1, work_dir=tmp_path / "sessions")
    first, second = list(environments.values())[:2]
    first.configure_additional_apps()
    second.configure_additional_apps()
    try:
        first_session = first.hidden_args["evaluation_session"]
        second_session = second.hidden_args["evaluation_session"]
        first_session["evaluations"].append({"candidate": {"planets": []}})

        assert first_session is not second_session
        assert second_session["evaluations"] == []
        assert (
            first.hidden_args["analysis_session"]
            is not second.hidden_args["analysis_session"]
        )
    finally:
        first.hidden_args["analysis_session"].close()
        second.hidden_args["analysis_session"].close()


def test_released_task_loads_into_isolated_corral_environment(tmp_path):
    selector = tmp_path / "selector.json"
    selector.write_text(
        json.dumps(
            [
                {
                    "source": "synthetic",
                    "difficulty_min": 1,
                    "difficulty_max": 1,
                    "task_ids": ["seed300016_diff1"],
                    "max_evaluations": 2,
                }
            ]
        )
    )
    environments = create_environments(
        level=1, selector_path=selector, work_dir=tmp_path / "work"
    )

    assert list(environments) == ["seed300016_diff1"]
    environment = environments["seed300016_diff1"]
    assert set(environment.tools) == {
        "planet_from_fit",
        "python_repl",
        "evaluate_candidate",
    }

    benchmark_task = environment.current_task.scoring_inputs["benchmark_task"]
    truth_period = str(benchmark_task.truth_planets[0].P_days)
    prompt = environment.get_task_prompt()
    assert truth_period not in prompt
    assert "times_days" in prompt
    assert '"noise_jitter_ms"' in prompt
    assert "Only the final Corral answer is scored" in prompt
    assert "Lomb" not in prompt

    environment.configure_additional_apps()
    assert set(environment.hidden_args) == {
        "analysis_session",
        "benchmark_task",
        "evaluation_session",
        "star_mass_sun",
    }
    assert "hidden_args" not in environment.state.snapshot()

    planet = asdict(benchmark_task.truth_planets[0])
    planet.pop("m_true_mjup")
    environment.hidden_args["evaluation_session"]["evaluations"].append(
        {"candidate": {"planets": []}, "feedback": {"success": False}}
    )
    environment.state.submitted_answer = json.dumps(
        {"planets": [planet], "noise_jitter_ms": 0.0}
    )
    assert environment.score() == 1.0
