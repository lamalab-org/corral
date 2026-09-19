"""Environment construction, the tool surface, and the scoring ladder."""

from __future__ import annotations

import json

import pytest
from inference_opt.env import create_environments
from inference_opt.score import resolve_submission

BENCHMARKS = {"gsm8k", "mmlu_pro", "gpqa_diamond", "bbh", "chembench", "arc_challenge"}


@pytest.fixture
def environments(tmp_path):
    return create_environments(level=1, work_dir=str(tmp_path / "work"))


class TestConstruction:
    @pytest.mark.parametrize("level", [1, 2])
    def test_twelve_tasks_per_level(self, tmp_path, level):
        built = create_environments(level=level, work_dir=str(tmp_path / f"w{level}"))
        assert len(built) == 12
        models = {
            len(environment.current_task.initial_input["models"])
            for environment in built.values()
        }
        assert models == ({1} if level == 1 else {2})

    def test_every_benchmark_is_covered(self, environments):
        covered = {
            environment.current_task.initial_input["benchmark"]
            for environment in environments.values()
        }
        assert covered == BENCHMARKS

    def test_building_needs_no_server_or_dataset(self, tmp_path, monkeypatch):
        """Corral's contract suite builds every task with no GPU and no network."""
        monkeypatch.setenv("CORRAL_INFERENCE_DATA_DIR", str(tmp_path / "absent"))
        monkeypatch.delenv("CORRAL_VLLM_URL", raising=False)
        assert create_environments(level=1, work_dir=str(tmp_path / "w")) != {}

    def test_level_two_tasks_are_joint(self, tmp_path):
        built = create_environments(level=2, work_dir=str(tmp_path / "w2"))
        assert all(
            environment.current_task.initial_input["joint"]
            for environment in built.values()
        )


class TestToolSurface:
    def test_the_agent_can_write_its_policy(self, environments):
        """Without workspace tools the agent cannot produce a submission at all."""
        tools = environments["gsm8k_a"].tools
        assert {"write_file", "read_file", "list_files"} <= set(tools)

    def test_all_nine_task_tools_are_present(self, environments):
        tools = environments["gsm8k_a"].tools
        assert {
            "get_baseline", "reveal_train_questions", "query_student",
            "dry_run_policy", "evaluate_candidate", "inspect_failures",
            "compare_runs", "get_budget", "submit_policy",
        } <= set(tools)

    def test_inference_tools_are_trusted_for_the_first_iteration(self, environments):
        """The environment uses trusted tools and committed session state."""
        tools = environments["gsm8k_a"].tools
        trusted = {name for name, tool in tools.items() if getattr(tool, "trusted", False)}
        assert trusted == {
            "get_baseline", "reveal_train_questions", "query_student",
            "dry_run_policy", "evaluate_candidate", "inspect_failures",
            "compare_runs", "get_budget", "submit_policy",
        }

    def test_tools_are_bound_to_their_own_task(self, environments):
        """One shared pool would bind every task to whichever was built last."""
        first = environments["gsm8k_a"].tools["get_baseline"]
        second = environments["chembench_a"].tools["get_baseline"]
        assert first is not second


class TestPrompt:
    def test_prompt_states_the_rule_and_how_to_finish(self, environments):
        from corral.core.state import ExecutionState

        environment = environments["gsm8k_a"]
        prompt = environment.current_task.prompt_fn(environment, ExecutionState)
        assert "submit_answer" in prompt
        assert "only" in prompt and "student client" in prompt
        assert "gsm8k" in prompt

    def test_prompt_does_not_name_strategies(self, environments):
        """Which strategies the agent reaches for is what we are measuring."""
        from corral.core.state import ExecutionState

        environment = environments["gsm8k_a"]
        prompt = environment.current_task.prompt_fn(environment, ExecutionState).lower()
        for leak in ("self-consistency", "majority vote", "debate", "chain of thought"):
            assert leak not in prompt


class TestSubmissionResolution:
    def _policy(self, root):
        policy = root / "policy"
        policy.mkdir(parents=True, exist_ok=True)
        (policy / "policy.py").write_text(
            "class Policy:\n    def solve(self, q, ctx): return 'A'\n", encoding="utf-8"
        )
        return policy

    def test_staged_submission_is_preferred(self, tmp_path):
        self._policy(tmp_path)
        (tmp_path / "submission.json").write_text(
            json.dumps({"policy_dir": "policy"}), encoding="utf-8"
        )
        resolved, _ = resolve_submission(str(tmp_path / "submission.json"), tmp_path)
        assert resolved == (tmp_path / "policy").resolve()

    def test_a_bare_directory_still_resolves(self, tmp_path):
        policy = self._policy(tmp_path)
        resolved, notes = resolve_submission(str(policy), tmp_path)
        assert resolved == policy.resolve()
        assert notes

    def test_garbage_falls_back_to_the_workspace_policy(self, tmp_path):
        self._policy(tmp_path)
        resolved, _ = resolve_submission("not-a-real-path", tmp_path)
        assert resolved == (tmp_path / "policy").resolve()

    def test_nothing_to_score_is_reported_not_guessed(self, tmp_path):
        resolved, notes = resolve_submission("nope", tmp_path)
        assert resolved is None
        assert any("no policy.py" in note for note in notes)
