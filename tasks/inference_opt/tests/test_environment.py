"""Environment construction, the tool surface, and the scoring ladder."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from inference_opt.env import create_environments, load_tasks_from_json

TASK_BENCHMARKS = {"mmlu_pro", "bbh", "gpqa_diamond", "math", "chembench"}


@pytest.fixture
def environments(tmp_path):
    return create_environments(level=1, work_dir=str(tmp_path / "work"))


class TestConstruction:
    @pytest.mark.parametrize("level", [1, 2])
    def test_ten_tasks_per_level(self, tmp_path, level):
        built = create_environments(level=level, work_dir=str(tmp_path / f"w{level}"))
        assert len(built) == 10
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
        assert covered == TASK_BENCHMARKS

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
        tools = environments["mmlu_pro_a"].tools
        assert {"write_file", "read_file", "list_files"} <= set(tools)

    def test_all_nine_task_tools_are_present(self, environments):
        tools = environments["mmlu_pro_a"].tools
        assert {
            "get_baseline", "reveal_train_questions", "query_student",
            "dry_run_policy", "evaluate_candidate", "inspect_failures",
            "compare_runs", "get_budget", "submit_policy",
        } <= set(tools)

    def test_inference_tools_are_trusted(self, environments):
        """The environment uses trusted tools and committed session state."""
        tools = environments["mmlu_pro_a"].tools
        trusted = {name for name, tool in tools.items() if getattr(tool, "trusted", False)}
        assert trusted == {
            "get_baseline", "reveal_train_questions", "query_student",
            "dry_run_policy", "evaluate_candidate", "inspect_failures",
            "compare_runs", "get_budget", "submit_policy",
        }

    def test_tools_are_bound_to_their_own_task(self, environments):
        """One shared pool would bind every task to whichever was built last."""
        first = environments["mmlu_pro_a"].tools["get_baseline"]
        second = environments["chembench_a"].tools["get_baseline"]
        assert first is not second


class TestPrompt:
    def test_prompt_states_the_rule_and_how_to_finish(self, environments):
        from corral.core.state import ExecutionState

        environment = environments["mmlu_pro_a"]
        prompt = environment.current_task.prompt_fn(environment, ExecutionState)
        assert "submit_answer" in prompt
        assert "modify model weights" in prompt
        assert "frozen student model" in prompt
        assert "mmlu_pro" in prompt

    def test_prompt_does_not_name_strategies(self, environments):
        """The prompt leaves strategy choice to the policy."""
        from corral.core.state import ExecutionState

        environment = environments["mmlu_pro_a"]
        prompt = environment.current_task.prompt_fn(environment, ExecutionState).lower()
        for leak in ("self-consistency", "majority vote", "debate", "chain of thought"):
            assert leak not in prompt


def _task_config(level, task_id):
    root = Path(__file__).resolve().parents[1] / f"environments/level_{level}/tasks_json"
    config = dict(load_tasks_from_json(root)[task_id].initial_input)
    config["model_specs"] = dict.fromkeys(config["models"], "mockllm/model")
    config["base_urls"] = {}
    return config


def _workspace_with_policy(tmp_path, source):
    workspace = tmp_path / "workspace"
    (workspace / "policy").mkdir(parents=True, exist_ok=True)
    (workspace / "policy" / "policy.py").write_text(source, encoding="utf-8")
    return workspace


class TestSubmission:
    POLICY = "class Policy:\n    def solve(self, q, ctx): return ctx.student.generate(q.text)\n"

    def _submit(self, tmp_path, ledger, source=POLICY):
        from inference_opt.tools import create_tools

        workspace = _workspace_with_policy(tmp_path, source)
        submit = create_tools(_task_config(1, "mmlu_pro_a"), "")["submit_policy"]._func
        return submit(policy_path="policy", work_dir=str(workspace), inference_state=ledger)

    def _state(self, ledger):
        from corral.core.state import EnvironmentState, ExecutionState

        return ExecutionState(
            through_commit_hash="a" * 64,
            execution_id="x",
            branch_id="main",
            environment=EnvironmentState(values={"resources": {"inference_state": ledger}}),
        )

    def test_the_test_run_is_recorded_but_not_shown(self, tmp_path):
        ledger = {}
        out = self._submit(tmp_path, ledger)
        assert out.startswith("Submitted.")
        final = ledger["final"]
        assert final["submission"] == 1 and len(final["policy_hash"]) == 64
        # The reply carries no test score, accuracy or per-model result.
        for leak in ("accuracy", "headroom", "delta", "score "):
            assert leak not in out
        assert "per_model" in final["result"]["metadata"]
        assert not (tmp_path / "workspace" / "runs").exists()

    def test_the_host_scorer_only_reads_the_recorded_result(self, tmp_path):
        from inference_opt.score import StateScorer

        from corral.evaluation.scorer import SubmissionScore

        ledger = {}
        self._submit(tmp_path, ledger)
        details = SubmissionScore.model_validate(
            StateScorer().evaluate_state(self._state(ledger))
        )
        metrics = details.metadata["metrics"]
        assert {"student_a_accuracy", "student_a_delta", "raw_delta"} <= set(metrics)
        assert details.metadata["outcome"] == "ok"
        assert details.metadata["policy_hash"] == ledger["final"]["policy_hash"]
        assert details.feedback.startswith("score ")

    def test_no_submission_scores_zero_with_a_reason(self):
        from inference_opt.score import StateScorer

        result = StateScorer().evaluate_state(self._state({}))
        assert result["score"] == 0.0
        assert result["metadata"]["outcome"] == "no_submission"

    def test_an_environment_fault_is_raised_not_scored(self):
        from inference_opt.score import HarnessError, StateScorer

        state = self._state({"final": {"harness_error": "student unreachable"}})
        with pytest.raises(HarnessError, match="student unreachable"):
            StateScorer().evaluate_state(state)

    def test_a_policy_that_does_not_load_uses_no_submission(self, tmp_path):
        ledger = {}
        out = self._submit(tmp_path, ledger, "raise ImportError('nope')\n")
        assert out.startswith("NOT submitted")
        assert ledger.get("submissions", 0) == 0 and ledger.get("final") is None

    def test_submissions_are_capped(self, tmp_path):
        ledger = {"submissions": 3}
        out = self._submit(tmp_path, ledger)
        assert out.startswith("NOT submitted: all 3 submissions are used")


def test_a_refunded_run_keeps_its_folder_and_id(tmp_path):
    from inference_opt.tools import create_tools

    workspace = _workspace_with_policy(tmp_path, "raise ImportError('nope')\n")
    dry_run = create_tools(_task_config(1, "mmlu_pro_a"), "")["dry_run_policy"]._func
    ledger = {}
    dry_run(policy_path="policy", work_dir=str(workspace), inference_state=ledger)
    (workspace / "policy" / "policy.py").write_text(TestSubmission.POLICY)

    out = dry_run(policy_path="policy", work_dir=str(workspace), inference_state=ledger)

    assert json.loads(out.split("\n")[1])["run_id"] == "dry-2"
    assert sorted(path.name for path in (workspace / "runs").iterdir()) == [
        "dry-1",
        "dry-2",
    ]


class TestQueryStudent:
    def test_a_cut_off_answer_is_explained_not_shown_as_none(
        self, environments, monkeypatch
    ):
        from inference_opt.client import StudentCompletion

        monkeypatch.setattr(
            "inference_opt.tools.probe_student",
            lambda *a, **k: [
                StudentCompletion("", "length", "...so 3/5 of 50 is"),
                StudentCompletion("30", "stop"),
            ],
        )
        query = environments["mmlu_pro_a"].tools["query_student"]._func
        out = query(prompt="3/5 of 50?", max_tokens=256, n=2, work_dir="", inference_state={})
        summary, payload = out.split("\n")[:2]
        data = json.loads(payload)
        assert data["completions"] == ["", "30"]
        assert data["finish_reasons"] == ["length", "stop"]
        assert "None" not in data["completions"]
        assert "1 hit max_tokens=256" in summary
        assert data["reasoning_tail"] == ["...so 3/5 of 50 is", ""]


GROUPED_TRACEBACK = """  + Exception Group Traceback (most recent call last):
  |   File "/usr/local/lib/python3.12/site-packages/inspect_ai/_eval/eval.py", line 657, in eval_async
  |     async with anyio.create_task_group() as tg:
  |                ^^^^^^^^^^^^^^^^^^^^^^^^^
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/opt/corral/tasks/inference_opt/inference_opt/eval_runner/solver.py", line 95, in solve
    |     raw = await _invoke(policy.solve, question, context)
    | KeyError: 'choices'
    +------------------------------------
"""


class TestToolErrors:
    def test_the_real_exception_is_named_not_the_group(self):
        from inference_opt.errors import error_line

        assert error_line(GROUPED_TRACEBACK) == "KeyError: 'choices'"

    def test_the_tail_of_a_long_traceback_is_kept(self):
        from inference_opt.errors import error_tail

        long = "noise\n" * 1000 + GROUPED_TRACEBACK
        tail = error_tail(long, 500)
        assert tail.startswith("...")
        assert "KeyError: 'choices'" in tail
        assert "solver.py" in tail

    def test_a_failed_dry_run_names_the_real_exception(
        self, environments, monkeypatch, tmp_path
    ):
        from inference_opt.runner import ModelRun

        monkeypatch.setattr(
            "inference_opt.tools.run_policy",
            lambda **kwargs: {
                model: ModelRun(
                    model=model,
                    split="train",
                    question_ids=list(kwargs["item_ids"]),
                    error=GROUPED_TRACEBACK,
                )
                for model in kwargs["models"]
            },
        )
        dry = environments["mmlu_pro_a"].tools["dry_run_policy"]._func
        workspace = _workspace_with_policy(
            tmp_path, "class Policy:\n    def solve(self, q, ctx): return 'A'\n"
        )
        out = dry(policy_path="policy", work_dir=str(workspace), inference_state={})
        headline, payload = out.split("\n")[:2]
        assert headline == "Dry run FAILED: KeyError: 'choices'"
        assert "solver.py" in json.loads(payload)["error"]


class TestHeadroomPassRule:
    def test_headroom_closed_counts_questions(self):
        from inference_opt.score import headroom_closed

        assert headroom_closed(29, 0.9333, 30) == 0.5  # 1 of the 2 missed
        assert headroom_closed(30, 0.8667, 30) == 1.0
        assert headroom_closed(20, 0.5, 30) < 0.5
        assert headroom_closed(12, 0.5, 30) < 0  # worse than the baseline
        assert headroom_closed(30, 1.0, 30) is None  # nothing left to improve

    @pytest.mark.parametrize(
        ("level", "task_id", "outcome", "expected"),
        [
            (1, "mmlu_pro_a", {"student_a": "all"}, 1.0),
            (1, "mmlu_pro_a", {"student_a": "just_short"}, 0.0),
            # Level 2 needs every student to pass: one perfect and one just short fails.
            (2, "mmlu_pro_ab", {"student_a": "all", "student_b": "just_short"}, 0.0),
            (2, "mmlu_pro_ab", {"student_a": "all", "student_b": "all"}, 1.0),
        ],
    )
    def test_score_is_one_only_when_every_student_closes_half(
        self, level, task_id, outcome, expected
    ):
        from inference_opt.runner import ModelRun
        from inference_opt.score import final_result

        config = _task_config(level, task_id)
        question_ids = [f"q{i}" for i in range(30)]

        def n_correct(model):
            if outcome[model] == "all":
                return 30
            # One question fewer than closing half the headroom over the baseline.
            baseline_correct = round(config["baselines"][model] * 30)
            return baseline_correct + math.ceil((30 - baseline_correct) / 2) - 1

        runs = {}
        for model in config["models"]:
            n = n_correct(model)
            runs[model] = ModelRun(
                model=model,
                split="test",
                question_ids=question_ids,
                answers=dict.fromkeys(question_ids, "ANSWER: A"),
                correct={qid: index < n for index, qid in enumerate(question_ids)},
                calls_used=30,
            )
        result = final_result(config, runs)
        assert result["score"] == expected
        assert result["metadata"]["metrics"]["passed"] == expected
        for model in config["models"]:
            assert f"{model}_headroom_closed" in result["metadata"]["metrics"]
