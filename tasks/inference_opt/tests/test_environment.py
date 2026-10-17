"""Environment construction, the tool surface, and the scoring ladder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inference_opt import datasets
from inference_opt.env import create_environments, load_tasks_from_json
from inference_opt.score import template_of

TASK_BENCHMARKS = {
    1: {
        "math_algebra",
        "math_prealgebra",
        "math_number_theory",
        "math_counting_probability",
        "math_geometry",
    },
    2: {"gsm_symbolic", "gsm_p1", "gsm_p2"},
}


@pytest.fixture
def environments(tmp_path):
    return create_environments(level=1, work_dir=str(tmp_path / "work"))


class TestConstruction:
    def test_unmeasured_example_baselines_stop_the_build(self, tmp_path, monkeypatch):
        """Otherwise every revealed example would show no zero-shot reply."""
        from inference_opt import datasets

        monkeypatch.delenv("CORRAL_INFERENCE_ALLOW_UNMEASURED")
        monkeypatch.setattr(
            datasets, "example_baselines_path", lambda *_: tmp_path / "none"
        )
        with pytest.raises(RuntimeError, match="measure_baselines.py --splits examples"):
            create_environments(level=1, work_dir=str(tmp_path / "w"))

    @pytest.mark.parametrize("level", [1, 2])
    def test_ten_tasks_per_level(self, tmp_path, level):
        built = create_environments(level=level, work_dir=str(tmp_path / f"w{level}"))
        assert len(built) == 10
        models = {
            len(environment.current_task.initial_input["models"])
            for environment in built.values()
        }
        assert models == {1}

    @pytest.mark.parametrize("level", [1, 2])
    def test_every_benchmark_of_the_level_is_covered(self, tmp_path, level):
        built = create_environments(level=level, work_dir=str(tmp_path / f"w{level}"))
        covered = {
            environment.current_task.initial_input["benchmark"]
            for environment in built.values()
        }
        assert covered == TASK_BENCHMARKS[level]

    def test_building_needs_no_server_or_dataset(self, tmp_path, monkeypatch):
        """Corral's contract suite builds every task with no GPU and no network."""
        monkeypatch.setenv("CORRAL_INFERENCE_DATA_DIR", str(tmp_path / "absent"))
        monkeypatch.delenv("CORRAL_VLLM_URL", raising=False)
        assert create_environments(level=1, work_dir=str(tmp_path / "w")) != {}

    @pytest.mark.parametrize(("level", "metric"), [(1, "accuracy"), (2, "template_consistency")])
    def test_each_level_matches_a_larger_model_on_its_own_metric(self, tmp_path, level, metric):
        built = create_environments(level=level, work_dir=str(tmp_path / f"w{level}"))
        for environment in built.values():
            rule = environment.current_task.initial_input["pass_rule"]
            assert rule["kind"] == "reference"
            assert rule["metric"] == metric
            assert bool(rule.get("no_regression")) == (level == 2)


class TestToolSurface:
    def test_the_agent_can_write_its_policy(self, environments):
        """Without workspace tools the agent cannot produce a submission at all."""
        tools = environments["math_algebra_b"].tools
        assert {"write_file", "read_file", "list_files"} <= set(tools)

    def test_all_nine_task_tools_are_present(self, environments):
        tools = environments["math_algebra_b"].tools
        assert {
            "get_baseline", "reveal_examples", "query_student",
            "dry_run_policy", "evaluate_candidate", "inspect_failures",
            "compare_runs", "get_budget", "submit_policy",
        } <= set(tools)

    def test_inference_tools_are_trusted(self, environments):
        """The environment uses trusted tools and committed session state."""
        tools = environments["math_algebra_b"].tools
        trusted = {name for name, tool in tools.items() if getattr(tool, "trusted", False)}
        assert trusted == {
            "get_baseline", "reveal_examples", "query_student",
            "dry_run_policy", "evaluate_candidate", "inspect_failures",
            "compare_runs", "get_budget", "submit_policy",
        }

    def test_tools_are_bound_to_their_own_task(self, environments):
        """One shared pool would bind every task to whichever was built last."""
        first = environments["math_algebra_b"].tools["get_baseline"]
        second = environments["math_prealgebra_d"].tools["get_baseline"]
        assert first is not second


class TestPrompt:
    def test_prompt_states_the_rule_and_how_to_finish(self, environments):
        from corral.core.state import ExecutionState

        environment = environments["math_algebra_b"]
        prompt = environment.current_task.prompt_fn(environment, ExecutionState)
        assert "submit_answer" in prompt
        assert "modify model weights" in prompt
        assert "frozen student model" in prompt
        assert "math_algebra" in prompt
        assert "Llama 3.1 70B Instruct" in prompt
        assert "76.7%" in prompt  # the larger model's train accuracy, the target

    def test_prompt_does_not_name_strategies(self, environments):
        """The prompt leaves strategy choice to the policy."""
        from corral.core.state import ExecutionState

        environment = environments["math_algebra_b"]
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
    POLICY = (
        "class Policy:\n"
        "    def run(self, questions, ctx):\n"
        "        texts = ctx.student.generate([q.text for q in questions])\n"
        "        ctx.answer({q.id: t for q, t in zip(questions, texts)})\n"
    )

    def _submit(self, tmp_path, ledger, source=POLICY):
        from inference_opt.tools import create_tools

        workspace = _workspace_with_policy(tmp_path, source)
        submit = create_tools(_task_config(1, "math_algebra_b"), "")["submit_policy"]._func
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
        assert {"student_b_accuracy", "student_b_delta", "raw_delta"} <= set(metrics)
        assert details.metadata["outcome"] == "ok"
        assert details.metadata["policy_hash"] == ledger["final"]["policy_hash"]
        assert details.feedback.startswith("score ")

    def test_no_submission_scores_zero_with_a_reason(self):
        from inference_opt.score import StateScorer

        result = StateScorer().evaluate_state(self._state({}))
        assert result["score"] == 0.0
        assert result["metadata"]["outcome"] == "no_submission"

    def test_the_score_carries_the_train_history(self):
        from inference_opt.score import StateScorer

        runs = [{"run_id": "exp-1", "kind": "experiment", "delta": 0.1}]
        result = StateScorer().evaluate_state(self._state({"runs": runs}))
        assert result["metadata"]["train_history"] == runs

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

    def test_a_policy_can_be_submitted_only_once(self, tmp_path):
        from inference_opt.tools import create_tools

        workspace = _workspace_with_policy(tmp_path, self.POLICY)
        submit = create_tools(_task_config(1, "math_algebra_b"), "")["submit_policy"]._func
        ledger = {}
        args = {"policy_path": "policy", "work_dir": str(workspace), "inference_state": ledger}
        assert submit(**args).startswith("Submitted")
        first = ledger["final"]
        out = submit(**args)
        assert out.startswith("NOT submitted: a policy was already submitted")
        assert ledger["final"] == first and ledger["submissions"] == 1


def test_a_refunded_run_keeps_its_folder_and_id(tmp_path):
    from inference_opt.tools import create_tools

    workspace = _workspace_with_policy(tmp_path, "raise ImportError('nope')\n")
    dry_run = create_tools(_task_config(1, "math_algebra_b"), "")["dry_run_policy"]._func
    ledger = {}
    dry_run(policy_path="policy", work_dir=str(workspace), inference_state=ledger)
    (workspace / "policy" / "policy.py").write_text(TestSubmission.POLICY)

    out = dry_run(policy_path="policy", work_dir=str(workspace), inference_state=ledger)

    assert json.loads(out.split("\n")[1])["run_id"] == "dry-2"
    assert sorted(path.name for path in (workspace / "runs").iterdir()) == [
        "dry-1",
        "dry-2",
    ]


@pytest.mark.parametrize(
    ("source", "warning"),
    [
        (
            "def run(questions, ctx):\n    ctx.answer({q.id: '' for q in questions})\n",
            "made no student calls and gave no non-empty answer",
        ),
        (TestSubmission.POLICY, None),
    ],
)
def test_a_dry_run_that_never_reaches_the_student_is_not_ok(tmp_path, source, warning):
    from inference_opt.tools import create_tools

    workspace = _workspace_with_policy(tmp_path, source)
    dry_run = create_tools(_task_config(1, "math_algebra_b"), "")["dry_run_policy"]._func

    headline = dry_run(
        policy_path="policy", work_dir=str(workspace), inference_state={}
    ).split("\n")[0]

    if warning:
        assert warning in headline and "OK" not in headline
    else:
        assert "Dry run OK" in headline


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
        query = environments["math_algebra_b"].tools["query_student"]._func
        out = query(prompt="3/5 of 50?", max_tokens=256, n=2, work_dir="", inference_state={})
        summary, payload = out.split("\n")[:2]
        data = json.loads(payload)
        assert data["completions"] == ["", "30"]
        assert data["finish_reasons"] == ["length", "stop"]
        assert "None" not in data["completions"]
        assert "1 hit max_tokens=256" in summary
        assert data["reasoning_tail"] == ["...so 3/5 of 50 is", ""]

    def test_max_tokens_is_capped_like_a_policy_call(self, environments, monkeypatch):
        from inference_opt.client import StudentCompletion
        from inference_opt.runner import MAX_TOKENS_CAP

        requested = []

        def probe(*_args, max_tokens, **_kwargs):
            requested.append(max_tokens)
            return [StudentCompletion("30", "stop")]

        monkeypatch.setattr("inference_opt.tools.probe_student", probe)
        query = environments["math_algebra_b"].tools["query_student"]._func
        query(prompt="3/5 of 50?", max_tokens=10**6, work_dir="", inference_state={})
        assert requested == [MAX_TOKENS_CAP]


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
        dry = environments["math_algebra_b"].tools["dry_run_policy"]._func
        workspace = _workspace_with_policy(
            tmp_path, "def run(questions, ctx):\n    ctx.answer(questions[0].id, 'A')\n"
        )
        out = dry(policy_path="policy", work_dir=str(workspace), inference_state={})
        headline, payload = out.split("\n")[:2]
        assert headline == "Dry run FAILED: KeyError: 'choices'"
        assert "solver.py" in json.loads(payload)["error"]


class TestPassRules:
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
            (1, "math_algebra_b", "all", 1.0),
            (1, "math_algebra_b", "just_short", 0.0),
            (2, "gsm_p1_d", "all", 1.0),
            (2, "gsm_p1_d", "just_short", 0.0),
        ],
    )
    def test_score_is_one_only_when_the_student_matches_the_larger_model(
        self, level, task_id, outcome, expected
    ):
        """Level 1 compares accuracy, level 2 template consistency."""
        from inference_opt.runner import ModelRun
        from inference_opt.score import final_result

        config = _task_config(level, task_id)
        rule = config["pass_rule"]
        ((model, reference),) = rule["references"].items()
        question_ids = [
            item.item_id for item in datasets.load_items(config["benchmark"], "test")
        ]
        n = len(question_ids)
        if outcome == "all":
            correct = dict.fromkeys(question_ids, True)
        elif rule["metric"] == "accuracy":
            # One question below the larger model.
            below = round(reference["test"] * n) - 1
            correct = {qid: i < below for i, qid in enumerate(question_ids)}
        else:
            # Every instance right but one per template, in one template fewer
            # than the larger model's share: accuracy stays high, consistency falls short.
            templates = sorted({template_of(qid) for qid in question_ids})
            solved = set(templates[: round(reference["test"] * len(templates)) - 1])
            seen: set[str] = set()
            correct = {}
            for qid in question_ids:
                template = template_of(qid)
                first = template not in seen
                seen.add(template)
                correct[qid] = template in solved or not first
        runs = {
            model: ModelRun(
                model=model,
                split="test",
                question_ids=question_ids,
                answers=dict.fromkeys(question_ids, "ANSWER: 1"),
                correct=correct,
                calls_used=n,
            )
        }
        result = final_result(config, runs)
        assert result["score"] == expected
        assert result["metadata"]["metrics"]["passed"] == expected

    def test_robustness_must_not_cost_accuracy(self):
        """Matching the larger model's consistency is not enough if accuracy falls."""
        from inference_opt.runner import ModelRun
        from inference_opt.score import final_result

        config = _task_config(2, "gsm_p1_d")
        ((model, reference),) = config["pass_rule"]["references"].items()
        question_ids = [item.item_id for item in datasets.load_items("gsm_p1", "test")]
        templates = sorted({template_of(qid) for qid in question_ids})
        # Exactly the larger model's share of templates solved, every other question wrong.
        solved = set(templates[: round(reference["test"] * len(templates))])
        correct = {qid: template_of(qid) in solved for qid in question_ids}
        run = ModelRun(
            model=model, split="test", question_ids=question_ids,
            answers=dict.fromkeys(question_ids, "ANSWER: 1"), correct=correct,
            calls_used=len(question_ids),
        )
        accuracy = sum(correct.values()) / len(correct)
        result = final_result(config, {model: run})
        assert result["metadata"]["per_model"][model]["template_consistency"] >= reference["test"]
        assert result["score"] == (1.0 if accuracy >= config["baselines"][model] else 0.0)

    @pytest.mark.parametrize("level", [1, 2])
    def test_the_headroom_rule_still_scores(self, level):
        from inference_opt.runner import ModelRun
        from inference_opt.score import final_result

        config = _task_config(level, "math_algebra_b" if level == 1 else "gsm_p1_d")
        config["pass_rule"] = {"kind": "headroom", "min_closed": 0.5}
        (model,) = config["models"]
        question_ids = [f"q{i}" for i in range(30)]
        run = ModelRun(
            model=model, split="test", question_ids=question_ids,
            answers=dict.fromkeys(question_ids, "ANSWER: 1"),
            correct=dict.fromkeys(question_ids, True), calls_used=30,
        )
        assert final_result(config, {model: run})["score"] == 1.0


def test_a_reference_task_passes_on_a_tie_with_the_larger_model():
    from inference_opt.runner import ModelRun
    from inference_opt.score import final_result

    config = _task_config(1, "math_counting_probability_b")
    bar = config["pass_rule"]["references"]["student_b"]["test"]
    question_ids = [f"q{i}" for i in range(30)]
    tie = round(bar * 30)
    run = ModelRun(
        model="student_b",
        split="test",
        question_ids=question_ids,
        answers=dict.fromkeys(question_ids, "ANSWER: 1"),
        correct={qid: index < tie for index, qid in enumerate(question_ids)},
        calls_used=30,
    )
    result = final_result(config, {"student_b": run})
    assert result["score"] == 1.0
    assert "(Llama 3.1 70B Instruct 33%)" in result["feedback"]
