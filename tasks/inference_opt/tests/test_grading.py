"""Tests for Inspect scorer selection and grading."""

import json

from inference_opt.scoring_specs import spec_for


def test_chembench_uses_official_scorer():
    spec = spec_for("chembench", "numeric")
    assert spec.answer_format == "numeric"
    assert spec.scorer_name == "chembench"


def test_other_numeric_answers_use_inspect_numeric_match():
    spec = spec_for("gsm8k", "numeric")
    assert spec.answer_format == "numeric"
    assert spec.scorer_name == "match_numeric"


def test_multiple_choice_uses_inspect_choice_scorer():
    assert spec_for("mmlu_pro", "mcq_single").scorer_name == "choice"
    assert spec_for("mmlu_pro", "mcq_multi").multiple_correct


def test_free_text_uses_inspect_match_scorer():
    assert spec_for("bbh", "text").scorer_name == "match"


def test_math_uses_inspect_evals_sympy_scorer():
    assert spec_for("math_algebra", "numeric").scorer_name == "expression_exact_match_sympy"


def test_math_baseline_prompt_is_inspect_evals_template():
    """The pinned baseline asks exactly what inspect_evals' MATH eval asks."""
    from inference_opt.baselines import BASELINE_POLICY_SOURCE
    from inspect_evals.math.math import USER_PROMPT_TEMPLATE

    namespace: dict = {}
    exec(BASELINE_POLICY_SOURCE, namespace)

    class Question:
        benchmark = "math_algebra"
        text = "What is $1 + {1}$?"
        choices = None

    assert namespace["_prompt"](Question()) == USER_PROMPT_TEMPLATE.format(
        prompt=Question.text
    )


def test_math_reads_only_the_answer_line(tmp_path):
    """As in inspect_evals: a boxed answer without an ANSWER line is not read."""
    from inference_opt.grading import Grader, GradeSpec
    from inference_opt.outcomes import read_outcomes

    answers = {
        "math_algebra:a": "so x = 1/2\nANSWER: \\frac{1}{2}",
        "math_algebra:b": "ANSWER: 0.5",
        "math_algebra:c": "\\[\n\\boxed{\\frac{1}{2}}\n\\]",
        "math_algebra:d": "ANSWER: 3",
    }
    from inference_opt import datasets

    datasets.write_jsonl(
        tmp_path / "questions.jsonl",
        [
            {"item_id": item_id, "benchmark": "math_algebra", "question": "q",
             "answer_format": "numeric"}
            for item_id in answers
        ],
    )
    (tmp_path / "answers.json").write_text(json.dumps(answers))
    spec = GradeSpec(
        run_id="math", questions_path=str(tmp_path / "questions.jsonl"),
        answers_path=str(tmp_path / "answers.json"), out_dir=str(tmp_path),
        benchmark="math_algebra", split="train",
    )
    targets = dict.fromkeys(answers, "\\frac{1}{2}")
    [summary] = Grader().grade_many([(spec, targets)])
    assert summary.ok, summary.error
    correct = {o.item_id: o.correct for o in read_outcomes(spec.log_dir)}
    assert correct == {
        "math_algebra:a": True, "math_algebra:b": True,
        "math_algebra:c": False, "math_algebra:d": False,
    }


def test_chembench_multi_answer_needs_every_correct_option(tmp_path):
    """A list target would credit one correct letter and refuse the full answer."""
    from inference_opt.grading import Grader, GradeSpec
    from inference_opt.outcomes import read_outcomes

    record = {
        "item_id": "chembench:multi",
        "benchmark": "chembench",
        "question": "Which are noble gases?",
        "answer_format": "mcq_multi",
        "options": ["Iron", "Neon", "Argon"],
        "category": "general_chemistry",
    }
    answers = {"full": "[ANSWER]B,C[/ANSWER]", "partial": "[ANSWER]B[/ANSWER]"}
    graded = {}
    for name, completion in answers.items():
        out = tmp_path / name
        out.mkdir()
        (out / "q.jsonl").write_text(json.dumps(record) + "\n")
        (out / "a.json").write_text(json.dumps({"chembench:multi": completion}))
        spec = GradeSpec(
            run_id=name,
            questions_path=str(out / "q.jsonl"),
            answers_path=str(out / "a.json"),
            out_dir=str(out),
            benchmark="chembench",
        )
        summary = Grader().grade(spec, {"chembench:multi": "B,C"})
        assert summary.ok, summary.error
        graded[name] = read_outcomes(spec.log_dir)[0].correct
    assert graded == {"full": True, "partial": False}


def _gold_completion(benchmark: str, answer_format: str, target: str) -> str:
    """The gold answer, written the way the task tells a policy to answer."""
    if benchmark == "chembench":
        return f"[ANSWER]{target}[/ANSWER]"
    return f"ANSWER: {target}"


def test_every_gold_answer_grades_as_correct(tmp_path):
    """Catches any mismatch between the samples we build and a scorer's conventions."""
    from inference_opt import datasets
    from inference_opt.grading import Grader, GradeSpec
    from inference_opt.outcomes import read_outcomes

    jobs = []
    for benchmark in datasets.BENCHMARKS:
        for split in ("train", "test"):
            items = datasets.load_items(benchmark, split)
            targets = datasets.load_targets(benchmark, split)
            out = tmp_path / f"{benchmark}-{split}"
            out.mkdir()
            datasets.write_jsonl(
                out / "questions.jsonl", [datasets.public_record(item) for item in items]
            )
            answers = {
                item.item_id: _gold_completion(
                    benchmark, item.answer_format, targets[item.item_id]
                )
                for item in items
            }
            (out / "answers.json").write_text(json.dumps(answers))
            spec = GradeSpec(
                run_id=f"{benchmark}-{split}",
                questions_path=str(out / "questions.jsonl"),
                answers_path=str(out / "answers.json"),
                out_dir=str(out),
                benchmark=benchmark,
                split=split,
            )
            jobs.append((spec, targets))

    wrong = {}
    for (spec, _), summary in zip(jobs, Grader().grade_many(jobs), strict=True):
        assert summary.ok, summary.error
        missed = [i.item_id for i in read_outcomes(spec.log_dir) if not i.correct]
        if missed:
            wrong[spec.run_id] = missed
    assert not wrong, wrong
