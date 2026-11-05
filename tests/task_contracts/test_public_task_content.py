"""Task instructions explain usable deliverables without disclosing grading rules.

These checks need no domain services. Scientific scores (e.g. questionnaire
totals) and analysis of model predictions remain legitimate task content.
"""

import json
import re
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"
GRADING_LANGUAGE = re.compile(
    r"\b(?:scorers?|graders?|grading|graded|headroom|pass_rule|"
    r"benchmark score|evaluation pipeline|scores? (?:0|1|nothing)|"
    r"evaluated against|held-out test)\b",
    re.IGNORECASE,
)
TASK_FILES = sorted(
    path
    for path in TASKS.glob("*/environments/level_*/*/*.json")
    if path.parent.name in {"tasks_json", "subtasks_json"}
)
INFERENCE_ROOT = TASKS / "inference_opt"
INFERENCE_TASKS = sorted(INFERENCE_ROOT.glob("environments/level_*/tasks_json/*.json"))


def assert_no_grading_language(text, source):
    match = GRADING_LANGUAGE.search(text)
    assert match is None, f"{source}: agent-facing grading language {match.group()!r}"


def assert_unsectioned(text, source):
    # Schemas and data may still use lists or code blocks; the request is prose.
    headings = re.compile(
        r"^\s*#{1,6}\s|^(?:Task|Description|Task Description|Required submission format|"
        r"Available input data|Available workspace files|Benchmark|Budget|Preloaded):",
        re.MULTILINE,
    )
    assert headings.search(text) is None, f"{source}: sectioned task instructions"


def test_shipped_task_instructions_describe_deliverables():
    assert {path.parts[-5] for path in TASK_FILES} == {
        "afm",
        "corral_md",
        "inference_opt",
        "psychometrics",
        "resistor_network",
        "retrosynthesis",
        "samplemath",
        "spectra_elucidation",
        "stargazer",
        "wetlab",
    }
    for path in TASK_FILES:
        data = json.loads(path.read_text())
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            for field in ("name", "description", "submission_format"):
                assert_no_grading_language(str(entry.get(field, "")), f"{path}:{field}")
                assert_unsectioned(str(entry.get(field, "")), f"{path}:{field}")


@pytest.mark.parametrize(
    "path", INFERENCE_TASKS, ids=lambda p: f"{p.parts[-3]}/{p.name}"
)
def test_inference_prompt_is_independent_of_private_grading_rules(path):
    task_prompt = runpy.run_path(str(INFERENCE_ROOT / "inference_opt/task_prompts.py"))[
        "task_prompt"
    ]
    task = json.loads(path.read_text())
    config = task["initial_input"]

    def render(inputs):
        env = SimpleNamespace(
            current_task=SimpleNamespace(
                description=task["description"], initial_input=inputs
            )
        )
        return task_prompt(env, None)

    prompt = render(config)
    assert_no_grading_language(prompt, path)
    assert_unsectioned(prompt, path)
    # Private grading changes must not influence the instructions an agent sees.
    assert prompt == render(
        {
            **config,
            "pass_rule": {"kind": "continuous"},
            "baselines": dict.fromkeys(config["models"], 0.987654),
            "baseline_items": {},
            "scale": 0.123456,
        }
    )
    for convention in (
        "policy/policy.py",
        "guide/policy_api.md",
        "ANSWER: <answer>",
        "[ANSWER]<answer>[/ANSWER]",
        "submit_policy('policy')",
        "submit_answer('submission.json')",
    ):
        assert convention in prompt
    assert str(config["final_max_student_calls"]) in prompt
    assert "{scoring}" not in prompt


def test_public_guides_and_starter_files_describe_usable_outputs():
    files = [
        *TASKS.glob("psychometrics/artifacts/**/*.md"),
        *TASKS.glob("psychometrics/src/corral_psychometrics/generators/**/task_*.md"),
        *INFERENCE_ROOT.glob("inference_opt/guide/*.md"),
        *INFERENCE_ROOT.glob("inference_opt/templates/**/*.py"),
        *TASKS.glob("corral_md/src/corral_md/submission_templates/**/*.json"),
        TASKS / "stargazer/src/stargazer/public_rv.py",
    ]
    for path in files:
        assert_no_grading_language(path.read_text(), path)


def test_regenerated_inference_descriptions_match_shipped_tasks():
    generate = runpy.run_path(str(INFERENCE_ROOT / "scripts/generate_tasks.py"))[
        "_task"
    ]
    for path in INFERENCE_TASKS:
        task = json.loads(path.read_text())
        config = task["initial_input"]
        level = int(path.parts[-3].removeprefix("level_"))
        regenerated = generate(config["benchmark"], config["models"], level)
        assert regenerated["description"] == task["description"]


def test_generated_md_workspace_guides_describe_reproducible_artifacts(monkeypatch):
    monkeypatch.syspath_prepend(str(TASKS / "corral_md/src"))
    from corral_md.submission_examples import example_files

    for level in (1, 2):
        for number in range(1, 11):
            files = example_files(number, level=level)
            assert "manifest.scripts" in files["README.md"]
            assert_unsectioned(files["README.md"], f"MD level {level} task {number}")
            for name, content in files.items():
                assert_no_grading_language(
                    content, f"MD level {level} task {number}/{name}"
                )


def test_stargazer_public_tool_descriptions_explain_analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(TASKS / "stargazer/src"))
    from stargazer.tools import STARGAZER_SUBMISSION_GUIDE, create_tools

    assert_no_grading_language(STARGAZER_SUBMISSION_GUIDE, "Stargazer guide")
    for tool in create_tools(analysis_assistance=False).values():
        assert_no_grading_language(tool.description, f"Stargazer tool {tool.name}")


def test_default_request_preserves_inputs_and_answer_conventions():
    from corral.core.environment import Environment, Toolset
    from corral.core.state import ExecutionState, TaskOutput, TaskState
    from corral.core.task import InputRef, TaskDefinition

    task = TaskDefinition(
        name="Calculate resistance",
        description="Please find the resistance from the supplied measurements.",
        tools=[],
        scoring_fn=lambda _: 0.0,
        submission_format="Return the resistance as a number in ohms.",
        initial_input={"voltage": 12, "work_dir": "private-path"},
        input_map={"current": InputRef("measurement")},
    )
    env = Environment("resistance", task, toolset=Toolset(workspace_factory=None))
    state = ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="request-check",
        branch_id="main",
        task=TaskState(
            dependency_outputs={"measurement": TaskOutput(output={"answer": 2})},
        ),
    )
    prompt = env.get_task_prompt(state)
    assert_unsectioned(prompt, "default prompt")
    assert task.description in prompt
    assert task.submission_format in prompt
    assert "voltage: 12" in prompt
    assert "current (from measurement): 2" in prompt
    assert "private-path" not in prompt
