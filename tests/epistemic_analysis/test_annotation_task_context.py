"""Rendered task instructions retain public evidence and pinned source provenance."""

import hashlib
import json
from pathlib import Path

import pytest

from corral.epistemic_analysis.annotation_transcript import annotation_transcript
from corral.epistemic_analysis.decision_applicability import (
    ValidationPolicy,
    extract_facts,
)
from corral.epistemic_analysis.decision_units import full_fragments
from corral.epistemic_analysis.sources.inputs import CheckInput
from corral.epistemic_analysis.storage import atomic_save
from corral.epistemic_analysis.task_prompt_context import task_prompt
from corral.epistemic_analysis.trace_checks import CheckPipeline

ROOT = Path(__file__).resolve().parents[2]
CORE_ENVIRONMENT = ROOT / "src/corral/core/environment.py"


def reconstruction_config(tmp_path, agent):
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    environment = tmp_path / "tasks/retrosynthesis"
    environment.mkdir(parents=True)
    (environment / "retrosynthesis").symlink_to(
        ROOT / "tasks/retrosynthesis/retrosynthesis", target_is_directory=True
    )
    directory = environment / "environments/level_1/tasks_json"
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(
        json.dumps(
            [
                {
                    "id": "example",
                    "input": {"prompt": "Plan the public synthesis."},
                    "submission_format": "Return a route.",
                    "output": {"target": "HIDDEN_TARGET_CANARY"},
                }
            ]
        )
    )
    return {
        "agent": agent,
        "source_root": str(tmp_path),
        "revision": "fixture-source",
        "environment": "retrosynthesis",
        "level": "1",
        "task_id": "example",
        "runtime_options": {"api_key": "PRIVATE_CONFIG_CANARY"},
        "enable_surrender": False,
    }


def test_annotation_recovers_react_context_without_constructing_agents(
    tmp_path, monkeypatch
):
    from corral.agents.base_agent import BaseAgent
    from corral.epistemic_analysis.schema import SourceReference
    from corral.epistemic_analysis.sources.files import normalize_document

    configuration = reconstruction_config(tmp_path, "react")
    monkeypatch.setattr(
        BaseAgent,
        "__init__",
        lambda *_a, **_k: pytest.fail("Must not construct agents"),
    )
    original = {
        "messages": [{"role": "assistant", "content": "Inspect public inputs."}]
    }
    inputs = CheckInput.from_document(
        original, {"prompt_reconstruction": configuration}
    )
    inputs.require_prompt_context()
    assert [m["role"] for m in inputs.trace.messages] == [
        "system",
        "user",
        "user",
        "assistant",
    ]
    assert "Plan the public synthesis." in inputs.trace.messages[1]["content"]
    assert inputs.trace.message_sources[3]["index"] == 0
    assert len(original["messages"]) == 1
    assert "PRIVATE_CONFIG_CANARY" not in json.dumps(inputs.public_state())
    assert "HIDDEN_TARGET_CANARY" not in json.dumps(inputs.public_state())
    graph_trace = normalize_document(
        {**original, "prompt_reconstruction": configuration},
        SourceReference("trace", "test", "hash"),
    )
    assert graph_trace.messages == inputs.trace.messages


def test_sdk_reconstruction_preserves_recorded_user_and_marks_hidden_system(tmp_path):
    configuration = reconstruction_config(tmp_path, "codex")
    original = {
        "messages": [
            {
                "role": "user",
                "content": "\nExact recorded request, including resumed history.\n",
            },
            {"role": "assistant", "content": "Working."},
        ]
    }
    inputs = CheckInput.from_document(
        original, {"prompt_reconstruction": configuration}
    )
    inputs.require_prompt_context()
    assert [m["role"] for m in inputs.trace.messages] == [
        "developer",
        "user",
        "assistant",
    ]
    assert inputs.trace.messages[1:] == original["messages"]
    assert inputs.trace.message_sources[1]["index"] == 0
    assert "SDK-private" in inputs.public_state()["capture"]["limitations"][0]
    configuration["system_prompt_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="recorded hash"):
        CheckInput.from_document(original, {"prompt_reconstruction": configuration})


def test_recorded_worker_prompts_need_no_reconstruction_and_keep_boundaries():
    original = {
        "messages": [
            {"role": "system", "content": "\nPlan experiments.\n"},
            {"role": "user", "content": "Planner input."},
            {"role": "assistant", "content": "Plan."},
            {"role": "system", "content": "Evaluate experiments."},
            {"role": "user", "content": "Critic input."},
            {"role": "assistant", "content": "Critique."},
        ]
    }
    inputs = CheckInput.from_document(
        original, {"prompt_reconstruction": {"agent": "ai-scientist"}}
    )
    assert inputs.trace.messages == original["messages"]


def test_resistor_prompt_uses_public_fields_and_core_renderer(tmp_path):
    from corral.epistemic_analysis.task_prompt_context import task_prompt

    root = Path(__file__).resolve().parents[2]
    tasks = tmp_path / "tasks"
    directory = tasks / "resistor_network/environments/level_1/tasks_json"
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(
        json.dumps(
            [
                {
                    "id": "example",
                    "description": "Measure the network resistance.",
                    "submission_format": "Return a number in ohms.",
                    "initial_input": {
                        "network": "public-network",
                        "work_dir": "/workspace",
                    },
                    "output": {"target": "HIDDEN_ANSWER"},
                    "scoring_params": {"truth": "HIDDEN_TRUTH"},
                }
            ]
        )
    )
    renderer = tasks / "resistor_network/src/resistor_network/env.py"
    renderer.parent.mkdir(parents=True)
    renderer.write_text("# This environment inherits the core prompt renderer.\n")
    prompt, evidence = task_prompt(
        tmp_path,
        {"environment": "resistor_network", "level": "1", "task_id": "example"},
        tasks,
        root / "src/corral/core/environment.py",
    )
    assert "public-network" in prompt
    assert "Return a number in ohms." in prompt
    assert "HIDDEN_" not in prompt
    assert len(evidence) == 3


@pytest.fixture
def source_run(tmp_path):
    source = tmp_path / "source" / "review"
    task = {
        "id": "task09",
        "description": "Evaluate the supplied item bank.",
        "submission_format": "Submit JSON.",
        "initial_input": {
            "public_inputs": ["data.csv", "holdout.csv"],
            "secret": "HIDDEN_INPUT",
        },
        "input": {"solution": "HIDDEN_COMPOSITION"},
        "output": "HIDDEN_ANSWER",
        "scoring_params": {"truth": "HIDDEN_TRUTH"},
    }
    path = "configuration-source/revision/tasks/psychometrics/environments/level_2/tasks_json/task_09.json"
    atomic_save(source.parent / path, [task])
    ref = {
        "path": path,
        "git_commit": "revision",
        "sha256": hashlib.sha256((source.parent / path).read_bytes()).hexdigest(),
    }
    renderer_path = "configuration-source/revision/tasks/psychometrics/src/corral_psychometrics/env.py"
    renderer = source.parent / renderer_path
    renderer.parent.mkdir(parents=True, exist_ok=True)
    renderer.write_text(
        (ROOT / "tasks/psychometrics/src/corral_psychometrics/env.py").read_text()
    )
    renderer_ref = {
        "path": renderer_path,
        "git_commit": "revision",
        "sha256": hashlib.sha256(renderer.read_bytes()).hexdigest(),
    }
    row = {
        "id": "example",
        "environment": "psychometrics",
        "level": "2",
        "task_id": "task09",
        "agent": "react",
        "model": "kimi",
        "pr_head_sha": "revision",
        "validation_tool_inventory": {"source_evidence": [ref, renderer_ref]},
    }
    return source, row


def test_rendered_prompt_reaches_current_checks_and_source_fragments(
    tmp_path, source_run
):
    source, row = source_run
    prompt, refs = task_prompt(source, row, ROOT / "tasks", CORE_ENVIRONMENT)
    assert "HIDDEN_" not in prompt
    assert "`holdout.csv`" in prompt
    assert "Before you answer" in prompt
    document = CheckInput.from_document(
        {"messages": [{"role": "assistant", "content": "I inspect the supplied data."}]}
    ).snapshot()
    prepared = annotation_transcript(
        document,
        "react",
        prompt,
        refs,
        initial_messages=[
            {"role": "system", "content": "Exact agent instructions"},
            {"role": "user", "content": prompt},
        ],
        initial_sources=[
            {"origin": "recorded"},
            {"origin": "reconstructed_task_prompt", "source_evidence": refs},
        ],
    )
    inputs = CheckInput.from_document(prepared)
    assert inputs.trace.messages[2:] == document["messages"]
    assert inputs.trace.message_sources[1]["origin"] == "reconstructed_task_prompt"
    assert (
        inputs.trace.message_sources[1]["source_evidence"][0]["git_commit"]
        == "revision"
    )
    fragments = full_fragments(inputs.trace)
    pipeline = CheckPipeline(tmp_path / "annotations", decision=None)
    state = pipeline.state(
        inputs, extract_facts(inputs, ValidationPolicy()), fragments[1:], partial=True
    )
    assert state["initial_instructions"][1]["content"] == prompt
    assert fragments[1]["msg_idx"] == 1
    assert fragments[1]["content"] == prompt
    assert (fragments[1]["start"], fragments[1]["end"]) == (0, len(prompt))


def test_local_tasks_tree_fallback_renders_prompt_and_missing_task_fails(source_run):
    source, row = source_run
    tasks_root = source.parent / "configuration-source/revision/tasks"
    row.pop("validation_tool_inventory")
    prompt, _ = task_prompt(source, row, tasks_root, CORE_ENVIRONMENT)
    assert "`holdout.csv`" in prompt
    row["task_id"] = "does_not_exist"
    with pytest.raises(ValueError, match="Expected one definition"):
        task_prompt(source, row, tasks_root, CORE_ENVIRONMENT)


def test_changed_source_configuration_is_rejected_before_rendering(source_run):
    source, row = source_run
    path = (
        source.parent / row["validation_tool_inventory"]["source_evidence"][0]["path"]
    )
    path.write_text("[]")
    with pytest.raises(ValueError, match="configuration has changed"):
        task_prompt(source, row, ROOT / "tasks", CORE_ENVIRONMENT)


@pytest.mark.parametrize(
    "environment", ["psychometrics", "wetlab", "spectra_elucidation", "corral_md"]
)
def test_renderers_follow_visible_task_fields_without_loading_hidden_answers(
    tmp_path, environment
):
    from corral.epistemic_analysis.task_prompt_context import ENVIRONMENT_FILES

    root = ROOT
    task = {
        "id": "example_task_6",
        "description": "DESCRIPTION_FIELD",
        "input": {"prompt": "VISIBLE_INPUT_PROMPT", "samples": ["HIDDEN_SAMPLE"]},
        "output": {"target": "HIDDEN_ANSWER"},
        "scoring_params": {"truth": "HIDDEN_TRUTH"},
        "submission_format": "Return model_syntax.",
        "initial_input": {"public_inputs": ["holdout.csv"], "work_dir": "/workspace"},
    }
    task_dir = tmp_path / environment / "environments/level_1/tasks_json"
    atomic_save(task_dir / "task.json", [task])
    renderer_path = tmp_path / environment / ENVIRONMENT_FILES[environment]
    renderer_path.parent.mkdir(parents=True, exist_ok=True)
    renderer_path.write_text(
        (root / "tasks" / environment / ENVIRONMENT_FILES[environment]).read_text()
    )
    row = {"environment": environment, "level": "1", "task_id": task["id"]}
    prompt, _ = task_prompt(
        tmp_path, row, tmp_path, root / "src/corral/core/environment.py"
    )
    assert "HIDDEN_" not in prompt
    assert "holdout.csv" in prompt
    assert "Return model_syntax." in prompt
    if environment in {"wetlab", "spectra_elucidation"}:
        assert "VISIBLE_INPUT_PROMPT" in prompt
        assert "DESCRIPTION_FIELD" not in prompt
    else:
        assert "DESCRIPTION_FIELD" in prompt
    if environment == "psychometrics":
        assert "semopy" in prompt
    if environment == "corral_md":
        assert "immutable declared resources" in prompt
        assert "Record Step, Temperature, Pressure, and Density" not in prompt
    task["input"]["input_from_task"] = ["previous_task"]
    atomic_save(task_dir / "task.json", [task])
    with pytest.raises(ValueError, match="Chained task prompts"):
        task_prompt(tmp_path, row, tmp_path, root / "src/corral/core/environment.py")
