"""Work-directory hygiene (Environment Concurrency Isolation, Phase 4).

Independent of the process-worker backend: concurrent trials each own an
isolated workspace, but *scoring* used to resolve a submitted filename against
the process-global `CORRAL_WORK_DIR`. Two trials that both submitted the same
bare filename would then resolve to whichever file was written most recently —
one trial scored against another trial's output.

These tests pin the fix: the fallback file search is scoped to the trial's own
workspace, so same-named submissions land in — and are scored from — distinct
directories.
"""

import json
from pathlib import Path

from corral.backend.env import Environment, Toolset, build_environments
from corral.backend.task import TaskDefinition
from corral.utils.tool_helpers import smart_resolve_path


def test_smart_resolve_path_scopes_the_search_to_base_dir(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "trial_output.json").write_text("A")
    (dir_b / "trial_output.json").write_text("B")

    # The same bare filename resolves within whichever base dir is requested,
    # never leaking across to the sibling directory.
    resolved_a = smart_resolve_path("trial_output.json", base_dir=str(dir_a))
    resolved_b = smart_resolve_path("trial_output.json", base_dir=str(dir_b))

    assert Path(resolved_a).read_text() == "A"
    assert Path(resolved_b).read_text() == "B"


def _score_from_file(path: str) -> float:
    """Scoring fn: return the number held in the resolved answer file."""
    return float(json.loads(Path(path).read_text())["value"])


def _writer_template(base_work_dir: str) -> Environment:
    task = TaskDefinition(
        name="writer",
        description="write a value to answer.json",
        tools=[],
        scoring_fn=_score_from_file,
        submission_format={},
    )
    envs = build_environments(
        {"writer": task},
        base_work_dir=base_work_dir,
        toolset=Toolset(workspace_factory=None),
    )
    return envs["writer"]


def test_concurrent_trials_score_against_their_own_workspace(tmp_path, monkeypatch):
    template = _writer_template(str(tmp_path))

    r1 = template.for_trial("tr_1")
    r2 = template.for_trial("tr_2")

    # Distinct, id-namespaced workspaces under the one shared base dir.
    assert r1.current_work_dir != r2.current_work_dir

    # Both trials write the *same filename* with different content.
    (Path(r1.current_work_dir) / "answer.json").write_text(json.dumps({"value": 1}))
    (Path(r2.current_work_dir) / "answer.json").write_text(json.dumps({"value": 2}))

    # Simulate the legacy process-global work dir pointing at the shared base:
    # under the old resolution this is exactly what made the two trials collide
    # (rglob over the shared base returned the most recently written file).
    monkeypatch.setenv("CORRAL_WORK_DIR", str(tmp_path))

    r1.state.submitted_answer = "answer.json"
    r2.state.submitted_answer = "answer.json"

    # Each trial resolves to its OWN file, not "the most recent" in the shared base.
    assert r1.score() == 1.0
    assert r2.score() == 2.0
