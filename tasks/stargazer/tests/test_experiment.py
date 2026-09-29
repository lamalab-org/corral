import json
import sqlite3
from datetime import datetime, timezone

import pytest
from stargazer.audit import DEFAULT_DATA_ROOT, reference_submission
from stargazer.env import create_environments
from stargazer.experiment import bank_identity, compare_run, main

from corral.agents.schema import AgentOutcome
from corral.core import Action
from corral.persistence import ShardedCommitStore, SQLiteCommitStore
from corral.runtime import TaskRuntime


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("initialized", [False, True])
async def test_offline_report_grades_only_final_answer_and_keeps_unfinished(
    tmp_path, initialized
):
    env = create_environments(work_dir=tmp_path / "work", development_mode=True)[
        "seed15_diff5"
    ]
    task = env.current_task.scoring_inputs["benchmark_task"]
    reference = reference_submission(
        task,
        json.loads((DEFAULT_DATA_ROOT / "synthetic/seed15_diff5.json").read_text()),
    ).canonical_payload()

    class Driver:
        def __init__(self, finish):
            self.finish = finish

        async def run_session(self, session):
            await session.execute(Action(name="validate_fit", arguments=reference))
            if self.finish:
                await session.execute(
                    Action(name="submit_answer", arguments={"answer": '{"planets":[]}'})
                )
                return AgentOutcome(status="completed", answer='{"planets":[]}')
            return AgentOutcome(
                status="iteration_limit", error="iteration budget exhausted"
            )

    for index, finish in enumerate((True, False)):
        directory = tmp_path / "runs" / str(index)
        directory.mkdir(parents=True)
        async with SQLiteCommitStore(directory / "commits.sqlite3") as store:
            await TaskRuntime(store).run(
                Driver(finish),
                env.for_task(str(index)),
                execution_id=str(index),
                started_at=datetime.now(timezone.utc),
                max_iterations=1,
            )
    # A failed startup leaves a shard even when no events were committed.
    shards = ShardedCommitStore(tmp_path / "runs", benchmark_run_id="benchmark")
    failed_id = "benchmark:seed15_diff5:2"
    directory = shards.execution_dir(failed_id)
    if initialized:
        async with SQLiteCommitStore(directory / "commits.sqlite3", failed_id):
            pass
    await shards.aclose()
    report = compare_run(tmp_path / "runs", DEFAULT_DATA_ROOT)
    assert report["total"] == 3
    assert report["submitted"] == 1
    assert report["unfinished_or_surrendered"] == 2
    assert report["completion_rate"] == pytest.approx(1 / 3)
    assert report["passes"] == {"legacy": 0, "complete": 0}
    assert sum(row["validation_calls"] for row in report["records"]) == 2
    assert report["provenance"]["source_hash"]
    assert report["bank_hash"] == bank_identity(DEFAULT_DATA_ROOT)
    # Corral records iteration exhaustion as execution.failed.
    assert report["errors"] == 2
    assert report["termination_counts"]["iteration_limit"] == 1
    assert report["termination_counts"]["no_commits"] == 1
    failed = next(row for row in report["records"] if row["execution_id"] == failed_id)
    assert failed["task_id"] == "seed15_diff5"
    assert failed["status"] == "unfinished"
    assert failed["error"] == "Execution has no committed events"
    submitted = next(row for row in report["records"] if row["status"] == "submitted")
    assert submitted["missed_planets"] == len(task.truth_planets)
    assert submitted["execution_provenance"]["bank_hash"] == report["bank_hash"]


def test_offline_report_does_not_hide_database_corruption(tmp_path):
    (tmp_path / "commits.sqlite3").write_bytes(b"not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError, match="not a database"):
        compare_run(tmp_path, DEFAULT_DATA_ROOT)


def test_prepare_records_bank_and_primary_scorer(tmp_path, monkeypatch):
    output = tmp_path / "run.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment",
            "prepare",
            "--data-root",
            str(DEFAULT_DATA_ROOT),
            "--output",
            str(output),
            "--model",
            "test-model",
            "--image-digest",
            "sha256:test",
            "--scorer",
            "legacy",
        ],
    )
    main()
    manifest = json.loads(output.read_text())
    assert manifest["bank_hash"] == bank_identity(DEFAULT_DATA_ROOT)
    assert manifest["primary_scorer"] == "legacy"
    # Preparation includes official tasks and the retained historical bank.
    assert set(manifest["task_hashes"]) == {
        path.stem for path in (DEFAULT_DATA_ROOT / "synthetic").glob("*.json")
    }


def test_prepared_arms_match_execution_fingerprints_and_effective_data(
    tmp_path, monkeypatch
):
    import hashlib
    from dataclasses import asdict

    from stargazer.env import DATA_ROOT
    from stargazer.models import load_task

    output = tmp_path / "ab.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment",
            "prepare",
            "--data-root",
            str(DATA_ROOT),
            "--output",
            str(output),
            "--model",
            "test",
            "--image-digest",
            "sha256:test",
        ],
    )
    main()
    manifest = json.loads(output.read_text())
    assert manifest["resources"]["concurrency"] == 2
    assert set(manifest["effective_observation_hashes"]) == {
        path.stem for path in (DATA_ROOT / "synthetic").glob("*.json")
    }
    assert manifest["launch_order"] == sorted(manifest["task_hashes"])
    for arm, assisted in [("A", False), ("B", True)]:
        env = create_environments(
            work_dir=tmp_path / arm, analysis_assistance=assisted
        )["seed15_diff5"]
        assert (
            env.current_task.execution_version
            == manifest["arms"][arm]["execution_version"]
        )
    task = load_task(DATA_ROOT / "synthetic/seed15_diff5.json")
    expected = hashlib.sha256(
        json.dumps(
            asdict(task.public_fit_context()),
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert manifest["effective_observation_hashes"][task.task_id] == expected
