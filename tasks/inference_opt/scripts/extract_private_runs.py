"""Unpack the graded logs a finished trial kept out of the agent's reach.

Each dry run, experiment and submission archives its private folder (Inspect
logs with targets, answers, student calls) into the trial's artifact store and
records the reference in the ledger. This writes one folder per run:

    uv run python scripts/extract_private_runs.py \\
        <run-dir>/task-mmlu_pro_a/k-1 out/mmlu_pro_a
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tarfile
import tempfile
from pathlib import Path

from inference_opt.budget import LEDGER_RESOURCE

from corral.persistence import LocalArtifactStore, SQLiteCommitStore


async def _ledger(trial: Path) -> dict:
    metadata = json.loads((trial / "metadata.json").read_text(encoding="utf-8"))
    sandbox = metadata["sandbox"]
    store = SQLiteCommitStore(trial / "commits.sqlite3")
    try:
        state = await store.for_execution(sandbox["execution_id"]).materialize(
            "main", sandbox["final_commit_hash"]
        )
    finally:
        await store.aclose()
    return state.environment.values["resources"][LEDGER_RESOURCE]


async def extract(trial: Path, out: Path) -> list[str]:
    ledger = await _ledger(trial)
    refs = {run["run_id"]: run.get("private_artifact", "") for run in ledger["runs"]}
    if final := ledger.get("final"):
        refs[final["run_id"]] = final.get("private_artifact", "")
    artifacts = LocalArtifactStore(trial / "artifacts")
    written = []
    for run_id, ref in refs.items():
        if not ref:
            continue
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / "run.tar.gz"
            await artifacts.materialize(ref, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(out / run_id, filter="data")
        written.append(run_id)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trial", type=Path, help="a finished trial's k-<n> folder")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    for run_id in asyncio.run(extract(args.trial, args.out)):
        print(args.out / run_id)


if __name__ == "__main__":
    main()
