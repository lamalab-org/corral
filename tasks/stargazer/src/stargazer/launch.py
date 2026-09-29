"""Launch the pinned Level 1 recovery comparison inside a prepared Docker image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from stargazer.bank import verify_bank
from stargazer.env import create_environments
from stargazer.experiment import bank_identity, provenance
from stargazer.identifiability import AUDIT_VERSION, BankRules
from stargazer.protocol import ASSISTED_PROTOCOL
from stargazer.score import EvaluationCriteria

DATA_ROOT = Path("/corral-private/0")


def verify_launch(config: dict, data_root: Path = DATA_ROOT) -> dict:
    """Resolve and check the actual execution definitions before any model calls."""
    manifest = verify_bank(data_root, purpose="evaluation")
    if manifest.get("audit_version") != AUDIT_VERSION:
        raise ValueError("Expected the strengthened recovery audit")
    for key in ("bank_hash", "calibration_hash"):
        if manifest[key] != config[key]:
            raise ValueError(f"Launch {key} mismatch")
    criteria = asdict(EvaluationCriteria())
    if config["criteria"] != criteria or manifest.get("criteria") != criteria:
        raise ValueError("Launch scorer criteria mismatch")
    if asdict(BankRules(**manifest["rules"]).criteria()) != criteria:
        raise ValueError("Recovery audit and launch criteria differ")
    if config["protocol"] != ASSISTED_PROTOCOL:
        raise ValueError("Launch protocol mismatch")
    if config["level"] != 1 or config["task_count"] != 10:
        raise ValueError("This comparison requires ten Level 1 tasks")
    identity = bank_identity(data_root)
    environments = create_environments(
        level=1,
        data_root=data_root,
        work_dir="/workspace/workspaces/level-1",
        development_mode=False,
        scorer="complete",
        analysis_assistance=True,
    )
    membership = manifest["membership"]["1"]
    if (
        len(environments) != 10
        or len(membership) != 10
        or set(environments) != set(membership)
    ):
        raise ValueError("Launch task membership mismatch")
    for environment in environments.values():
        task = environment.current_task
        if (
            task.scoring_fn.bank_hash != identity
            or task.scoring_fn.criteria != EvaluationCriteria()
        ):
            raise ValueError("Execution bank or scorer mismatch")
        if not task.execution_version.startswith(config["protocol"] + ":"):
            raise ValueError("Execution protocol mismatch")
    return {
        "bank_hash": identity,
        "calibration_hash": manifest["calibration_hash"],
        "audit_version": AUDIT_VERSION,
        "protocol": config["protocol"],
        "criteria": criteria,
        "task_count": len(environments),
        "task_ids": sorted(environments),
        "provenance": provenance(),
        "execution_version": next(
            iter(environments.values())
        ).current_task.execution_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-id", default="stargazer-recovery-v2-l1-sol-medium-k1")
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        parser.error(
            "Run this launcher inside Docker with a private read-only bank mount"
        )
    if not os.statvfs(DATA_ROOT).f_flag & os.ST_RDONLY:
        parser.error("/corral-private/0 must be mounted read-only")
    summary = verify_launch(json.loads(args.config.read_text()))
    print(json.dumps(summary, indent=2), flush=True)  # noqa: T201 - CLI summary
    if args.check_only:
        return 0
    if Path("/workspace/commits.sqlite3").exists() or Path("/workspace/runs").exists():
        parser.error("Use a fresh workspace volume for this comparison")
    from corral.runtime import permissions

    checkpoints = Path("/workspace/control/checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True, mode=0o700)
    checkpoints.chmod(0o700)
    permissions.configure(checkpoints)
    os.environ.update(CORRAL_LANGFUSE_ENABLED="true", LANGFUSE_TRACING_ENABLED="true")
    # Reuse the existing runner, within this container's controller/worker boundary.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "run_scripts"))
    from run_tool_calling import main as run

    return run(
        [
            "--environment",
            "stargazer",
            "--env-kwargs",
            json.dumps(
                {
                    "level": 1,
                    "data_root": str(DATA_ROOT),
                    "development_mode": False,
                    "work_dir": "/workspace/workspaces/level-1",
                    "scorer": "complete",
                    "analysis_assistance": True,
                }
            ),
            "--model",
            "openai/gpt-5.6-sol",
            "--agent-kwargs",
            '{"reasoning_effort":"medium","additional_drop_params":["temperature"]}',
            "--trials",
            "1",
            "--max-parallel",
            "5",
            "--max-parallel-total",
            "5",
            "--max-parallel-per-task",
            "1",
            "--max-iterations",
            "50",
            "--max-attempts",
            "1",
            "--verbose",
            "--sandbox",
            "local",
            "--run-id",
            args.run_id,
            "--commit-file",
            "/workspace/commits.sqlite3",
            "--report",
            "/workspace/report.json",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
