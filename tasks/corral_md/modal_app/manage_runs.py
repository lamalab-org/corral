"""Close completed MD runs and prune closed runs after their recovery window."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import PurePosixPath

import modal

RETENTION_DAYS = 30
ROOT = PurePosixPath("/corral/runs")
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _id(value: str) -> str:
    if not RUN_ID.fullmatch(value):
        raise ValueError(f"Invalid run ID: {value!r}")
    return value


def _read(volume: modal.Volume, run_id: str) -> dict:
    remote = ROOT / _id(run_id) / "run.json"
    result = json.loads(b"".join(volume.read_file(str(remote))))
    if result.get("schema") != 1 or not result.get("release_id"):
        raise ValueError(f"Invalid MD run manifest: {remote}")
    return result


def close_run(volume: modal.Volume, run_id: str, status: str) -> None:
    """Mark an externally confirmed completion or cancellation."""
    if status not in {"completed", "cancelled"}:
        raise ValueError("Run status must be completed or cancelled")
    state = _read(volume, run_id)
    if state.get("closed_at"):
        if state.get("status") != status:
            raise ValueError("Run was closed with a different status")
        return
    state["status"] = status
    state["closed_at"] = datetime.now(timezone.utc).isoformat()
    payload = (json.dumps(state, sort_keys=True, indent=2) + "\n").encode()
    with volume.batch_upload(force=True) as upload:
        upload.put_file(BytesIO(payload), str(ROOT / run_id / "run.json"))


def prune_closed_runs(volume: modal.Volume, *, execute: bool = False) -> list[str]:
    """List or remove runs closed at least 30 days ago."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    eligible: list[str] = []
    for entry in volume.listdir(str(ROOT)):
        path = PurePosixPath("/" + str(entry.path).lstrip("/"))
        if path.parent != ROOT or not RUN_ID.fullmatch(path.name):
            continue
        state = _read(volume, path.name)
        closed = state.get("closed_at")
        if state.get("status") not in {"completed", "cancelled"} or not closed:
            continue
        closed_at = datetime.fromisoformat(closed)
        if closed_at.tzinfo is None:
            raise ValueError(f"Run {path.name} has a naive closed_at timestamp")
        if closed_at <= cutoff:
            eligible.append(path.name)
    if execute:
        for run_id in eligible:
            volume.remove_file(str(ROOT / run_id), recursive=True)
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    close = subcommands.add_parser("close")
    close.add_argument("run_id")
    close.add_argument("status", choices=("completed", "cancelled"))
    prune = subcommands.add_parser("prune")
    prune.add_argument("--execute", action="store_true", help="Delete eligible runs")
    args = parser.parse_args()
    volume = modal.Volume.from_name(os.getenv("CORRAL_MD_MODAL_VOLUME", "simulations"))
    if args.command == "close":
        close_run(volume, args.run_id, args.status)
    else:
        for run_id in prune_closed_runs(volume, execute=args.execute):
            print(run_id)  # noqa: T201


if __name__ == "__main__":
    main()
