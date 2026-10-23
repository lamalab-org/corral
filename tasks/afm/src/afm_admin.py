"""Operator-only health/recovery and an explicitly requested hardware smoke run."""

# ruff: noqa: T201 -- this module is a human-facing CLI.
import argparse
import json
from pathlib import Path

from afm_runtime import Config, Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "recover", "smoke"))
    parser.add_argument(
        "--take-control",
        action="store_true",
        help="Acknowledge stopping inherited/manual scans",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        help="JSON object of previously validated AFM task-style parameters",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if args.command != "check" and not args.take_control:
        parser.error("recover/smoke require --take-control; these stop existing scans")
    if args.command == "smoke":
        if (
            not args.params_json
            or not args.output_dir
            or args.repeats < 1
            or args.frames < 1
        ):
            parser.error(
                "smoke requires --params-json, --output-dir, and positive repeats/frames"
            )
        params = json.loads(args.params_json.read_text(encoding="utf-8"))
        if not isinstance(params, dict) or not params:
            parser.error("--params-json must contain a nonempty settings object")
        # An empty new directory avoids confusing old files with this experiment.
        args.output_dir.mkdir(parents=True, exist_ok=False)
    worker = Worker(Config.from_env(), recovery=args.command in ("check", "recover"))
    try:
        if args.command == "recover":
            print(json.dumps(worker.recover(), indent=2))
        elif args.command == "check":
            worker.call("connect")
            print(json.dumps(worker.call("health"), indent=2))
            if worker.journal.marker.exists():
                print("Still quarantined. check does not clear the marker.")
        else:
            for trial in range(args.repeats):
                workspace = args.output_dir.resolve() / f"trial-{trial + 1}"
                workspace.mkdir()
                worker.call("begin", params, workspace, workspace.name)
                try:
                    for _ in range(args.frames):
                        print(worker.call("acquire"))
                finally:
                    # A quarantined worker rejects further hardware commands.
                    if not worker.poison.is_set():
                        worker.call("finish")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
