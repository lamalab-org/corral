"""Measure zero-shot baselines and write them into task definitions.

Needs a running vLLM server. Writes the baselines artifact, prints the headroom
table, and with ``--write-tasks`` patches the ``baselines`` fields in the task
JSONs, which ship as placeholder zeros until this has been run.

    uv run python scripts/measure_baselines.py \\
        --model student_a=Qwen/Qwen2.5-7B-Instruct \\
        --base-url http://127.0.0.1:8000/v1 --write-tasks

The agent is shown the **train** baseline; the scorer subtracts the **test**
baseline. Showing the test number would leak how hard the held-out split is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_opt import datasets  # noqa: E402
from inference_opt.baselines import measure_baseline  # noqa: E402


def _parse_models(pairs: list[str]) -> dict[str, str]:
    models = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--model expects name=served_spec, got {pair!r}")
        name, spec = pair.split("=", 1)
        models[name.strip()] = spec.strip()
    return models


def _patch_tasks(level: int, measured: dict) -> int:
    directory = ROOT / "environments" / f"level_{level}" / "tasks_json"
    paths = sorted(directory.glob("task_*.json"))
    tasks = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    patched = 0
    for task in tasks:
        config = task["initial_input"]
        benchmark = config["benchmark"]
        for model in config["models"]:
            entry = measured.get(model, {}).get(benchmark)
            if not entry:
                continue
            config["baselines"][model] = entry["test"]["accuracy"]
            config["baselines_train"][model] = entry["train"]["accuracy"]
            config.setdefault("baseline_items", {})[model] = entry["train"]["per_item"]
            patched += 1
    for path, task in zip(paths, tasks, strict=True):
        path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[],
                        help="name=served_spec, e.g. student_a=vllm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint, ending in /v1")
    parser.add_argument("--benchmarks", nargs="*", default=list(datasets.BENCHMARKS))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "inference_opt" / "data" / "baselines" / "v1.json")
    parser.add_argument("--write-tasks", action="store_true")
    parser.add_argument("--max-connections", type=int, default=8,
                        help="concurrent requests against the student endpoint")
    args = parser.parse_args()

    models = _parse_models(args.model)
    if not models:
        raise SystemExit("at least one --model name=spec is required")

    manifest = datasets.load_manifest()
    measured: dict[str, dict] = {}
    rows: list[tuple[str, str, float, float, float, str]] = []

    for model, spec in models.items():
        measured[model] = {}
        for benchmark in args.benchmarks:
            entry = {}
            for split in ("train", "test"):
                result = measure_baseline(
                    benchmark, model, split,
                    model_spec=spec, base_url=args.base_url,
                    max_connections=args.max_connections,
                )
                if result.error:
                    raise SystemExit(
                        f"baseline for {model} on {benchmark}/{split} failed:\n{result.error}"
                    )
                entry[split] = {**result.to_dict(), "per_item": result.per_item}
            measured[model][benchmark] = entry

            test = entry["test"]
            rows.append(
                (benchmark, model, test["accuracy"], test["headroom"],
                 test["floor_margin"], test["verdict"])
            )

    # Merge with any prior run's artifact so re-running for one model (e.g. to add
    # student_b after student_a) doesn't drop earlier students' results.
    prior_models: dict[str, str] = {}
    if args.out.exists():
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        prior_models = prior.get("models", {})
        for model, per_benchmark in prior.get("results", {}).items():
            measured.setdefault(model, {}).update(
                {b: e for b, e in per_benchmark.items() if b not in measured.get(model, {})}
            )
    models = {**prior_models, **models}

    print(f"\n{'benchmark':16}{'model':14}{'acc':>7}{'headroom':>10}"
          f"{'floor':>8}  verdict")
    print("-" * 70)
    for benchmark, model, accuracy, headroom, floor, verdict in rows:
        print(
            f"{benchmark:16}{model:14}{accuracy:7.3f}{headroom:10.3f}{floor:8.3f}"
            f"  {verdict}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "dataset_version": manifest.get("dataset_version"),
                "content_fingerprint": manifest.get("content_fingerprint"),
                "policy": {
                    "temperature": 0.0, "seed": 0, "system": None,
                    "shots": 0, "calls_per_question": 1,
                },
                "models": models,
                "results": measured,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out.relative_to(ROOT)}")

    if args.write_tasks:
        for level in (1, 2):
            count = _patch_tasks(level, measured)
            print(f"level {level}: patched {count} baseline entries")


if __name__ == "__main__":
    main()
