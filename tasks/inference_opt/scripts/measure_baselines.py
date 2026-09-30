"""Measure zero-shot baselines and write them into task definitions.

Needs a running vLLM server. Writes the baselines artifact, prints the headroom
table, and with ``--write-tasks`` patches the ``baselines`` fields in the task
JSONs, which ship as placeholder zeros until this has been run.

    uv run python scripts/measure_baselines.py \\
        --model student_a=vllm/Qwen/Qwen3.5-9B \\
        --model student_b=vllm/Qwen/Qwen3-8B \\
        --base-url student_a=http://127.0.0.1:8086/v1 \\
        --base-url student_b=http://127.0.0.1:8085/v1 --write-tasks

``--base-url`` takes either one URL shared by every model or ``name=url`` per
model. Models are measured in parallel; ``--runs-per-model`` additionally runs
several benchmark/split evaluations side by side against the same server.

The agent is shown the **train** baseline; the scorer subtracts the **test**
baseline. Showing the test number would leak how hard the held-out split is.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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


def _parse_base_urls(values: list[str], models: dict[str, str]) -> dict[str, str | None]:
    shared: str | None = None
    per_model: dict[str, str] = {}
    for value in values:
        name, sep, url = value.partition("=")
        if sep and not name.startswith("http"):
            if name.strip() not in models:
                raise SystemExit(f"--base-url for unknown model {name!r}")
            per_model[name.strip()] = url.strip()
        else:
            shared = value.strip()
    return {model: per_model.get(model, shared) for model in models}


def _patch_tasks(level: int, measured: dict, specs: dict[str, str]) -> int:
    directory = ROOT / "environments" / f"level_{level}" / "tasks_json"
    paths = sorted(directory.glob("task_*.json"))
    tasks = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    patched = 0
    for task in tasks:
        config = task["initial_input"]
        benchmark = config["benchmark"]
        for model in config["models"]:
            # The baselines are only valid against the deployment they measured.
            if model in specs:
                config.setdefault("model_specs", {})[model] = specs[model]
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
    parser.add_argument("--base-url", action="append", default=[],
                        help="OpenAI-compatible endpoint ending in /v1, either shared "
                             "or name=url per model (repeatable)")
    parser.add_argument("--benchmarks", nargs="*", default=list(datasets.BENCHMARKS))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "inference_opt" / "data" / "baselines" / "v1.json")
    parser.add_argument("--write-tasks", action="store_true")
    parser.add_argument("--max-connections", type=int, default=8,
                        help="concurrent requests per evaluation run")
    parser.add_argument("--runs-per-model", type=int, default=1,
                        help="benchmark/split evaluations run side by side per model")
    args = parser.parse_args()

    models = _parse_models(args.model)
    if not models:
        raise SystemExit("at least one --model name=spec is required")

    base_urls = _parse_base_urls(args.base_url, models)
    manifest = datasets.load_manifest()

    jobs = [
        (model, benchmark, split)
        for model in models
        for benchmark in args.benchmarks
        for split in ("train", "test")
    ]
    slots = {model: threading.Semaphore(max(1, args.runs_per_model)) for model in models}
    print_lock = threading.Lock()

    def run_job(job: tuple[str, str, str]):
        model, benchmark, split = job
        with slots[model]:
            result = measure_baseline(
                benchmark, model, split,
                model_spec=models[model], base_url=base_urls[model],
                max_connections=args.max_connections,
            )
        with print_lock:
            status = "FAILED" if result.error else f"acc={result.accuracy:.3f}"
            print(f"[done] {model:10} {benchmark:14} {split:5} {status}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = dict(zip(jobs, pool.map(run_job, jobs), strict=True))

    failures = [
        f"{model} on {benchmark}/{split}:\n{result.error}"
        for (model, benchmark, split), result in results.items() if result.error
    ]
    if failures:
        raise SystemExit("baseline runs failed:\n\n" + "\n\n".join(failures))

    measured: dict[str, dict] = {}
    rows: list[tuple[str, str, float, float, float, str]] = []
    for model in models:
        measured[model] = {}
        for benchmark in args.benchmarks:
            entry = {
                split: {**results[(model, benchmark, split)].to_dict(),
                        "per_item": results[(model, benchmark, split)].per_item}
                for split in ("train", "test")
            }
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
    print(f"\nwrote {args.out}")

    if args.write_tasks:
        for level in (1, 2):
            count = _patch_tasks(level, measured, models)
            print(f"level {level}: patched {count} baseline entries")


if __name__ == "__main__":
    main()
