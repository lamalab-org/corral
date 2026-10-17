"""Measure zero-shot baselines on the frozen question set.

A baseline is the pinned zero-shot policy in ``inference_opt/baselines.py``: one
call per question at temperature 0, seed 0 and at most 8192 tokens, with
Inspect's MATH prompt for MATH and GSM-Symbolic's prompt (without its worked
examples) for GSM-Symbolic, graded by the same scorer as every policy run. The
same command measures students and the larger models they are compared with.

Train and test are measured together and written to
``inference_opt/data/baselines/<version>.json``, merged with the results already
there. ``scripts/generate_tasks.py`` reads task baselines and pass bars from
that file. The agent is shown train scores; the scorer uses test scores.

``--splits examples`` records each student's reply to the example questions in
``inference_opt/data/baselines/examples_<version>.jsonl``, shown to the agent
when it reveals them. Only students need it. Rows for other students and
benchmarks are kept.

    export OPENROUTER_API_KEY=...
    uv run python scripts/measure_baselines.py \\
        --model student_b=openrouter/meta-llama/llama-3.1-8b-instruct@coreweave \\
        --model reference_b=openrouter/meta-llama/llama-3.1-70b-instruct@amazon-bedrock \\
        --runs-per-model 2
    uv run python scripts/measure_baselines.py --model student_b=... --splits examples

``--no-reasoning NAME`` turns a reasoning model's thinking off on every call, as
the tasks do for Qwen 3.5. ``--benchmarks`` limits the run to some benchmarks.
A vLLM model is given as ``name=vllm/<served model>`` with
``--base-url name=http://host:port/v1``. Models are measured in parallel;
``--runs-per-model`` sets how many benchmark/split evaluations run side by side
for one model.
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


def _write_example_baselines(
    results: dict, models: dict[str, str], options: dict[str, dict]
) -> Path:
    """Replace the rows of each measured model and benchmark; keep every other row."""
    path = datasets.example_baselines_path()
    measured = {
        (model, benchmark)
        for (model, benchmark, split) in results
        if split == "examples"
    }
    kept = []
    if path.is_file():
        kept = [
            row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if (row["model"], row["item_id"].split(":", 1)[0]) not in measured
        ]
    rows = [
        {
            "model": model,
            "model_spec": models[model],
            "student_options": options[model],
            "item_id": item_id,
            "correct": correct,
            **result.completions.get(
                item_id, {"completion": "", "reasoning": "", "finish_reason": ""}
            ),
        }
        for (model, _benchmark, split), result in results.items()
        if split == "examples"
        for item_id, correct in result.per_item.items()
    ]
    rows = sorted(kept + rows, key=lambda row: (row["model"], row["item_id"]))
    datasets.write_jsonl(path, rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[],
                        help="name=served_spec, e.g. student_a=vllm/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--no-reasoning", action="append", default=[], metavar="NAME",
                        help="turn this student's thinking off on every call (repeatable)")
    parser.add_argument("--base-url", action="append", default=[],
                        help="OpenAI-compatible endpoint ending in /v1, either shared "
                             "or name=url per model (repeatable)")
    parser.add_argument("--benchmarks", nargs="*", default=list(datasets.BENCHMARKS))
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=["train", "test", "examples"],
                        help="train and test are measured together; examples on its own")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "inference_opt" / "data" / "baselines"
                        / f"{datasets.DATASET_VERSION}.json")
    parser.add_argument("--max-connections", type=int, default=8,
                        help="concurrent requests per evaluation run")
    parser.add_argument("--runs-per-model", type=int, default=1,
                        help="benchmark/split evaluations run side by side per model")
    args = parser.parse_args()

    models = _parse_models(args.model)
    unknown = sorted(set(args.no_reasoning) - set(models))
    if unknown:
        raise SystemExit(f"--no-reasoning for unknown model(s): {unknown}")
    options = {m: ({"reasoning": False} if m in args.no_reasoning else {}) for m in models}
    if not models:
        raise SystemExit("at least one --model name=spec is required")
    scored = [split for split in args.splits if split != "examples"]
    if scored and sorted(set(scored)) != ["test", "train"]:
        raise SystemExit("--splits: measure train and test together")

    base_urls = _parse_base_urls(args.base_url, models)
    manifest = datasets.load_manifest()

    jobs = [
        (model, benchmark, split)
        for model in models
        for benchmark in args.benchmarks
        for split in args.splits
    ]
    slots = {model: threading.Semaphore(max(1, args.runs_per_model)) for model in models}
    print_lock = threading.Lock()

    def run_job(job: tuple[str, str, str]):
        model, benchmark, split = job
        with slots[model]:
            result = measure_baseline(
                benchmark, model, split,
                model_spec=models[model], base_url=base_urls[model],
                options=options[model],
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

    if "examples" in args.splits:
        print(f"\nwrote {_write_example_baselines(results, models, options)}")
    if not scored:
        return

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
    prior_options: dict[str, dict] = {}
    if args.out.exists():
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        prior_models = prior.get("models", {})
        prior_options = prior.get("student_options", {})
        for model, per_benchmark in prior.get("results", {}).items():
            measured.setdefault(model, {}).update(
                {b: e for b, e in per_benchmark.items() if b not in measured.get(model, {})}
            )
    models = {**prior_models, **models}
    options = {**prior_options, **options}

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
                "student_options": options,
                "results": measured,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
