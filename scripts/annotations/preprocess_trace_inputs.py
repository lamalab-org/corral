"""Prepare overflow-only tool excerpts and audit native context fit, without inference.

Scope: a frozen annotation manifest plus current indexed retrosynthesis reports.
Original reports/inputs and existing labels are preserved. Resume reuses only
records with matching source, rubric, tokenizer, and preprocessing identities.
"""

from __future__ import annotations

# Standalone CLI bootstraps the repository and prints audit progress.
# ruff: noqa: E402, T201
import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import lzma
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Run snapshots may place this file directly under scripts/.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "annotations" else SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from corral.epistemic_analysis.annotation_transcript import annotation_transcript
from corral.epistemic_analysis.annotators.local import KnowLineClient, ProviderConfig
from corral.epistemic_analysis.decision_applicability import (
    ValidationPolicy,
    extract_facts,
)
from corral.epistemic_analysis.overflow import OVERFLOW_VERSION, prepare_overflow
from corral.epistemic_analysis.schema import content_hash
from corral.epistemic_analysis.scientist_units import (
    SCIENTIST_UNIT_VERSION,
    is_scientist,
    scientist_units,
)
from corral.epistemic_analysis.sources.inputs import CheckInput
from corral.epistemic_analysis.storage import atomic_save
from corral.epistemic_analysis.task_prompt_context import task_prompt
from corral.epistemic_analysis.trace_checks import CheckPipeline, check_definitions


class NativeCounter:
    """Use the pinned official renderer and tokenizer, with exact prefix reuse."""

    def __init__(self, model_path, config):
        from transformers import AutoTokenizer

        sys.path.insert(0, str(model_path))
        import knowline_layout

        self.layout = knowline_layout
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.config = config
        self.last_state_object = None
        self.scaffold = None
        self.prefix_cache = {}
        self.specials = set(self.tokenizer.all_special_tokens)

    body = KnowLineClient.body

    def exact_length(self, prompt):
        """A recognized special token is a hard tokenizer boundary.

        Requests for the same state differ only after the question marker. Cache
        its long prefix through the final special token; tokenize the remainder
        exactly. This avoids re-tokenizing the same multi-million-character CSV
        twelve times. It is not a character/token approximation.
        """
        prefix = self.scaffold[0]
        best = max(
            (prefix.rfind(token) + len(token), token)
            for token in self.specials
            if token in prefix
        )
        split = best[0]
        common = prefix[:split]
        if not prompt.startswith(common):
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))
        key = hashlib.sha256(common.encode()).digest()
        if key not in self.prefix_cache:
            self.prefix_cache = {
                key: len(self.tokenizer.encode(common, add_special_tokens=False))
            }
        return self.prefix_cache[key] + len(
            self.tokenizer.encode(prompt[split:], add_special_tokens=False)
        )

    async def count(self, body):
        if body["state"] is not self.last_state_object:
            self.scaffold = self.layout._scaffold(
                self.tokenizer, body["state"], None, False
            )
            self.last_state_object = body["state"]
        prefix, _, ending, media = self.scaffold
        if media:
            raise ValueError("Text-only audit received media")
        question = next(iter(body["questions"].values()))
        block, _ = self.layout.question_block(question, ["A", "B", "C"])
        prompt = prefix + block + ending + self.layout.ANSWER
        count = self.exact_length(prompt)
        if count <= self.config.scoring["layout"]["twice_max_tokens"]:
            rendered = self.layout.render_request(
                body,
                self.tokenizer,
                self.layout.Layout(**self.config.scoring["layout"]),
                labels=["A", "B", "C"],
            )
            prompt = next(iter(rendered["prompts"].values()))[0][0]
            count = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        return count


def import_retro(output):
    directory = ROOT / "reports/retrosynthesis"
    runs = list(csv.DictReader((directory / "runs.csv").open()))
    jobs = []
    for run in runs:
        path = (
            directory / run["report_path"]
            if "report_path" in run
            else directory / run["report_file"]
        )
        signature = hashlib.sha256(path.read_bytes()).hexdigest()
        with (lzma.open if path.suffix == ".xz" else gzip.open)(path, "rt") as handle:
            report = json.load(handle)
        for task_id, task in report["task_results"].items():
            for index, trial in enumerate(task["trials"]):
                origin = {
                    "environment": "retrosynthesis",
                    "agent": run["agent"],
                    "model": run["model"],
                    "level": run["level"],
                    "task_id": task_id,
                    "trial": index,
                    "execution_id": trial["trial_id"],
                    "report_file": str(path.relative_to(ROOT / "reports")),
                    "source_report_sha256": signature,
                }
                identifier = content_hash(origin)[:24]
                if not trial.get("messages"):
                    jobs.append(
                        {"id": identifier, "origin": origin, "missing_messages": True}
                    )
                    continue
                prompt, refs = task_prompt(
                    output,
                    origin,
                    ROOT / "tasks",
                    ROOT / "src/corral/core/environment.py",
                )
                trace = CheckInput.from_document(trial).snapshot()
                trace["source"].update(
                    identity=origin["execution_id"],
                    locator=origin["report_file"],
                    content_hash=content_hash(trial),
                )
                trace["completion_status"] = (
                    "failed" if trial.get("error_message") else "complete"
                )
                trace["reconstruction_status"] = "complete"
                trace = annotation_transcript(trace, origin["agent"], prompt, refs)
                trace["grouping"] = {
                    "environment": "retrosynthesis",
                    "agent_model": f"{origin['agent']}__{origin['model']}",
                    "level": origin["level"],
                    "task": task_id,
                    "trial": str(index),
                }
                context = {"source_evidence": refs}
                inputs = CheckInput.from_document(trace, context)
                target = output / "retro-inputs" / f"{identifier}.json"
                atomic_save(target, {"trace": inputs.snapshot(), "context": context})
                jobs.append(
                    {
                        "id": identifier,
                        "input": str(target.resolve()),
                        "input_fingerprint": inputs.fingerprint,
                        "origin": origin,
                    }
                )
    return jobs


COUNTER = None


def initialize(model_path, config):
    global COUNTER  # noqa: PLW0603
    COUNTER = NativeCounter(Path(model_path), ProviderConfig(**config))


def audit_job(payload):
    job, output, identity = payload
    path = Path(output) / "records" / f"{job['id']}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("audit_identity") == identity and saved.get(
            "input_fingerprint"
        ) == job.get("input_fingerprint"):
            return saved
    if job.get("missing_messages"):
        record = {**job, "audit_identity": identity, "state": "missing_messages"}
    else:
        document = json.loads(Path(job["input"]).read_text())
        inputs = CheckInput.from_document(document["trace"], document["context"])
        if inputs.fingerprint != job["input_fingerprint"]:
            raise ValueError(f"Input fingerprint changed: {job['id']}")
        if is_scientist(inputs) and not job.get("scientist_unit"):
            unit_records = []
            for unit in scientist_units(inputs):
                unit_inputs = unit["inputs"]
                unit_path = (
                    Path(output) / "scientist-inputs" / job["id"] / f"{unit['id']}.json"
                )
                atomic_save(
                    unit_path,
                    {"trace": unit_inputs.snapshot(), "context": unit_inputs.context},
                )
                unit_job = {
                    **job,
                    "id": f"{job['id']}--{unit['id']}",
                    "input": str(unit_path),
                    "input_fingerprint": unit_inputs.fingerprint,
                    "scientist_unit": {k: v for k, v in unit.items() if k != "inputs"},
                    "parent_id": job["id"],
                }
                unit_records.append(audit_job((unit_job, output, identity)))
            record = {
                **job,
                "audit_identity": identity,
                "unit_policy": SCIENTIST_UNIT_VERSION,
                "unit_count": len(unit_records),
                "unit_records": [r["id"] for r in unit_records],
                "full_fits": all(r["full_fits"] for r in unit_records),
                "fits": all(r["fits"] for r in unit_records),
                "processable": all(r["processable"] for r in unit_records),
                "window_count": sum(r["window_count"] for r in unit_records),
                "windowed_units": sum(r["state"] == "windowed" for r in unit_records),
                "long_message_units": sum(
                    r.get("stage") == "long_messages" for r in unit_records
                ),
                "full_oversized_units": sum(not r["full_fits"] for r in unit_records),
                "still_oversized_units": sum(not r["fits"] for r in unit_records),
                "full_counts": {
                    r["id"]: max(r["full_counts"].values()) for r in unit_records
                },
                "counts": {r["id"]: max(r["counts"].values()) for r in unit_records},
                "view": None,
            }
            record["state"] = (
                "unchanged"
                if record["full_fits"]
                else "excerpted"
                if record["fits"]
                else "windowed"
                if record["processable"]
                else "unwindowable"
            )
            atomic_save(path, record)
            return record
        facts = extract_facts(inputs, ValidationPolicy())
        pipeline = CheckPipeline(Path(output) / "unused", COUNTER)
        prepared = asyncio.run(
            prepare_overflow(
                inputs,
                list(check_definitions().values()),
                COUNTER,
                lambda parts: pipeline.state(inputs, facts, parts),
            )
        )
        view = prepared.pop("view")
        catalogs = [
            info
            for mapping in (view["mappings"] if view else [])
            for info in (
                (
                    [mapping["catalog_reactions"]]
                    if "catalog_reactions" in mapping
                    else []
                )
                + mapping.get("embedded_catalog_reactions", [])
            )
        ]
        if view:
            atomic_save(
                Path(output) / "views" / f"{inputs.fingerprint}.json",
                {
                    "audit_identity": identity,
                    **prepared,
                    "view": view,
                },
            )
        record = {
            **job,
            **prepared,
            "audit_identity": identity,
            "state": "unchanged"
            if prepared["full_fits"]
            else "excerpted"
            if prepared["fits"]
            else "windowed"
            if prepared["windows"]
            else "unwindowable",
            "processable": prepared["fits"] or bool(prepared["windows"]),
            "window_count": len(prepared["windows"]),
            "original_characters": sum(
                len(m["content"]) for m in inputs.trace.messages
            ),
            "message_count": len(inputs.trace.messages),
            "view": str(Path(output) / "views" / f"{inputs.fingerprint}.json")
            if view
            else None,
            "catalog_outputs": len(catalogs),
            "catalog_reactions_retained": sorted(
                {info["retained"] for info in catalogs}
            ),
        }
    atomic_save(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = json.loads((args.run / "profile.json").read_text())
    config = profile["replicas"]["0"]
    source = json.loads((args.run / "manifest.json").read_text())
    jobs = [
        {**j, "input": str((args.run / j["input"]).resolve())}
        for j in source["selection"]
    ]
    retro_manifest = args.output / "retro-manifest.json"
    if retro_manifest.exists():
        retro = json.loads(retro_manifest.read_text())
    else:
        retro = import_retro(args.output)
        atomic_save(retro_manifest, retro)
    jobs.extend(retro)
    files = [
        args.model_path / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "knowline_layout.py",
            "knowline_server.py",
        )
    ]
    files.append(Path(__file__))
    files.extend(
        ROOT / "src/corral/epistemic_analysis" / name
        for name in (
            "overflow.py",
            "overflow_windows.py",
            "trace_checks.py",
            "decision_applicability.py",
            "scientist_units.py",
            "context.py",
        )
    )
    identity = content_hash(
        {
            "version": OVERFLOW_VERSION,
            "config": config,
            "definitions": [d.fingerprint for d in check_definitions().values()],
            "code": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        }
    )
    atomic_save(
        args.output / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "audit_identity": identity,
            "scope": "Frozen 1650-trace run plus current indexed retrosynthesis reports",
            "selection": jobs,
            "config": config,
            "context_tokens": config["context_tokens"],
        },
    )
    records = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(str(args.model_path), config),
    ) as pool:
        payloads = ((j, str(args.output.resolve()), identity) for j in jobs)
        for record in pool.map(audit_job, payloads, chunksize=1):
            records.append(record)
            if len(records) % 50 == 0:
                print(
                    len(records),
                    "/",
                    len(jobs),
                    dict(Counter(r["state"] for r in records)),
                    flush=True,
                )
    groups = defaultdict(Counter)
    stages = defaultdict(Counter)
    unit_groups = defaultdict(Counter)
    unit_rows = []
    for row in records:
        groups[row["origin"]["environment"]]["total"] += 1
        groups[row["origin"]["environment"]][row["state"]] += 1
        stages[row["origin"]["environment"]][
            row.get("stage", "independent_scientist_units")
        ] += 1
        for unit_id in row.get("unit_records", []):
            unit = json.loads((args.output / "records" / f"{unit_id}.json").read_text())
            unit_rows.append(unit)
            kind = unit["scientist_unit"]["kind"]
            unit_groups[kind]["total"] += 1
            unit_groups[kind][unit["state"]] += 1
    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "audit_identity": identity,
        "context_tokens": config["context_tokens"],
        "output_reserve": config["output_tokens"],
        "by_environment": dict(groups),
        "preprocessing_stages": dict(stages),
        "total": len(records),
        "scientist_unit_policy": SCIENTIST_UNIT_VERSION,
        "scientist_trace_fit_rule": "Every main/node unit must fit independently; a trace overflows if any unit overflows.",
        "scientist_units": dict(unit_groups),
        "token_count_note": "Maximum measured request per trace (or unit); counting stops at the first overflow, so oversized counts can be lower bounds across rubrics.",
    }
    atomic_save(args.output / "summary.json", summary)
    with (args.output / "trace-fit.csv").open("w") as handle:
        fields = [
            "id",
            "environment",
            "agent",
            "model",
            "level",
            "task_id",
            "trial",
            "state",
            "full_tokens",
            "prepared_tokens",
            "view",
            "unit_count",
            "full_oversized_units",
            "still_oversized_units",
            "stage",
            "window_count",
            "windowed_units",
            "long_message_units",
        ]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    **row,
                    **row["origin"],
                    "full_tokens": max(row.get("full_counts", {}).values(), default=0),
                    "prepared_tokens": max(row.get("counts", {}).values(), default=0),
                }
            )
    with (args.output / "scientist-unit-fit.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            [
                "parent_id",
                "id",
                "environment",
                "agent",
                "model",
                "level",
                "task_id",
                "trial",
                "kind",
                "node_id",
                "agent_run_id",
                "state",
                "full_tokens",
                "prepared_tokens",
                "view",
                "stage",
                "window_count",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in unit_rows:
            writer.writerow(
                {
                    **row,
                    **row["origin"],
                    **row["scientist_unit"],
                    "id": row["id"],
                    "full_tokens": max(row["full_counts"].values()),
                    "prepared_tokens": max(row["counts"].values()),
                }
            )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
