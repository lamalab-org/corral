"""Independent AI Scientist node/main assessments with deterministic check OR."""

from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from dataclasses import asdict, replace

from .annotation_transcript import TASK_PROMPT_ORIGIN
from .decision_units import NODE_NAME, aggregate_or
from .overflow import OVERFLOW_VERSION
from .scheduling import DEFAULT_TRACE_CONCURRENCY, rolling_map
from .schema import content_hash
from .sources.inputs import CheckInput
from .storage import atomic_save, read_record, save_record

SCIENTIST_UNIT_VERSION = "scientist-independent-node-main-or-v1"


def is_scientist(inputs):
    return (inputs.trace.grouping.agent_model or "").startswith("ai-scientist__")


def scientist_units(inputs):
    """Partition observed messages once; task instructions are shared context.

    Node identities include the agent run so equal node numbers from different
    runs are never merged. Experiment steps and evaluations belong to that node;
    planning, task formulation, synthesis, and submissions belong to the main
    agent. Captured messages are not deduplicated or cut at a submission.
    """
    trace = inputs.trace
    anchors = [
        i
        for i, s in enumerate(trace.message_sources)
        if s.get("origin") == TASK_PROMPT_ORIGIN
    ]
    actors = {s["agent_run_id"] for s in trace.message_sources if s.get("agent_run_id")}
    default_actor = next(iter(actors)) if len(actors) == 1 else "unscoped"
    groups = defaultdict(list)
    for i, source in enumerate(trace.message_sources):
        if i in anchors:
            continue
        match = NODE_NAME.fullmatch(source.get("name", ""))
        node = (match[1] or match[2]) if match else "main"
        groups[(source.get("agent_run_id") or default_actor, node)].append(i)
    result = []
    for (actor, node), observed in sorted(groups.items()):
        identity = {
            "agent_run_id": actor,
            "node_id": node,
            "kind": "main" if node == "main" else "node",
        }
        unit_id = f"{node}-{content_hash(identity)[:12]}"
        indices = sorted([*anchors, *observed])
        sources = [
            {
                **copy.deepcopy(trace.message_sources[i]),
                "parent_msg_idx": i,
                "scientist_unit": identity,
                "context_only": i in anchors,
            }
            for i in indices
        ]
        unit_trace = replace(
            trace,
            messages=[copy.deepcopy(trace.messages[i]) for i in indices],
            message_sources=sources,
            source=replace(
                trace.source,
                identity=f"{trace.source.identity}#{unit_id}",
                details={**trace.source.details, "scientist_unit": identity},
            ),
            snapshot=None,
        )
        result.append(
            {
                "id": unit_id,
                **identity,
                "indices": indices,
                "observed_indices": observed,
                "inputs": CheckInput(unit_trace, inputs.context),
            }
        )
    if sorted(i for unit in result for i in unit["observed_indices"]) != [
        i for i in range(len(trace.messages)) if i not in anchors
    ]:
        raise ValueError("Scientist partition lost or duplicated observed messages")
    return result


async def run_scientist_checks(
    pipeline,
    inputs,
    *,
    policy,
    checks,
    evidence_only,
    force_labels,
    force_evidence,
    label_records,
    on_label,
):
    """Run each unit through the normal pipeline; never ask a model to aggregate."""
    from .decision_applicability import ValidationPolicy, extract_facts
    from .trace_checks import CheckPipeline, check_definitions

    policy = policy or ValidationPolicy()
    definitions = check_definitions(checks)
    if not definitions:
        raise ValueError("Select at least one check")
    units = scientist_units(inputs)
    output = pipeline.output
    settings = {
        "format": SCIENTIST_UNIT_VERSION,
        "input": inputs.fingerprint,
        "definitions": [d.fingerprint for d in definitions.values()],
        "policy": asdict(policy),
        "provider": asdict(pipeline.decision.config),
        "threshold": pipeline.threshold,
        "tool_cap": pipeline.overflow_tool_output_max_chars,
        "overflow_policy": OVERFLOW_VERSION,
        "windowed": pipeline.windowed,
        "window_tokens": pipeline.decision_window_tokens,
        "reconcile_windowed": pipeline.reconcile_windowed,
    }
    key = content_hash(
        {
            **settings,
            "evidence": asdict(pipeline.evidence.config) if pipeline.evidence else None,
        }
    )
    index_key = content_hash({"scientist_label_index": settings})
    index_path = (
        output / "inputs" / inputs.fingerprint / f"scientist-labels-{index_key}.json"
    )
    if evidence_only and not label_records:
        saved = read_record(index_path, index_key)
        if not saved:
            raise ValueError("Evidence-only scientist run requires saved unit labels")
        label_records = saved["labels"]
    run = {
        "format": SCIENTIST_UNIT_VERSION,
        "fingerprint": key,
        "input_fingerprint": inputs.fingerprint,
        "state": "running",
        "labels": {},
        "evidence": {},
        "units": [],
        "code_checks": [],
        "qualification": {
            "status": "opt_in_unqualified",
            "unit_policy": SCIENTIST_UNIT_VERSION,
        },
    }
    path = output / "runs" / f"{key}.json"
    atomic_save(
        output / "inputs" / inputs.fingerprint / "trace.json", inputs.snapshot()
    )
    atomic_save(output / "inputs" / inputs.fingerprint / "context.json", inputs.context)
    facts = extract_facts(inputs, policy)
    run["facts_hash"] = content_hash(facts)
    atomic_save(
        output / "inputs" / inputs.fingerprint / f"facts-{run['facts_hash']}.json",
        facts,
    )
    atomic_save(path, run)
    prior = {}
    for name, label_key in (label_records or {}).items():
        label = read_record(output / "labels" / f"{label_key}.json", label_key)
        if (
            not label
            or label.get("input_fingerprint") != inputs.fingerprint
            or label.get("definition") != asdict(definitions[name])
            or label.get("unit_policy") != SCIENTIST_UNIT_VERSION
        ):
            raise ValueError(
                "Prior aggregate does not match scientist input, rubric, or unit policy"
            )
        prior[name] = label
    if evidence_only and set(prior) != set(definitions):
        raise ValueError("Evidence-only scientist run requires every aggregate label")

    def child(directory, evidence=None):
        return CheckPipeline(
            directory,
            pipeline.decision,
            evidence,
            threshold=pipeline.threshold,
            windowed=pipeline.windowed,
            repair_attempts=pipeline.repair_attempts,
            reconcile_windowed=pipeline.reconcile_windowed,
            decision_window_tokens=pipeline.decision_window_tokens,
            share_window_prefill=pipeline.share_window_prefill,
            overflow_tool_output_max_chars=pipeline.overflow_tool_output_max_chars,
            split_scientist=False,
        )

    try:
        assessments = defaultdict(list)

        async def label_unit(unit):
            directory = output / "scientist-units" / inputs.fingerprint / unit["id"]
            labels = {
                name: prior[name]["unit_labels"][unit["id"]] for name in prior
            } or None
            unit_run = await child(directory).run(
                unit["inputs"],
                policy=policy,
                checks=list(definitions),
                evidence_only=evidence_only,
                force_labels=force_labels,
                label_records=labels,
            )
            unit["directory"], unit["run"] = directory, unit_run

        await rolling_map(label_unit, units, concurrency=DEFAULT_TRACE_CONCURRENCY)
        # Aggregate in source order so scheduling never changes label identities.
        for unit in units:
            directory, unit_run = unit["directory"], unit["run"]
            run["units"].append(
                {
                    k: v
                    for k, v in unit.items()
                    if k not in {"inputs", "directory", "run"}
                }
                | {"output": str(directory), "run": unit_run["fingerprint"]}
            )
            for name, label_key in unit_run["labels"].items():
                label = read_record(
                    directory / "labels" / f"{label_key}.json", label_key
                )
                assessments[name].append(
                    {
                        "unit_id": unit["id"],
                        "label": label_key,
                        "status": label["status"],
                        "state": label["state"],
                    }
                )
        for name, definition in definitions.items():
            values = assessments[name]
            label_key = content_hash(
                {
                    "unit_policy": SCIENTIST_UNIT_VERSION,
                    "input": inputs.fingerprint,
                    "definition": definition.fingerprint,
                    "units": values,
                }
            )
            label = {
                "fingerprint": label_key,
                "input_fingerprint": inputs.fingerprint,
                "name": name,
                "definition": asdict(definition),
                "policy": asdict(policy),
                "provenance": asdict(pipeline.decision.config),
                "unit_policy": SCIENTIST_UNIT_VERSION,
                "decision_source": "code_any_unit",
                "model_called": False,
                "exchanges": [],
                "status": aggregate_or(values),
                "unit_assessments": values,
                "unit_labels": {v["unit_id"]: v["label"] for v in values},
                "state": "complete"
                if values and all(v["state"] == "complete" for v in values)
                else "requires_review",
            }
            save_record(output / "labels" / f"{label_key}.json", label)
            run["labels"][name] = label_key
            if on_label is not None:
                await on_label(label)
        atomic_save(path, run)
        save_record(index_path, {"fingerprint": index_key, "labels": run["labels"]})
        if pipeline.evidence is not None:
            evidence_by_check = defaultdict(list)

            async def explain_unit(unit):
                return await child(unit["directory"], pipeline.evidence).run(
                    unit["inputs"],
                    policy=policy,
                    checks=list(definitions),
                    evidence_only=True,
                    label_records=unit["run"]["labels"],
                    force_evidence=force_evidence,
                )

            explained = await rolling_map(
                explain_unit, units, concurrency=DEFAULT_TRACE_CONCURRENCY
            )
            for unit, unit_run in zip(units, explained, strict=True):
                for name, evidence_key in unit_run["evidence"].items():
                    record = read_record(
                        unit["directory"] / "evidence" / f"{evidence_key}.json",
                        evidence_key,
                    )
                    evidence_by_check[name].append(
                        {
                            "unit_id": unit["id"],
                            "record": evidence_key,
                            "state": record["state"],
                            "evidence": [
                                {
                                    **quote,
                                    "msg_idx": unit["indices"][quote["msg_idx"]],
                                    "message_id": f"m{unit['indices'][quote['msg_idx']]}",
                                    "unit_id": unit["id"],
                                }
                                for quote in record.get("evidence", [])
                            ],
                        }
                    )
            for name, records in evidence_by_check.items():
                evidence_key = content_hash(
                    {"label": run["labels"][name], "unit_evidence": records}
                )
                record = {
                    "fingerprint": evidence_key,
                    "label_fingerprint": run["labels"][name],
                    "name": name,
                    "unit_evidence": records,
                    "evidence": [q for r in records for q in r["evidence"]],
                    "state": "complete"
                    if all(r["state"] == "complete" for r in records)
                    else "requires_review",
                }
                save_record(output / "evidence" / f"{evidence_key}.json", record)
                run["evidence"][name] = evidence_key
        states = [
            read_record(output / stage / f"{fingerprint}.json", fingerprint)["state"]
            for stage in ("labels", "evidence")
            for fingerprint in run[stage].values()
        ]
        run["state"] = (
            "complete"
            if states and all(s == "complete" for s in states)
            else "requires_review"
        )
    except asyncio.CancelledError:
        run["state"] = "cancelled"
        raise
    except Exception:
        run["state"] = "failed"
        raise
    finally:
        atomic_save(path, run)
    return run
