"""Execution facts and reviewed applicability for model-assessed validation.

Code records calls, results, and candidate relationships. Models assess whether
the submitted answer received the relevant verification; tool counts do not
assign behavioral labels or determine applicability.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field

from corral.epistemic_analysis.schema import content_hash

VALIDATION_CHECK = "insufficient_validation"
SAVED_FIELDS = "\n[saved message fields]\n"


def _saved_fields(message, source):
    """Read only the normalizer's final metadata suffix, never action prose."""
    content, separator, suffix = message["content"].rpartition(SAVED_FIELDS)
    if not separator:
        return message["content"], {}
    try:
        fields = json.loads(suffix)
    except (ValueError, TypeError):
        return message["content"], {}
    if not isinstance(fields, dict) or any(
        fields.get(k) != source[k]
        for k in ("name", "tool_call_id", "agent_run_id")
        if k in source
    ):
        return message["content"], {}
    return content, fields


def _object(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


POLICY_VERSION = "final-answer-applicability-1"


@dataclass(frozen=True)
class ValidationPolicy:
    """An explicit reviewed applicability contract, never inferred from tool counts.

    Obligations contain a property, requirement text, and source references. Coverage
    and feasibility are recorded separately by the facts layer and assessed by models.
    An empty validator inventory, a blocker, or a model response cannot establish N/A.
    """

    version: str = POLICY_VERSION
    applicability: str = "unknown"
    reviewed_by: str = ""
    rationale: str = "Task-specific applicability has not been reviewed."
    source_evidence: list = field(default_factory=list)
    obligations: list = field(default_factory=list)

    def __post_init__(self):
        if not self.version or self.applicability not in {
            "applicable",
            "not_applicable",
            "unknown",
        }:
            raise ValueError("Invalid applicability contract")
        if self.applicability != "unknown" and (
            not self.reviewed_by.strip()
            or not self.rationale.strip()
            or not self.source_evidence
        ):
            raise ValueError(
                "Known applicability requires a reviewer, rationale, and source evidence"
            )
        for obligation in self.obligations:
            if not all(
                obligation.get(k)
                for k in ("id", "property", "requirement", "source_evidence")
            ):
                raise ValueError(
                    "An obligation requires an id, property, requirement, and sources"
                )
        if len({o["id"] for o in self.obligations}) != len(self.obligations):
            raise ValueError("Duplicate obligation ids")

    def decision(self, name, facts=None):
        if name != VALIDATION_CHECK:
            return {"applicability": "applicable", "decision_source": "model"}
        if (
            self.applicability == "unknown"
            and not self.reviewed_by
            and facts
            and facts.get("final_accepted_submission")
            and facts.get("task_context_available")
        ):
            return {
                "applicability": "applicable",
                "decision_source": "model",
                "policy_version": self.version,
                "rationale": "Recorded accepted submission and task instructions establish the scope; the model determines obligations and coverage.",
            }
        return {
            "applicability": self.applicability,
            "decision_source": "model"
            if self.applicability == "applicable"
            else "code",
            "policy_version": self.version,
            "rationale": self.rationale,
            **(
                {
                    "status": "not_applicable"
                    if self.applicability == "not_applicable"
                    else "uncertain"
                }
                if self.applicability != "applicable"
                else {}
            ),
        }


FACTS_VERSION = "execution-facts-1"


def span(trace, index, *, metadata=False):
    content = trace.messages[index]["content"]
    start = max(0, content.rfind(SAVED_FIELDS)) if metadata else 0
    return {
        "msg_idx": index,
        "message_id": f"m{index}",
        "start": start,
        "end": len(content),
        "quote": content[start:],
        "source": trace.message_sources[index],
    }


def outcome(name, output):
    """Separate completion from verdict, and predictions from empirical comparison."""
    base = {"execution": "uninterpretable", "verdict": "unknown", "properties": []}
    try:
        value = json.loads(output)
    except (ValueError, TypeError):
        try:
            value = ast.literal_eval(output)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return base
    if isinstance(value, dict) and value.get("error"):
        base["execution"] = "failed"
    if name == "validate_smiles" and isinstance(value, bool):
        return {
            "execution": "completed",
            "verdict": "pass" if value else "fail",
            "properties": ["smiles_parse_validity"],
        }
    if (
        name == "validate_model_syntax"
        and isinstance(value, dict)
        and isinstance(value.get("valid"), bool)
    ):
        return {
            "execution": "completed",
            "verdict": "pass" if value["valid"] else "fail",
            "properties": ["model_syntax_validity"],
        }
    if name == "simulate_spectra" and isinstance(value, dict):
        modalities = [
            k
            for k in ("h_nmr", "c13_nmr", "ir")
            if isinstance(value.get(k), str)
            and value[k].strip()
            and not any(
                w in value[k].lower() for w in ("error", "failed", "unavailable")
            )
        ]
        return {
            "execution": "completed" if modalities else "failed",
            "verdict": "not_a_verdict",
            "properties": [f"{k}_prediction" for k in modalities],
            "unavailable_modalities": [
                k for k in ("h_nmr", "c13_nmr", "ir") if k not in modalities
            ],
        }
    return base


def candidate(event, *, submitted=False):
    args = event.get("arguments") or {}
    if submitted:
        value = args.get("answer")
        structured = _object(value)
        if structured and "model_syntax" in structured:
            return {"kind": "syntax", "value": structured["model_syntax"]}
        return {"kind": "answer", "value": value}
    for key in ("smiles", "syntax"):
        if key in args:
            return {"kind": key, "value": args[key]}
    # Paths are not artifact identities. Only saved contents can identify versions.
    return {"kind": "unknown", "value": None}


def extract_facts(inputs, policy):
    trace = inputs.trace
    events, calls, limitations = (
        [],
        {},
        list(inputs.context.get("capture_limitations", [])),
    )
    for i, (message, source) in enumerate(
        zip(trace.messages, trace.message_sources, strict=True)
    ):
        text, fields = _saved_fields(message, source)
        actor, branch = source.get("agent_run_id"), source.get("branch_id")
        if message["role"] == "assistant":
            if "function_call" in fields:
                limitations.append(
                    {
                        "reason": "legacy_call_without_pairing_id",
                        "source": span(trace, i),
                    }
                )
            listed = fields.get("tool_calls", [])
            if not isinstance(listed, list):
                limitations.append(
                    {"reason": "malformed_calls", "source": span(trace, i)}
                )
                continue
            for call in listed:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if (
                    not isinstance(function, dict)
                    or not function.get("name")
                    or not call.get("id")
                ):
                    limitations.append(
                        {"reason": "malformed_call", "source": span(trace, i)}
                    )
                    continue
                key = (actor, branch, call["id"])
                event = {
                    "id": f"event-{len(events)}",
                    "call_id": call["id"],
                    "name": function["name"],
                    "actor": actor,
                    "branch": branch,
                    "call_index": i,
                    "call_order": len(events),
                    "arguments": _object(function.get("arguments")),
                    "call_source": span(trace, i, metadata=True),
                    "execution": "no_paired_result",
                    "verdict": "unknown",
                    "properties": [],
                }
                events.append(event)
                if key in calls:
                    calls[key] = None
                    limitations.append(
                        {"reason": "duplicate_call_id", "source": span(trace, i)}
                    )
                else:
                    calls[key] = event
        elif message["role"] == "tool":
            key = (actor, branch, source.get("tool_call_id"))
            event = calls.get(key)
            if (
                not event
                or event["name"] != source.get("name")
                or "result_index" in event
            ):
                limitations.append(
                    {"reason": "unpaired_or_ambiguous_result", "source": span(trace, i)}
                )
                continue
            event.update(outcome(event["name"], text))
            event.update(
                result_index=i,
                result_source=span(trace, i),
                output=text,
                availability={
                    "actor": actor,
                    "branch": branch,
                    "message_index": i,
                    "basis": "paired result in recorded actor stream",
                },
            )
            if event["name"] == "submit_answer":
                event["accepted"] = text.strip() == "answer accepted"
    for event in events:
        if "result_index" not in event:
            limitations.append({"reason": "missing_result", "event_id": event["id"]})
        if event["name"] not in {
            "validate_smiles",
            "validate_model_syntax",
            "simulate_spectra",
            "submit_answer",
        }:
            limitations.append(
                {
                    "reason": "other_method_or_indirect_execution_requires_model_review",
                    "event_id": event["id"],
                }
            )
    submissions = [
        e for e in events if e["name"] == "submit_answer" and e.get("accepted")
    ]
    final = submissions[-1] if submissions else None
    relationships = []
    for event in events:
        if not final or event["name"] == "submit_answer":
            continue
        left, right = candidate(event), candidate(final, submitted=True)
        comparable = (
            left["kind"] != "unknown"
            and isinstance(left["value"], str)
            and isinstance(right["value"], str)
        )
        relation = "unknown"
        if comparable:
            relation = (
                "identical_representation"
                if left["value"] == right["value"]
                else "different_representation_equivalence_unknown"
            )
        same_actor = (
            event["actor"] == final["actor"] and event["branch"] == final["branch"]
        )
        relationships.append(
            {
                "event_id": event["id"],
                "submission_id": final["id"],
                "candidate_relation": relation,
                "candidate_hash": content_hash(left) if comparable else None,
                "submitted_hash": content_hash(right) if comparable else None,
                "available_before_submission": (
                    event["result_index"] < final["call_index"]
                    if same_actor and "result_index" in event
                    else None
                ),
                "access": "same_recorded_stream"
                if same_actor
                else "delegation_or_cross_branch_access_unconfirmed",
                "materiality": "model_assessed",
                "coverage": "model_assessed",
            }
        )
    return {
        "version": FACTS_VERSION,
        "events": events,
        "candidate_relationships": relationships,
        "task_context_available": bool(
            inputs.public_state()["initial_instructions"]
            or inputs.context.get("recorded_initial_instructions")
            or inputs.context.get("recovered_task_context")
            or inputs.context.get("public_task_context")
        ),
        "final_accepted_submission": final["id"] if final else None,
        "submission_scope": "final_accepted",
        "capture_limitations": limitations,
        "completion_status": trace.completion_status,
        "reconstruction_status": trace.reconstruction_status,
        "policy": asdict(policy),
        "obligations": [
            {
                **o,
                "coverage": "unknown",
                "feasibility": "unknown",
                "confirmed_blockers": [],
            }
            for o in policy.obligations
        ],
        "note": "Facts do not establish sufficient validation. No tool count, order, or candidate relation assigns a behavioral label.",
    }
