"""Build a shared, source-mapped transcript for decision labels and quotations."""

from __future__ import annotations

import copy
import json
import re

from .schema import content_hash

TRANSCRIPT_VERSION = "initial-instructions-and-react-batches-v4"
TASK_PROMPT_ORIGIN = "reconstructed_task_prompt"
SAVED_FIELDS = "\n[saved message fields]\n"
ACTION = re.compile(
    r"<action>(.*?)</action>\s*<action_input>(.*?)</action_input>", re.DOTALL
)


def _parts(message, source):
    text, separator, suffix = message["content"].rpartition(SAVED_FIELDS)
    if not separator:
        return message["content"], None
    try:
        fields = json.loads(suffix)
    except ValueError:
        return message["content"], None
    if not isinstance(fields, dict) or any(
        fields.get(key) != source[key]
        for key in ("agent_run_id", "name", "tool_call_id")
        if key in source
    ):
        return message["content"], None
    return text, fields


def _call_signature(call):
    try:
        function = call["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or not call.get("id"):
            return None
        return function["name"], arguments
    except (KeyError, TypeError, ValueError):
        return None


def _explicit_actions(text):
    """Only merge dispatches whose actions and arguments are unambiguous."""
    matches = list(ACTION.finditer(text))
    if len(matches) != text.count("<action>"):
        return []
    try:
        actions = [(m[1].strip(), json.loads(m[2])) for m in matches]
    except ValueError:
        return []
    return actions if all(isinstance(args, dict) for _, args in actions) else []


def annotation_transcript(
    document,
    agent,
    prompt,
    prompt_sources,
    *,
    fold_react_dispatches=True,
    initial_messages=None,
    initial_sources=None,
):
    """Prepend task instructions and fold only proven ReAct dispatch duplicates.

    All tool results and real assistant prose survive. The original model text
    already contains every proposed XML action; the merged metadata separately
    lists the calls that were actually dispatched. Unexecuted proposals never
    acquire fabricated call IDs or results. Other agents are not coalesced.
    """
    if document.get("annotation_preprocessing"):
        raise ValueError("Prepare from the original transcript, not an annotated view")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A rendered task prompt is required")
    result = copy.deepcopy(document)
    messages = [{"role": "user", "content": prompt}]
    sources = [
        {
            "index": None,
            "origin": TASK_PROMPT_ORIGIN,
            "source_evidence": copy.deepcopy(prompt_sources),
            "description": (
                "Task instructions reconstructed from the task definition and prompt "
                "renderer for annotation. This is not an observed agent action. "
                "Use recorded runtime instructions if they differ."
            ),
        }
    ]
    if initial_messages is not None or initial_sources is not None:
        if (
            not initial_messages
            or not initial_sources
            or len(initial_messages) != len(initial_sources)
        ):
            raise ValueError("Initial instructions require one source per message")
        if any(
            m.get("role") not in {"system", "developer", "user"}
            or not isinstance(m.get("content"), str)
            or not m["content"].strip()
            for m in initial_messages
        ):
            raise ValueError(
                "Initial instructions must be nonempty instruction messages"
            )
        if not any(prompt in m["content"] for m in initial_messages):
            raise ValueError(
                "Initial instructions must include the rendered task prompt"
            )
        messages = copy.deepcopy(initial_messages)
        sources = copy.deepcopy(initial_sources)
    initial_count = len(messages)
    old_to_new, folded, batch = {}, [], None
    for index, (message, source) in enumerate(
        zip(document["messages"], document["message_sources"], strict=True)
    ):
        text, fields = _parts(message, source)
        actor = source.get("agent_run_id")
        calls = fields.get("tool_calls", []) if fields else []
        if not isinstance(calls, list):
            calls = []
        signatures = [_call_signature(call) for call in calls]
        if (
            fold_react_dispatches
            and agent == "react"
            and message["role"] == "assistant"
        ):
            # Never erase prose, error records, actor changes, unmatched calls,
            # or independently issued repetitions of the same operation.
            if (
                batch is not None
                and text.strip() in {"", "null"}
                and fields is not None
                and set(fields) <= {"tool_calls", "agent_run_id"}
                and actor == batch["actor"]
                and signatures
                and all(signature is not None for signature in signatures)
                and signatures
                == batch["actions"][batch["next"] : batch["next"] + len(calls)]
                and index > 0
                and document["messages"][index - 1]["role"] == "tool"
                and all(call["id"] not in batch["ids"] for call in calls)
            ):
                target = batch["target"]
                batch["fields"]["tool_calls"].extend(copy.deepcopy(calls))
                batch["next"] += len(calls)
                batch["ids"].update(call["id"] for call in calls)
                messages[target]["content"] = (
                    batch["text"]
                    + SAVED_FIELDS
                    + json.dumps(batch["fields"], ensure_ascii=False)
                )
                sources[target].setdefault("merged_dispatch_indices", []).append(index)
                old_to_new[str(index)] = target
                folded.append(
                    {"original_message_index": index, "message_index": target}
                )
                continue
            batch = None
            actions = _explicit_actions(text)
            if (
                fields is not None
                and actions
                # A later submission must stay after the observations preceding
                # its dispatch. Folding it into the proposal would move the
                # graph's submission boundary ahead of those observations.
                and all(name != "submit_answer" for name, _ in actions)
                and signatures
                and all(signature is not None for signature in signatures)
                and signatures == actions[: len(signatures)]
            ):
                batch = {
                    "target": len(messages),
                    "actor": actor,
                    "text": text,
                    "fields": copy.deepcopy(fields),
                    "actions": actions,
                    "next": len(calls),
                    "ids": {call["id"] for call in calls},
                }
        elif batch is not None and (
            message["role"] != "tool"
            or actor != batch["actor"]
            or source.get("tool_call_id") not in batch["ids"]
        ):
            batch = None
        old_to_new[str(index)] = len(messages)
        messages.append(copy.deepcopy(message))
        sources.append({**copy.deepcopy(source), "original_message_index": index})

    result["messages"], result["message_sources"] = messages, sources
    result["annotation_preprocessing"] = {
        "version": TRANSCRIPT_VERSION,
        "original_messages_sha256": content_hash(document["messages"]),
        "original_message_count": len(document["messages"]),
        "message_count": len(messages),
        "original_to_annotation_indices": old_to_new,
        "folded_react_dispatches": folded,
        "task_prompt_index": next(
            i for i, m in enumerate(messages[:initial_count]) if prompt in m["content"]
        ),
        "initial_instruction_indices": list(range(initial_count)),
    }
    return result
