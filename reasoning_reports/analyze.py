"""
Unified reasoning analysis pipeline for annotated traces.

Combines three former scripts:
  1. LLM-based trace annotation (Pass A: nodes, Pass B: edges)
  2. Per-directory analysis (aggregate stats, plots, timelines)
  3. Cross-model/level aggregation

Directory structure expected under <root>/:
  <model>/
    <env>/
      <level>/
        *.json                 <- raw trace files
        annotated/
          *.annotated.json     <- annotation outputs
          analysis/            <- per-level analysis outputs
  analysis/                    <- cross-model aggregation outputs

Usage:
  python analyze.py                                # full pipeline
  python analyze.py --skip_annotate                # only analysis + aggregation
  python analyze.py --skip_analysis                # only annotation
  python analyze.py --dry_run                      # validate without LLM calls
  python analyze.py --concurrency=4                # limit to 4 concurrent files
  python analyze.py --model=openai/gpt-4o          # use a different LLM
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    import matplotlib  # noqa

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SCRIPT_DIR = Path(__file__).resolve().parent

NODE_TYPES = ["H", "T", "E", "J", "U", "C"]
NODE_TYPES_SET = set(NODE_TYPES)
EDGE_RELATIONS = [
    "tests",
    "observes",
    "uses",
    "updates_to",
    "competes_with",
    "contradicts",
]
EDGE_RELATIONS_SET = set(EDGE_RELATIONS)

SUBGRAPH_NAMES = [
    "popperian_falsification",
    "ml_make_it_work",
    "exploratory_to_confirmatory",
    "bayesian_belief_updating",
    "abductive",
    "triangulation",
    "preregistered",
    "active_learning",
]

ANTIPATTERN_NAMES = [
    "untested_hypothesis",
    "evidence_ignored",
    "judgment_without_evidence",
    "dead_end_update",
    "unresolved_contradiction",
    "hypothesis_to_commitment_shortcut",
    "test_without_evidence",
    "no_belief_revision",
    "orphan_evidence",
    "confirmation_only",
]

ANTIPATTERN_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        "untested_hypothesis",
        "unresolved_contradiction",
        "confirmation_only",
    ],
    "evidence_handling": [
        "evidence_ignored",
        "orphan_evidence",
        "judgment_without_evidence",
        "test_without_evidence",
    ],
    "experimental_strategy": [
        "dead_end_update",
        "no_belief_revision",
        "hypothesis_to_commitment_shortcut",
    ],
}
ANTIPATTERN_FAMILY_NAMES = list(ANTIPATTERN_FAMILIES.keys())

SUBGRAPH_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        "popperian_falsification",
        "bayesian_belief_updating",
        "abductive",
    ],
    "evidence_handling": [
        "triangulation",
        "exploratory_to_confirmatory",
    ],
    "experimental_strategy": [
        "ml_make_it_work",
        "preregistered",
        "active_learning",
    ],
}
SUBGRAPH_FAMILY_NAMES = list(SUBGRAPH_FAMILIES.keys())

DEFAULT_WINDOW = 16
DEFAULT_OVERLAP = 4
DEFAULT_MAX_NODES_PER_WINDOW = 100
DEFAULT_CONCURRENCY = 8

METRIC_FIELDS = [
    "workflow_completeness",
    "loop_density",
    "update_grounding_rate",
    "orphan_evidence_rate",
    "refute_neglect_rate",
    "hypothesis_switch_without_eval_rate",
    "scientificness_score",
]

COUNT_FIELDS = (
    ["nodes_total", "edges_total"]
    + [f"n_{t}" for t in NODE_TYPES]
    + [f"e_{r}" for r in EDGE_RELATIONS]
)

NODE_COUNT_FIELDS = ["nodes_total", *[f"n_{t}" for t in NODE_TYPES]]


PASS_A_SYSTEM = """You are a careful annotator. You MUST only extract information explicitly present in the provided messages.
Rules:
- Do NOT invent hidden thoughts or implied steps.
- Every node MUST include at least one support quote with message indices.
- For messages with several nodes, you MUST follow the order in which they appear in the text to assign message indices.
- If uncertain, omit the node rather than guessing.
- Output JSON only, matching the required schema.
"""

PASS_A_INSTRUCTIONS = """Extract 0..k nodes from the provided message window.

Node types:
H = Hypothesis (candidate explanation, or state)
T = Test design or information-seeking plan
E = Evidence / observation (result of a test, tool call, or direct observation)
J = Judgment (interpretation of evidence, or comparison - it cannot be only paraphrasing the tool output; "what does one observation imply?")
U = Update (change in beliefs, revision of best hypothesis, revising)
C = Commitment / final answer / lock-in statement / committing to one hypothesis ("the answer is X", "I conclude ...")

Constraints:
- Only label what is explicitly present in text.
- Every node must have support quotes (exact substrings) and msg indices.
- If you normalize text, still cite original quote(s).

Hypothesis identity:
- For each H node, produce "canonical" (short normalized form) and "hid" = sha1(canonical).
- If hypothesis is vague, canonical can be the minimal explicit claim.

Return JSON with keys:
{
  "nodes": [
    {
      "node_id": "N1",
      "type": "H|T|E|J|U|V|C",
      "time": <int message index of earliest support>,
      "text": <normalized short node text>,
      "support": [{"msg_idx": <int>, "quote": <exact substring from that message>}],
      "hypothesis": {"canonical": <str>, "hid": <str>}   // only if type == "H"
    }, ...
  ]
}
"""

PASS_B_SYSTEM = """You are a careful annotator. You MUST only add edges supported by explicit text.
Rules:
- You may only connect nodes provided to you.
- Every edge MUST include at least one support quote with message indices.
- If uncertain, omit the edge rather than guessing.
- Output JSON only, matching the required schema.
"""

PASS_B_INSTRUCTIONS = """Given the message window and a list of extracted nodes (with node_id and text), extract supported edges among these nodes.

Allowed relations:
tests, observes, uses, updates_to, competes_with, contradicts

Guidance:
- tests: T evaluates a hypothesis or leads to evidence
- observes: E is an observation from T or from a tool result
- uses: J or U uses E (or uses another node) as support
- updates_to: U transitions from one H to another H, or updates belief state
- competes_with: between H alternatives
- contradicts: E contradicts/refutes H, or J refutes H explicitly

Return JSON with keys:
{
  "edges": [
    {
      "src": "<node_id>",
      "dst": "<node_id>",
      "relation": "<one of allowed>",
      "time": <int message index of earliest support>,
      "support": [{"msg_idx": <int>, "quote": <exact substring from that message>}]
    }, ...
  ]
}
"""


class LLMError(RuntimeError):
    pass


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def safe_read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _mean(values: list[float | None]) -> float | None:
    clean = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return sum(clean) / len(clean) if clean else None


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _load_litellm():
    import litellm

    return litellm


def _extract_json_text(raw: str) -> str:
    """Strip markdown fences and find the outermost JSON object in *raw*."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


async def llm_json_call_async(
    model: str,
    system: str,
    user: str,
    temperature: float = 1.0,
    max_retries: int = 5,
    timeout_s: int = 120,
) -> dict[str, Any]:
    litellm = _load_litellm()
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                timeout=timeout_s,
            )
            content = resp["choices"][0]["message"]["content"] or ""
            content = _extract_json_text(content)
            if not content:
                raise ValueError("LLM returned empty content")
            return json.loads(content)
        except Exception as e:
            last_err = e
            logger.warning(f"LLM call attempt {attempt}/{max_retries} failed: {e}")
            await asyncio.sleep(min(2**attempt, 30))
    raise LLMError(
        f"LLM call failed after {max_retries} retries: {last_err}"
    ) from last_err


def iter_windows(n: int, window: int, overlap: int) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    if window <= 0:
        return [(0, n)]
    step = max(1, window - overlap)
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + window)
        out.append((start, end))
        if end == n:
            break
        start += step
    return out


def format_window(messages: list[dict[str, Any]], start: int, end: int) -> str:
    lines = []
    for i in range(start, end):
        role = messages[i].get("role", "")
        content = messages[i].get("content", "")
        if content is None:
            content = ""
        content = str(content)
        lines.append(f"[{i}] role={role}\n{content}\n")
    return "\n".join(lines)


def validate_support_quotes(
    messages: list[dict[str, Any]],
    support: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    ok = True
    for s in support:
        if "msg_idx" not in s or "quote" not in s:
            ok = False
            warnings.append("Support item missing msg_idx or quote.")
            continue
        idx = s["msg_idx"]
        quote = s["quote"]
        if not isinstance(idx, int) or idx < 0 or idx >= len(messages):
            ok = False
            warnings.append(f"Support msg_idx out of range: {idx}")
            continue
        msg_content = str(messages[idx].get("content", "") or "")
        if quote not in msg_content:
            ok = False
            warnings.append(
                f"Quote not found verbatim in message {idx}: {quote[:80]!r}"
            )
    return ok, warnings


def coerce_node(
    node: dict[str, Any], messages: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[str]]:
    warns: list[str] = []
    t = node.get("type")
    if t not in NODE_TYPES_SET:
        return None, [f"Invalid node type: {t}"]
    support = node.get("support", [])
    _ok, w = validate_support_quotes(
        messages, support if isinstance(support, list) else []
    )
    warns.extend(w)

    time_idx = node.get("time")
    if not isinstance(time_idx, int):
        try:
            time_idx = min(int(x["msg_idx"]) for x in support)
        except Exception:
            time_idx = 0
            warns.append("Node time missing and could not be derived; defaulting to 0.")
        node["time"] = time_idx

    node["text"] = normalize_whitespace(str(node.get("text", "")))

    if t == "H":
        hyp = node.get("hypothesis") or {}
        canonical = normalize_whitespace(str(hyp.get("canonical", node["text"])))
        hid = hyp.get("hid")
        if not hid or not isinstance(hid, str) or len(hid) < 10:
            hid = sha1_hex(canonical)
        node["hypothesis"] = {"canonical": canonical, "hid": hid}
    else:
        node.pop("hypothesis", None)

    return node, warns


def coerce_edge(
    edge: dict[str, Any], messages: list[dict[str, Any]], node_ids: set
) -> tuple[dict[str, Any] | None, list[str]]:
    warns: list[str] = []
    rel = edge.get("relation")
    if rel not in EDGE_RELATIONS_SET:
        return None, [f"Invalid edge relation: {rel}"]
    src = edge.get("src")
    dst = edge.get("dst")
    if src not in node_ids or dst not in node_ids:
        return None, [f"Edge references unknown node_id: src={src}, dst={dst}"]

    support = edge.get("support", [])
    _ok, w = validate_support_quotes(
        messages, support if isinstance(support, list) else []
    )
    warns.extend(w)

    time_idx = edge.get("time")
    if not isinstance(time_idx, int):
        try:
            time_idx = min(int(x["msg_idx"]) for x in support)
        except Exception:
            time_idx = 0
            warns.append("Edge time missing and could not be derived; defaulting to 0.")
        edge["time"] = time_idx

    return edge, warns


async def extract_nodes_pass_a(
    messages: list[dict[str, Any]],
    model: str,
    window: int,
    overlap: int,
    max_nodes_per_window: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    all_nodes: list[dict[str, Any]] = []
    node_counter = 0

    for start, end in iter_windows(len(messages), window, overlap):
        window_text = format_window(messages, start, end)
        user = (
            f"{PASS_A_INSTRUCTIONS}\n\n"
            f"Message window:\n{window_text}\n\n"
            f"Return at most {max_nodes_per_window} nodes."
        )

        if dry_run:
            continue

        out = await llm_json_call_async(
            model=model, system=PASS_A_SYSTEM, user=user, temperature=0.0
        )
        nodes = out.get("nodes", [])
        if not isinstance(nodes, list):
            warnings.append(f"Pass A returned non-list nodes for window {start}-{end}.")
            continue

        for n in nodes[:max_nodes_per_window]:
            if not isinstance(n, dict):
                continue
            node_counter += 1
            node_dict = dict(n)
            node_dict["node_id"] = f"N{node_counter}"
            coerced, w = coerce_node(node_dict, messages)
            warnings.extend([f"PassA[{start}-{end}] {x}" for x in w])
            if coerced is not None:
                all_nodes.append(coerced)

    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for n in all_nodes:
        earliest = n.get("time", 0)
        key = (n.get("type"), n.get("text"), earliest)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(n)

    return deduped, warnings


async def extract_edges_pass_b(
    messages: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    model: str,
    window: int,
    overlap: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    all_edges: list[dict[str, Any]] = []
    node_ids = {n["node_id"] for n in nodes if "node_id" in n}

    for start, end in iter_windows(len(messages), window, overlap):
        window_text = format_window(messages, start, end)
        window_nodes = [
            n
            for n in nodes
            if isinstance(n.get("time"), int) and start <= n["time"] < end
        ]
        if len(window_nodes) < 2:
            continue

        node_list = [
            {
                "node_id": n["node_id"],
                "type": n["type"],
                "text": n["text"],
                "time": n["time"],
            }
            for n in window_nodes
        ]
        user = (
            f"{PASS_B_INSTRUCTIONS}\n\n"
            f"Message window:\n{window_text}\n\n"
            f"Nodes:\n{json.dumps(node_list, ensure_ascii=False, indent=2)}"
        )

        if dry_run:
            continue

        out = await llm_json_call_async(
            model=model, system=PASS_B_SYSTEM, user=user, temperature=0.0
        )
        edges = out.get("edges", [])
        if not isinstance(edges, list):
            warnings.append(f"Pass B returned non-list edges for window {start}-{end}.")
            continue

        for e in edges:
            if not isinstance(e, dict):
                continue
            coerced, w = coerce_edge(e, messages, node_ids)
            warnings.extend([f"PassB[{start}-{end}] {x}" for x in w])
            if coerced is not None:
                all_edges.append(coerced)

    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for e in all_edges:
        key = (e.get("src"), e.get("dst"), e.get("relation"), e.get("time"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    return deduped, warnings


def build_adjacency(
    edges: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    out_edges: dict[str, list[dict[str, Any]]] = {}
    in_edges: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        s = e["src"]
        d = e["dst"]
        out_edges.setdefault(s, []).append(e)
        in_edges.setdefault(d, []).append(e)
    return out_edges, in_edges


def build_index(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    node_by_id = {n["node_id"]: n for n in nodes if "node_id" in n}
    out_edges: dict[str, list[dict[str, Any]]] = {}
    in_edges: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        s, d = e.get("src"), e.get("dst")
        if s is None or d is None:
            continue
        out_edges.setdefault(s, []).append(e)
        in_edges.setdefault(d, []).append(e)
    return node_by_id, out_edges, in_edges


def node_time(n: dict[str, Any]) -> int:
    t = n.get("time", 0)
    try:
        return int(t)
    except Exception:
        return 0


def earliest_time_of_type(nodes: list[dict[str, Any]], t: str) -> int | None:
    ts = [node_time(n) for n in nodes if n.get("type") == t]
    return min(ts) if ts else None


def earliest_evidence_used_time(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> int | None:
    node_by_id, out_edges, in_edges = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    candidate_times: list[int] = []
    for nid, n in node_by_id.items():
        if n.get("type") != "E":
            continue
        candidate_times.extend(
            node_time(n)
            for e in out_edges.get(nid, [])
            if e.get("relation") == "uses"
            and node_type_map.get(e.get("dst")) in {"J", "U"}
        )
        candidate_times.extend(
            node_time(n)
            for e in in_edges.get(nid, [])
            if e.get("relation") == "uses"
            and node_type_map.get(e.get("src")) in {"J", "U"}
        )
    return min(candidate_times) if candidate_times else None


def metric_update_grounding_rate(nodes: list, edges: list) -> dict[str, Any]:
    node_type_map = {n["node_id"]: n["type"] for n in nodes}
    in_edges = build_adjacency(edges)[1]
    u_nodes = [n for n in nodes if n["type"] == "U"]
    if not u_nodes:
        return {"value": None, "numerator": 0, "denominator": 0}
    grounded = 0
    for u in u_nodes:
        for e in in_edges.get(u["node_id"], []):
            if e["relation"] == "uses" and node_type_map.get(e["src"]) == "E":
                grounded += 1
                break
    return {
        "value": grounded / len(u_nodes),
        "numerator": grounded,
        "denominator": len(u_nodes),
    }


def metric_orphan_evidence_rate(nodes: list, edges: list) -> dict[str, Any]:
    out_edges = build_adjacency(edges)[0]
    e_nodes = [n for n in nodes if n["type"] == "E"]
    if not e_nodes:
        return {"value": None, "numerator": 0, "denominator": 0}
    orphan = 0
    for en in e_nodes:
        outs = out_edges.get(en["node_id"], [])
        used = any(ed["relation"] == "uses" for ed in outs)
        if not used:
            orphan += 1
    return {
        "value": orphan / len(e_nodes),
        "numerator": orphan,
        "denominator": len(e_nodes),
    }


def metric_refute_neglect_rate(nodes: list, edges: list) -> dict[str, Any]:
    node_type_map = {n["node_id"]: n["type"] for n in nodes}
    out_edges, in_edges = build_adjacency(edges)
    refuting = [
        e
        for e in edges
        if e["relation"] in {"refutes", "contradicts"}
        and node_type_map.get(e["src"]) == "E"
        and node_type_map.get(e["dst"]) == "H"
    ]
    if not refuting:
        return {"value": None, "numerator": 0, "denominator": 0}
    neglected = 0
    for e in refuting:
        ev = e["src"]
        outs = out_edges.get(ev, [])
        ins = in_edges.get(ev, [])
        linked = False
        for ed in outs:
            if ed["relation"] == "uses" and node_type_map.get(ed["dst"]) in {"J", "U"}:
                linked = True
                break
        if not linked:
            for ed in ins:
                if ed["relation"] == "uses" and node_type_map.get(ed["src"]) in {
                    "J",
                    "U",
                }:
                    linked = True
                    break
        if not linked:
            neglected += 1
    return {
        "value": neglected / len(refuting),
        "numerator": neglected,
        "denominator": len(refuting),
    }


def metric_loop_density(nodes: list, edges: list) -> dict[str, Any]:
    node_type_map = {n["node_id"]: n["type"] for n in nodes}
    out_edges = build_adjacency(edges)[0]
    Hs = [n["node_id"] for n in nodes if n["type"] == "H"]
    Js = [n["node_id"] for n in nodes if n["type"] == "J"]
    Us = [n["node_id"] for n in nodes if n["type"] == "U"]

    def outs_by_rel(src: str, rel: str) -> list[str]:
        return [e["dst"] for e in out_edges.get(src, []) if e["relation"] == rel]

    def has_uses(a: str, b: str) -> bool:
        for e in out_edges.get(a, []):
            if e["relation"] == "uses" and e["dst"] == b:
                return True
        for e in out_edges.get(b, []):
            if e["relation"] == "uses" and e["dst"] == a:
                return True
        return False

    motif_count = 0
    for h in Hs:
        for t in outs_by_rel(h, "tests"):
            if node_type_map.get(t) != "T":
                continue
            for ev in outs_by_rel(t, "observes"):
                if node_type_map.get(ev) != "E":
                    continue
                for j in Js:
                    if has_uses(j, ev):
                        for u in Us:
                            if has_uses(u, ev):
                                motif_count += 1
    denom = max(1, len(nodes))
    return {
        "value": motif_count / denom,
        "count": motif_count,
        "denominator_nodes": len(nodes),
    }


def metric_hypothesis_switch_without_eval(nodes: list, edges: list) -> dict[str, Any]:
    node_by_id = {n["node_id"]: n for n in nodes}
    node_type_map = {nid: n["type"] for nid, n in node_by_id.items()}
    out_edges_map, in_edges_map = build_adjacency(edges)

    switches: list[tuple[str, str, int]] = []
    for e in edges:
        if e["relation"] == "updates_to":
            s, d = e["src"], e["dst"]
            if node_type_map.get(s) == "H" and node_type_map.get(d) == "H":
                switches.append((s, d, int(e.get("time", 0))))
            elif node_type_map.get(s) == "U" and node_type_map.get(d) == "H":
                for ie in in_edges_map.get(s, []):
                    if node_type_map.get(ie["src"]) == "H":
                        switches.append((ie["src"], d, int(e.get("time", 0))))
                        break

    if not switches:
        return {"value": None, "numerator": 0, "denominator": 0}

    def _node_time(nid: str) -> int:
        t = node_by_id.get(nid, {}).get("time", 0)
        return int(t) if isinstance(t, int) else 0

    def has_nearby_eval(switch_time: int, target_h: str) -> bool:
        for n in nodes:
            if n["type"] not in {"J", "V"}:
                continue
            t = _node_time(n["node_id"])
            if switch_time - 2 <= t <= switch_time:
                nid = n["node_id"]
                for e in out_edges_map.get(nid, []):
                    if e["dst"] == target_h:
                        return True
                for e in in_edges_map.get(nid, []):
                    if e["src"] == target_h:
                        return True
        return False

    denom = len(switches)
    bad = sum(
        1
        for hold, hnew, t in switches
        if not (has_nearby_eval(t, hold) or has_nearby_eval(t, hnew))
    )
    return {"value": bad / denom, "numerator": bad, "denominator": denom}


def metric_alternative_coverage(nodes: list, edges: list) -> dict[str, Any]:
    h_nodes = [n for n in nodes if n["type"] == "H"]
    if not h_nodes:
        return {"value": None, "numerator": 0, "denominator": 0}
    h_ids = {n["node_id"] for n in h_nodes}
    h_with_alt: set[str] = set()
    for e in edges:
        if e["relation"] == "competes_with":
            if e["src"] in h_ids:
                h_with_alt.add(e["src"])
            if e["dst"] in h_ids:
                h_with_alt.add(e["dst"])
    covered = len(h_with_alt)
    return {
        "value": covered / len(h_nodes),
        "numerator": covered,
        "denominator": len(h_nodes),
    }


def metric_unlinked_test_evidence_rate(nodes: list, edges: list) -> dict[str, Any]:
    te_nodes = [n for n in nodes if n["type"] in {"T", "E"}]
    if not te_nodes:
        return {"value": None, "numerator": 0, "denominator": 0}
    h_ids = {n["node_id"] for n in nodes if n["type"] == "H"}
    edge_pairs: set[tuple[str, str]] = {(e["src"], e["dst"]) for e in edges}
    unlinked = 0
    for te in te_nodes:
        nid = te["node_id"]
        linked = any((nid, h) in edge_pairs or (h, nid) in edge_pairs for h in h_ids)
        if not linked:
            unlinked += 1
    return {
        "value": unlinked / len(te_nodes),
        "numerator": unlinked,
        "denominator": len(te_nodes),
    }


def antipattern_hypothesis_hopping(nodes: list, edges: list) -> dict[str, Any]:
    node_type_map = {n["node_id"]: n["type"] for n in nodes}
    h_transitions: dict[str, list[tuple[str, int]]] = {}
    for e in edges:
        if e["relation"] == "updates_to":
            s, d = e["src"], e["dst"]
            if node_type_map.get(s) == "H" and node_type_map.get(d) == "H":
                h_transitions.setdefault(s, []).append((d, int(e.get("time", 0))))

    out_edges_map, in_edges_map = build_adjacency(edges)

    def has_eval_between(h1: str, h2: str, t: int) -> bool:
        for n in nodes:
            if n["type"] not in {"J", "U"}:
                continue
            nt = int(n.get("time", 0))
            if abs(nt - t) > 2:
                continue
            nid = n["node_id"]
            for ed in out_edges_map.get(nid, []):
                if ed["dst"] in {h1, h2}:
                    return True
            for ed in in_edges_map.get(nid, []):
                if ed["src"] in {h1, h2}:
                    return True
        return False

    bare_next: dict[str, list[str]] = {}
    for h1, nexts in h_transitions.items():
        for h2, t in nexts:
            if not has_eval_between(h1, h2, t):
                bare_next.setdefault(h1, []).append(h2)

    if not bare_next:
        return {"detected": False, "hop_chain_count": 0, "longest_hop_chain": 0}

    def chain_len(h: str, visited: set) -> int:
        if h in visited:
            return 0
        nexts = bare_next.get(h, [])
        if not nexts:
            return 1
        visited.add(h)
        best = 1 + max(chain_len(nx, visited) for nx in nexts)
        visited.discard(h)
        return best

    reached = {h2 for nexts_list in bare_next.values() for h2 in nexts_list}
    starts = [h for h in bare_next if h not in reached]
    longest = 0
    hop_chains = 0
    for s in starts:
        length = chain_len(s, set())
        if length >= 3:
            hop_chains += 1
            longest = max(longest, length)
    return {
        "detected": hop_chains > 0,
        "hop_chain_count": hop_chains,
        "longest_hop_chain": longest,
    }


def antipattern_no_hypothesis(nodes: list) -> dict[str, Any]:
    h_count = sum(1 for n in nodes if n["type"] == "H")
    return {"detected": h_count == 0, "hypothesis_count": h_count}


def compute_metrics(nodes: list, edges: list) -> dict[str, Any]:
    return {
        "update_grounding_rate": metric_update_grounding_rate(nodes, edges),
        "orphan_evidence_rate": metric_orphan_evidence_rate(nodes, edges),
        "refute_neglect_rate": metric_refute_neglect_rate(nodes, edges),
        "loop_density": metric_loop_density(nodes, edges),
        "hypothesis_switch_without_eval_rate": metric_hypothesis_switch_without_eval(
            nodes, edges
        ),
        "alternative_coverage": metric_alternative_coverage(nodes, edges),
        "unlinked_test_evidence_rate": metric_unlinked_test_evidence_rate(nodes, edges),
        "antipatterns": {
            "hypothesis_hopping": antipattern_hypothesis_hopping(nodes, edges),
            "no_hypothesis": antipattern_no_hypothesis(nodes),
        },
        "counts": {
            "nodes_total": len(nodes),
            "edges_total": len(edges),
            "nodes_by_type": {
                t: sum(1 for n in nodes if n["type"] == t)
                for t in sorted(NODE_TYPES_SET)
            },
            "edges_by_relation": {
                r: sum(1 for e in edges if e["relation"] == r)
                for r in sorted(EDGE_RELATIONS_SET)
            },
        },
    }


def _nodes_of_type(ntype: str, node_type_map: dict[str, str]) -> list[str]:
    return [nid for nid, t in node_type_map.items() if t == ntype]


def _has_edge(
    src: str, dst: str, relation: str, out_edges: dict[str, list[dict[str, Any]]]
) -> bool:
    for e in out_edges.get(src, []):
        if e.get("relation") == relation and e.get("dst") == dst:
            return True
    return False


def _has_uses_link(a: str, b: str, out_edges: dict[str, list[dict[str, Any]]]) -> bool:
    return _has_edge(a, b, "uses", out_edges) or _has_edge(b, a, "uses", out_edges)


def _neighbours(
    src: str,
    relation: str,
    dst_type: str,
    out_edges: dict[str, list[dict[str, Any]]],
    node_type_map: dict[str, str],
) -> list[str]:
    return [
        e["dst"]
        for e in out_edges.get(src, [])
        if e.get("relation") == relation and node_type_map.get(e.get("dst")) == dst_type
    ]


def _match_popperian(node_type_map, _node_by_id, out_edges):
    count = 0
    for h1 in _nodes_of_type("H", node_type_map):
        matched = False
        for t in _neighbours(h1, "tests", "T", out_edges, node_type_map):
            for ev in _neighbours(t, "observes", "E", out_edges, node_type_map):
                for j in _nodes_of_type("J", node_type_map):
                    if not _has_uses_link(ev, j, out_edges):
                        continue
                    for u in _nodes_of_type("U", node_type_map):
                        if not _has_uses_link(ev, u, out_edges):
                            continue
                        for h2 in _neighbours(
                            u, "updates_to", "H", out_edges, node_type_map
                        ):
                            if h2 != h1:
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                break
        if matched:
            count += 1
    return count


def _match_ml_make_it_work(node_type_map, _node_by_id, out_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        matched = False
        for t1 in _neighbours(h, "tests", "T", out_edges, node_type_map):
            for ev in _neighbours(t1, "observes", "E", out_edges, node_type_map):
                for j in _nodes_of_type("J", node_type_map):
                    if not _has_uses_link(ev, j, out_edges):
                        continue
                    for u in _nodes_of_type("U", node_type_map):
                        u_to_Ts = _neighbours(
                            u, "updates_to", "T", out_edges, node_type_map
                        )
                        if not u_to_Ts:
                            continue
                        u_to_Hs = _neighbours(
                            u, "updates_to", "H", out_edges, node_type_map
                        )
                        if u_to_Hs:
                            continue
                        matched = True
                        break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                break
        if matched:
            count += 1
    return count


def _match_exploratory_to_confirmatory(node_type_map, node_by_id, out_edges):
    count = 0
    for t0 in _nodes_of_type("T", node_type_map):
        t0_time = node_time(node_by_id[t0])
        evs = _neighbours(t0, "observes", "E", out_edges, node_type_map)
        if not evs:
            continue
        for h1 in _nodes_of_type("H", node_type_map):
            if node_time(node_by_id[h1]) <= t0_time:
                continue
            if _neighbours(h1, "tests", "T", out_edges, node_type_map):
                count += 1
                break
    return count


def _match_bayesian(node_type_map, _node_by_id, out_edges):
    count = 0
    Hs = _nodes_of_type("H", node_type_map)
    for i, h1 in enumerate(Hs):
        for h2 in Hs[i + 1 :]:
            competes = _has_edge(h1, h2, "competes_with", out_edges) or _has_edge(
                h2, h1, "competes_with", out_edges
            )
            if not competes:
                continue
            if _neighbours(h1, "tests", "T", out_edges, node_type_map) and _neighbours(
                h2, "tests", "T", out_edges, node_type_map
            ):
                count += 1
    return count


def _match_abductive(node_type_map, node_by_id, out_edges):
    count = 0
    for e0 in _nodes_of_type("E", node_type_map):
        e0_time = node_time(node_by_id[e0])
        matched = False
        for j0 in _nodes_of_type("J", node_type_map):
            if not _has_uses_link(e0, j0, out_edges):
                continue
            for h1 in _nodes_of_type("H", node_type_map):
                if node_time(node_by_id[h1]) <= e0_time:
                    continue
                if _neighbours(h1, "tests", "T", out_edges, node_type_map):
                    matched = True
                    break
            if matched:
                break
        if matched:
            count += 1
    return count


def _match_triangulation(node_type_map, _node_by_id, out_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        test_targets = _neighbours(h, "tests", "T", out_edges, node_type_map)
        test_with_ev = [
            t
            for t in test_targets
            if _neighbours(t, "observes", "E", out_edges, node_type_map)
        ]
        if len(set(test_with_ev)) >= 3:
            count += 1
    return count


def _match_preregistered(node_type_map, node_by_id, out_edges):
    count = 0
    Cs = _nodes_of_type("C", node_type_map)
    Es = _nodes_of_type("E", node_type_map)
    for h in _nodes_of_type("H", node_type_map):
        matched = False
        for t in _neighbours(h, "tests", "T", out_edges, node_type_map):
            t_time = node_time(node_by_id[t])
            for c in Cs:
                c_time = node_time(node_by_id[c])
                if c_time > t_time:
                    continue
                for ev in Es:
                    if node_time(node_by_id[ev]) > c_time:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if matched:
            count += 1
    return count


def _match_active_learning(node_type_map, _node_by_id, out_edges):
    count = 0
    for u in _nodes_of_type("U", node_type_map):
        for t in _neighbours(u, "updates_to", "T", out_edges, node_type_map):
            if _neighbours(t, "observes", "E", out_edges, node_type_map):
                count += 1
                break
    return count


_SUBGRAPH_MATCHERS = {
    "popperian_falsification": _match_popperian,
    "ml_make_it_work": _match_ml_make_it_work,
    "exploratory_to_confirmatory": _match_exploratory_to_confirmatory,
    "bayesian_belief_updating": _match_bayesian,
    "abductive": _match_abductive,
    "triangulation": _match_triangulation,
    "preregistered": _match_preregistered,
    "active_learning": _match_active_learning,
}


def detect_subgraphs_local(nodes: list, edges: list) -> dict[str, int]:
    node_by_id, out_edges, _in = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    return {
        name: matcher(node_type_map, node_by_id, out_edges)
        for name, matcher in _SUBGRAPH_MATCHERS.items()
    }


def _typed_edge_exists(src_type, dst_type, relation, node_by_id, out_edges):
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    for nid, ntype in node_type_map.items():
        if ntype != src_type:
            continue
        for e in out_edges.get(nid, []):
            if (
                e.get("relation") == relation
                and node_type_map.get(e.get("dst")) == dst_type
            ):
                return True
    return False


def _fan_out_global(src_type, dst_type, relation, node_by_id, out_edges):
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    best = 0
    for nid, ntype in node_type_map.items():
        if ntype != src_type:
            continue
        targets = {
            e.get("dst")
            for e in out_edges.get(nid, [])
            if e.get("relation") == relation
            and node_type_map.get(e.get("dst")) == dst_type
        }
        best = max(best, len(targets))
    return best


def detect_subgraphs_global(nodes: list, edges: list) -> dict[str, int]:
    node_by_id, out_edges, _in_edges = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}

    n_H = sum(1 for t in node_type_map.values() if t == "H")
    n_T = sum(1 for t in node_type_map.values() if t == "T")

    t_first_H = earliest_time_of_type(nodes, "H")
    t_first_T = earliest_time_of_type(nodes, "T")
    t_first_E = earliest_time_of_type(nodes, "E")
    t_first_C = earliest_time_of_type(nodes, "C")

    has_HT_tests = _typed_edge_exists("H", "T", "tests", node_by_id, out_edges)
    has_TE_observes = _typed_edge_exists("T", "E", "observes", node_by_id, out_edges)
    has_UH_updates = _typed_edge_exists("U", "H", "updates_to", node_by_id, out_edges)
    has_UT_updates = _typed_edge_exists("U", "T", "updates_to", node_by_id, out_edges)
    has_HH_competes = _typed_edge_exists(
        "H", "H", "competes_with", node_by_id, out_edges
    )
    fan_out_H_T = _fan_out_global("H", "T", "tests", node_by_id, out_edges)
    has_EJ_uses = _typed_edge_exists(
        "E", "J", "uses", node_by_id, out_edges
    ) or _typed_edge_exists("J", "E", "uses", node_by_id, out_edges)

    results: dict[str, int] = {}
    results["popperian_falsification"] = int(
        has_HT_tests and has_TE_observes and has_UH_updates and n_H >= 2
    )
    results["ml_make_it_work"] = int(
        n_H <= 1
        and n_T >= 2
        and has_HT_tests
        and has_TE_observes
        and has_EJ_uses
        and not has_UH_updates
    )
    results["exploratory_to_confirmatory"] = int(
        t_first_T is not None
        and t_first_H is not None
        and t_first_T < t_first_H
        and has_HT_tests
    )
    results["bayesian_belief_updating"] = int(n_H >= 2 and has_HH_competes)
    results["abductive"] = int(
        t_first_E is not None and t_first_H is not None and t_first_E < t_first_H
    )
    results["triangulation"] = int(fan_out_H_T >= 3)
    results["preregistered"] = int(
        t_first_C is not None and t_first_E is not None and t_first_C < t_first_E
    )
    results["active_learning"] = int(has_UT_updates and has_TE_observes)
    return results


def _ap_untested_hypothesis(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if not tested:
            count += 1
    return count


def _ap_evidence_ignored(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for ev in _nodes_of_type("E", node_type_map):
        used = any(
            e.get("relation") == "uses"
            and node_type_map.get(e.get("dst")) in {"J", "U"}
            for e in out_edges.get(ev, [])
        )
        if not used:
            used = any(
                e.get("relation") == "uses"
                and node_type_map.get(e.get("src")) in {"J", "U"}
                for e in in_edges.get(ev, [])
            )
        if not used:
            count += 1
    return count


def _ap_judgment_without_evidence(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for j in _nodes_of_type("J", node_type_map):
        has_e = any(
            e.get("relation") == "uses" and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(j, [])
        )
        if not has_e:
            has_e = any(
                e.get("relation") == "uses" and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(j, [])
            )
        if not has_e:
            count += 1
    return count


def _ap_dead_end_update(node_type_map, _node_by_id, out_edges, _in_edges):
    return sum(1 for u in _nodes_of_type("U", node_type_map) if not out_edges.get(u))


def _ap_unresolved_contradiction(node_type_map, node_by_id, out_edges, in_edges):
    count = 0
    for nid in list(node_type_map):
        for e in out_edges.get(nid, []):
            if e.get("relation") != "contradicts":
                continue
            dst = e.get("dst")
            if dst is None or node_type_map.get(dst) != "H":
                continue
            h_t = node_time(node_by_id.get(dst, {}))
            resolved = False
            for ie in in_edges.get(dst, []):
                src = ie.get("src")
                if (
                    ie.get("relation") == "updates_to"
                    and node_type_map.get(src) == "U"
                    and node_time(node_by_id.get(src, {})) >= h_t
                ):
                    resolved = True
                    break
            if not resolved:
                for oe in out_edges.get(dst, []):
                    if oe.get("relation") == "competes_with":
                        resolved = True
                        break
                if not resolved:
                    for ie in in_edges.get(dst, []):
                        if ie.get("relation") == "competes_with":
                            resolved = True
                            break
            if not resolved:
                count += 1
    return count


def _ap_hypothesis_to_commitment_shortcut(
    node_type_map, _node_by_id, out_edges, in_edges
):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        links_c = any(
            e.get("relation") == "uses" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(h, [])
        )
        if not links_c:
            links_c = any(
                e.get("relation") == "uses" and node_type_map.get(e.get("src")) == "C"
                for e in in_edges.get(h, [])
            )
        if not links_c:
            continue
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if not tested:
            count += 1
    return count


def _ap_test_without_evidence(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for t in _nodes_of_type("T", node_type_map):
        has_ev = any(
            e.get("relation") in ("observes", "tests")
            and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(t, [])
        )
        if not has_ev:
            has_ev = any(
                e.get("relation") in ("observes", "tests")
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(t, [])
            )
        if not has_ev:
            count += 1
    return count


def _ap_no_belief_revision(node_type_map, _node_by_id, _out_edges, _in_edges):
    return 1 if len(_nodes_of_type("U", node_type_map)) == 0 else 0


def _ap_orphan_evidence(node_type_map, _node_by_id, out_edges, in_edges):
    return sum(
        1
        for ev in _nodes_of_type("E", node_type_map)
        if not out_edges.get(ev) and not in_edges.get(ev)
    )


def _ap_confirmation_only(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        committed = any(
            e.get("relation") == "uses" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(h, [])
        )
        if not committed:
            committed = any(
                e.get("relation") == "uses" and node_type_map.get(e.get("src")) == "C"
                for e in in_edges.get(h, [])
            )
        if not committed:
            continue
        has_support = False
        for ie in in_edges.get(h, []):
            src = ie.get("src")
            if ie.get("relation") == "uses" and node_type_map.get(src) == "J":
                for je in in_edges.get(src, []):
                    if (
                        je.get("relation") == "uses"
                        and node_type_map.get(je.get("src")) == "E"
                    ):
                        has_support = True
                        break
            if ie.get("relation") == "uses" and node_type_map.get(src) == "E":
                has_support = True
            if has_support:
                break
        if not has_support:
            continue
        has_contradiction = any(
            ie.get("relation") == "contradicts" for ie in in_edges.get(h, [])
        )
        if not has_contradiction:
            has_contradiction = any(
                oe.get("relation") == "contradicts" for oe in out_edges.get(h, [])
            )
        if not has_contradiction:
            count += 1
    return count


_ANTIPATTERN_MATCHERS = {
    "untested_hypothesis": _ap_untested_hypothesis,
    "evidence_ignored": _ap_evidence_ignored,
    "judgment_without_evidence": _ap_judgment_without_evidence,
    "dead_end_update": _ap_dead_end_update,
    "unresolved_contradiction": _ap_unresolved_contradiction,
    "hypothesis_to_commitment_shortcut": _ap_hypothesis_to_commitment_shortcut,
    "test_without_evidence": _ap_test_without_evidence,
    "no_belief_revision": _ap_no_belief_revision,
    "orphan_evidence": _ap_orphan_evidence,
    "confirmation_only": _ap_confirmation_only,
}


def detect_antipatterns_local(nodes: list, edges: list) -> dict[str, int]:
    node_by_id, out_edges, in_edges = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    return {
        name: matcher(node_type_map, node_by_id, out_edges, in_edges)
        for name, matcher in _ANTIPATTERN_MATCHERS.items()
    }


def detect_antipatterns_global(
    nodes: list, edges: list
) -> tuple[dict[str, bool], dict[str, int]]:
    node_by_id, out_edges, in_edges = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}

    n_H = sum(1 for t in node_type_map.values() if t == "H")
    n_T = sum(1 for t in node_type_map.values() if t == "T")
    n_E = sum(1 for t in node_type_map.values() if t == "E")
    n_U = sum(1 for t in node_type_map.values() if t == "U")

    h_tested = 0
    for h in _nodes_of_type("H", node_type_map):
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if tested:
            h_tested += 1

    e_used = 0
    for ev in _nodes_of_type("E", node_type_map):
        used = any(
            e.get("relation") == "uses"
            and node_type_map.get(e.get("dst")) in {"J", "U"}
            for e in out_edges.get(ev, [])
        )
        if not used:
            used = any(
                e.get("relation") == "uses"
                and node_type_map.get(e.get("src")) in {"J", "U"}
                for e in in_edges.get(ev, [])
            )
        if used:
            e_used += 1

    t_with_ev = 0
    for t in _nodes_of_type("T", node_type_map):
        has_ev = any(
            e.get("relation") in ("observes", "tests")
            and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(t, [])
        )
        if not has_ev:
            has_ev = any(
                e.get("relation") in ("observes", "tests")
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(t, [])
            )
        if has_ev:
            t_with_ev += 1

    n_contradicts = 0
    n_unresolved = 0
    for nid in node_type_map:
        for e in out_edges.get(nid, []):
            if e.get("relation") != "contradicts":
                continue
            dst = e.get("dst")
            if node_type_map.get(dst) != "H":
                continue
            n_contradicts += 1
            resolved = any(
                ie.get("relation") in ("updates_to", "competes_with")
                for ie in in_edges.get(dst, [])
            )
            if not resolved:
                resolved = any(
                    oe.get("relation") == "competes_with"
                    for oe in out_edges.get(dst, [])
                )
            if not resolved:
                n_unresolved += 1

    h_to_c_untested = 0
    for h in _nodes_of_type("H", node_type_map):
        links_c = any(
            e.get("relation") == "uses" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(h, [])
        )
        if not links_c:
            links_c = any(
                e.get("relation") == "uses" and node_type_map.get(e.get("src")) == "C"
                for e in in_edges.get(h, [])
            )
        if not links_c:
            continue
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if not tested:
            h_to_c_untested += 1

    e_orphan = sum(
        1
        for ev in _nodes_of_type("E", node_type_map)
        if not out_edges.get(ev) and not in_edges.get(ev)
    )

    untested_h_rate = (n_H - h_tested) / n_H if n_H else 0.0
    evidence_ignored_rate = (n_E - e_used) / n_E if n_E else 0.0
    t_no_ev_rate = (n_T - t_with_ev) / n_T if n_T else 0.0
    unresolved_rate = n_unresolved / n_contradicts if n_contradicts else 0.0

    j_without_e = 0
    for j in _nodes_of_type("J", node_type_map):
        has_e = any(
            e.get("relation") == "uses" and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(j, [])
        )
        if not has_e:
            has_e = any(
                e.get("relation") == "uses" and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(j, [])
            )
        if not has_e:
            j_without_e += 1

    dead_u = sum(1 for u in _nodes_of_type("U", node_type_map) if not out_edges.get(u))
    ap_local_counts = detect_antipatterns_local(nodes, edges)

    binary: dict[str, bool] = {
        "untested_hypothesis": untested_h_rate > 0.25,
        "evidence_ignored": evidence_ignored_rate > 0.25,
        "judgment_without_evidence": j_without_e > 0,
        "dead_end_update": dead_u > 0,
        "unresolved_contradiction": unresolved_rate > 0.5 and n_contradicts > 0,
        "hypothesis_to_commitment_shortcut": h_to_c_untested > 0,
        "test_without_evidence": t_no_ev_rate > 0.5,
        "no_belief_revision": n_U == 0,
        "orphan_evidence": e_orphan > 0,
        "confirmation_only": ap_local_counts.get("confirmation_only", 0) > 0,
    }
    counts: dict[str, int] = {
        "untested_hypothesis": n_H - h_tested,
        "evidence_ignored": n_E - e_used,
        "judgment_without_evidence": j_without_e,
        "dead_end_update": dead_u,
        "unresolved_contradiction": n_unresolved,
        "hypothesis_to_commitment_shortcut": h_to_c_untested,
        "test_without_evidence": n_T - t_with_ev,
        "no_belief_revision": 1 if n_U == 0 else 0,
        "orphan_evidence": e_orphan,
        "confirmation_only": ap_local_counts.get("confirmation_only", 0),
    }
    return binary, counts


def count_nodes_by_type_summary(nodes: list) -> dict[str, int]:
    out = dict.fromkeys(NODE_TYPES, 0)
    for n in nodes:
        t = n.get("type")
        if t in out:
            out[t] += 1
    return out


def count_nodes_by_type_annotated(nodes: list) -> dict[str, int]:
    counts = {f"n_{t}": 0 for t in NODE_TYPES}
    for n in nodes:
        t = n.get("type")
        key = f"n_{t}"
        if key in counts:
            counts[key] += 1
    counts["nodes_total"] = len(nodes)
    return counts


def count_edges_by_relation(edges: list) -> dict[str, int]:
    out = dict.fromkeys(EDGE_RELATIONS, 0)
    for e in edges:
        r = e.get("relation")
        if r in out:
            out[r] += 1
    return out


def workflow_completeness(nodes: list) -> float:
    present = dict.fromkeys(["H", "T", "E", "J", "U"], False)
    for n in nodes:
        if n.get("type") in present:
            present[n["type"]] = True
    return sum(1 for v in present.values() if v) / 5.0


def scientificness_score(summary_row: dict[str, Any]) -> float | None:
    ld = safe_float(summary_row.get("loop_density"))
    ugr = safe_float(summary_row.get("update_grounding_rate"))
    oer = safe_float(summary_row.get("orphan_evidence_rate"))
    rnr = safe_float(summary_row.get("refute_neglect_rate"))
    pc = summary_row.get("premature_commit")
    if ld is None or ugr is None or oer is None:
        return None
    ld_norm = ld / (ld + 0.1) if ld >= 0 else 0.0
    inv_oer = (1.0 - oer) if oer is not None else 0.0
    inv_rnr = (1.0 - rnr) if rnr is not None else 0.0
    inv_pc = 0.0 if pc is None else (0.0 if pc else 1.0)
    return 0.30 * ld_norm + 0.25 * ugr + 0.20 * inv_oer + 0.15 * inv_rnr + 0.10 * inv_pc


def detect_motifs_HTEJU(
    nodes: list, edges: list
) -> list[tuple[int, int, tuple[str, str, str, str, str]]]:
    node_by_id, out_edges, _in = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}

    def outs_by_rel(src, rel):
        return [
            e["dst"]
            for e in out_edges.get(src, [])
            if e.get("relation") == rel and "dst" in e
        ]

    Js = [nid for nid, t in node_type_map.items() if t == "J"]
    Us = [nid for nid, t in node_type_map.items() if t == "U"]
    Hs = [nid for nid, t in node_type_map.items() if t == "H"]

    motifs = []
    for h in Hs:
        for t in outs_by_rel(h, "tests"):
            if node_type_map.get(t) != "T":
                continue
            for ev in outs_by_rel(t, "observes"):
                if node_type_map.get(ev) != "E":
                    continue
                for j in Js:
                    if not _has_uses_link(j, ev, out_edges):
                        continue
                    for u in Us:
                        if not _has_uses_link(u, ev, out_edges):
                            continue
                        times = [
                            node_time(node_by_id[x])
                            for x in [h, t, ev, j, u]
                            if x in node_by_id
                        ]
                        if times:
                            motifs.append((min(times), max(times), (h, t, ev, j, u)))

    seen: set[tuple] = set()
    dedup = []
    for m in motifs:
        if m[2] in seen:
            continue
        seen.add(m[2])
        dedup.append(m)
    return dedup


def summarize_trace(doc: dict[str, Any], file_path: Path) -> dict[str, Any]:
    nodes = doc.get("nodes", []) or []
    edges = doc.get("edges", []) or []
    metrics = doc.get("metrics", {}) or {}

    tC = earliest_time_of_type(nodes, "C")
    tE_used = earliest_evidence_used_time(nodes, edges)
    premature = None
    if tC is not None and tE_used is not None:
        premature = tC < tE_used

    ncounts = count_nodes_by_type_summary(nodes)
    ecounts = count_edges_by_relation(edges)

    row: dict[str, Any] = {
        "file": file_path.name,
        "input_file": doc.get("input_file", ""),
        "model": doc.get("provenance", {}).get("model", ""),
        "nodes_total": len(nodes),
        "edges_total": len(edges),
        "workflow_completeness": workflow_completeness(nodes),
        "t_commit": tC,
        "t_evidence_used": tE_used,
        "premature_commit": premature,
        "loop_density": safe_float(metrics.get("loop_density", {}).get("value")),
        "update_grounding_rate": safe_float(
            metrics.get("update_grounding_rate", {}).get("value")
        ),
        "orphan_evidence_rate": safe_float(
            metrics.get("orphan_evidence_rate", {}).get("value")
        ),
        "refute_neglect_rate": safe_float(
            metrics.get("refute_neglect_rate", {}).get("value")
        ),
        "hypothesis_switch_without_eval_rate": safe_float(
            metrics.get("hypothesis_switch_without_eval_rate", {}).get("value")
        ),
    }
    for t, c in ncounts.items():
        row[f"n_{t}"] = c
    for r, c in ecounts.items():
        row[f"e_{r}"] = c

    row["scientificness_score"] = scientificness_score(row)

    sg_local = detect_subgraphs_local(nodes, edges)
    sg_global = detect_subgraphs_global(nodes, edges)
    for sg_name in SUBGRAPH_NAMES:
        row[f"sg_local_{sg_name}"] = sg_local.get(sg_name, 0) > 0
        row[f"sg_local_count_{sg_name}"] = sg_local.get(sg_name, 0)
        row[f"sg_global_{sg_name}"] = sg_global.get(sg_name, 0) > 0
        row[f"sg_global_count_{sg_name}"] = sg_global.get(sg_name, 0)

    ap_local = detect_antipatterns_local(nodes, edges)
    ap_global_binary, ap_global_counts = detect_antipatterns_global(nodes, edges)
    for ap_name in ANTIPATTERN_NAMES:
        row[f"ap_local_{ap_name}"] = ap_local.get(ap_name, 0) > 0
        row[f"ap_local_count_{ap_name}"] = ap_local.get(ap_name, 0)
        row[f"ap_global_{ap_name}"] = ap_global_binary.get(ap_name, False)
        row[f"ap_global_count_{ap_name}"] = ap_global_counts.get(ap_name, 0)

    for fam_name, members in ANTIPATTERN_FAMILIES.items():
        row[f"ap_family_local_{fam_name}"] = any(
            ap_local.get(m, 0) > 0 for m in members
        )
        row[f"ap_family_local_count_{fam_name}"] = sum(
            ap_local.get(m, 0) for m in members
        )
        row[f"ap_family_global_{fam_name}"] = any(
            ap_global_binary.get(m, False) for m in members
        )
        row[f"ap_family_global_count_{fam_name}"] = sum(
            ap_global_counts.get(m, 0) for m in members
        )

    for fam_name, members in SUBGRAPH_FAMILIES.items():
        row[f"sg_family_local_{fam_name}"] = any(
            sg_local.get(m, 0) > 0 for m in members
        )
        row[f"sg_family_local_count_{fam_name}"] = sum(
            sg_local.get(m, 0) for m in members
        )
        row[f"sg_family_global_{fam_name}"] = any(
            sg_global.get(m, 0) > 0 for m in members
        )
        row[f"sg_family_global_count_{fam_name}"] = sum(
            sg_global.get(m, 0) for m in members
        )

    return row


def _save_fig(path: Path) -> None:
    if not HAS_MPL:
        return
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_stacked_node_type_fractions(
    rows: list[dict[str, Any]], out_path: Path
) -> None:
    if not HAS_MPL or not rows:
        return

    def sort_key(r):
        s = safe_float(r.get("scientificness_score"))
        return -s if s is not None else 1e9

    rows2 = sorted(rows, key=sort_key)
    fractions_by_type = {t: [] for t in NODE_TYPES}
    labels = []
    max_label_len = 20
    for r in rows2:
        total = max(1, int(r.get("nodes_total", 1)))
        raw = Path(r.get("file", "")).stem
        short = raw if len(raw) <= max_label_len else raw[: max_label_len - 1] + "…"
        labels.append(short)
        for t in NODE_TYPES:
            fractions_by_type[t].append(float(r.get(f"n_{t}", 0)) / total)

    x = list(range(len(rows2)))
    bottom = [0.0] * len(rows2)
    plt.figure(figsize=(max(8, len(rows2) * 0.5), 5))
    for t in NODE_TYPES:
        plt.bar(x, fractions_by_type[t], bottom=bottom, label=t)
        bottom = [bottom[i] + fractions_by_type[t][i] for i in range(len(bottom))]
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("fraction of nodes")
    plt.title("Node type composition per trace")
    plt.legend(ncol=7, fontsize=8)
    _save_fig(out_path)


def plot_timeline(
    doc: dict[str, Any], out_path: Path, shade_motifs: bool = True
) -> None:
    if not HAS_MPL:
        return
    nodes = doc.get("nodes", []) or []
    edges = doc.get("edges", []) or []
    if not nodes:
        return

    by_type: dict[str, list] = {t: [] for t in NODE_TYPES}
    for n in nodes:
        t = n.get("type")
        if t in by_type:
            by_type[t].append(n)

    y_pos = {t: i for i, t in enumerate(NODE_TYPES)}
    plt.figure(figsize=(10, 3.5))

    if shade_motifs:
        motifs = detect_motifs_HTEJU(nodes, edges)
        for a, b, _ in motifs:
            plt.axvspan(a - 0.2, b + 0.2, alpha=0.15)

    for t in NODE_TYPES:
        xs = [node_time(n) for n in by_type[t]]
        ys = [y_pos[t]] * len(xs)
        if xs:
            plt.scatter(xs, ys, label=t)

    tC = earliest_time_of_type(nodes, "C")
    if tC is not None:
        plt.axvline(tC, linestyle="--")
    tE_used = earliest_evidence_used_time(nodes, edges)
    if tE_used is not None:
        plt.axvline(tE_used, linestyle=":")

    plt.yticks(list(y_pos.values()), list(y_pos.keys()))
    plt.xlabel("time (message index)")
    plt.title(f"Timeline: {Path(doc.get('input_file', '')).name or ''}")
    plt.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    _save_fig(out_path)


def compute_aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    metrics_means: dict[str, float | None] = {}
    for field in METRIC_FIELDS:
        vals = [safe_float(r.get(field)) for r in rows]
        metrics_means[field] = _mean(vals)

    counts: dict[str, dict[str, Any]] = {}
    for field in COUNT_FIELDS:
        vals = [safe_float(r.get(field)) for r in rows]
        clean = [v for v in vals if v is not None]
        total = sum(clean) if clean else None
        mean_val = (total / len(clean)) if clean else None
        counts[field] = {
            "mean": mean_val,
            "total": int(total) if total is not None else None,
        }

    subgraph_local: dict[str, dict[str, Any]] = {}
    subgraph_global: dict[str, dict[str, Any]] = {}
    for sg in SUBGRAPH_NAMES:
        loc = [bool(r.get(f"sg_local_{sg}")) for r in rows]
        loc_raw = [int(r.get(f"sg_local_count_{sg}", 0)) for r in rows]
        glb = [bool(r.get(f"sg_global_{sg}")) for r in rows]
        glb_raw = [int(r.get(f"sg_global_count_{sg}", 0)) for r in rows]
        subgraph_local[sg] = {
            "count": sum(loc),
            "fraction": sum(loc) / n if n else None,
            "raw_total": sum(loc_raw),
            "raw_mean": _mean([float(v) for v in loc_raw]),
        }
        subgraph_global[sg] = {
            "count": sum(glb),
            "fraction": sum(glb) / n if n else None,
            "raw_total": sum(glb_raw),
            "raw_mean": _mean([float(v) for v in glb_raw]),
        }

    antipattern_local: dict[str, dict[str, Any]] = {}
    antipattern_global: dict[str, dict[str, Any]] = {}
    for ap in ANTIPATTERN_NAMES:
        loc = [bool(r.get(f"ap_local_{ap}")) for r in rows]
        loc_raw = [int(r.get(f"ap_local_count_{ap}", 0)) for r in rows]
        glb = [bool(r.get(f"ap_global_{ap}")) for r in rows]
        glb_raw = [int(r.get(f"ap_global_count_{ap}", 0)) for r in rows]
        antipattern_local[ap] = {
            "count": sum(loc),
            "fraction": sum(loc) / n if n else None,
            "raw_total": sum(loc_raw),
            "raw_mean": _mean([float(v) for v in loc_raw]),
        }
        antipattern_global[ap] = {
            "count": sum(glb),
            "fraction": sum(glb) / n if n else None,
            "raw_total": sum(glb_raw),
            "raw_mean": _mean([float(v) for v in glb_raw]),
        }

    ap_family_local: dict[str, dict[str, Any]] = {}
    ap_family_global: dict[str, dict[str, Any]] = {}
    for fam_name in ANTIPATTERN_FAMILY_NAMES:
        loc = [bool(r.get(f"ap_family_local_{fam_name}")) for r in rows]
        loc_raw = [int(r.get(f"ap_family_local_count_{fam_name}", 0)) for r in rows]
        glb = [bool(r.get(f"ap_family_global_{fam_name}")) for r in rows]
        glb_raw = [int(r.get(f"ap_family_global_count_{fam_name}", 0)) for r in rows]
        ap_family_local[fam_name] = {
            "count": sum(loc),
            "fraction": sum(loc) / n if n else None,
            "raw_total": sum(loc_raw),
            "raw_mean": _mean([float(v) for v in loc_raw]),
        }
        ap_family_global[fam_name] = {
            "count": sum(glb),
            "fraction": sum(glb) / n if n else None,
            "raw_total": sum(glb_raw),
            "raw_mean": _mean([float(v) for v in glb_raw]),
        }

    sg_family_local: dict[str, dict[str, Any]] = {}
    sg_family_global: dict[str, dict[str, Any]] = {}
    for fam_name in SUBGRAPH_FAMILY_NAMES:
        loc = [bool(r.get(f"sg_family_local_{fam_name}")) for r in rows]
        loc_raw = [int(r.get(f"sg_family_local_count_{fam_name}", 0)) for r in rows]
        glb = [bool(r.get(f"sg_family_global_{fam_name}")) for r in rows]
        glb_raw = [int(r.get(f"sg_family_global_count_{fam_name}", 0)) for r in rows]
        sg_family_local[fam_name] = {
            "count": sum(loc),
            "fraction": sum(loc) / n if n else None,
            "raw_total": sum(loc_raw),
            "raw_mean": _mean([float(v) for v in loc_raw]),
        }
        sg_family_global[fam_name] = {
            "count": sum(glb),
            "fraction": sum(glb) / n if n else None,
            "raw_total": sum(glb_raw),
            "raw_mean": _mean([float(v) for v in glb_raw]),
        }

    return {
        "n_traces": n,
        "metric_means": metrics_means,
        "counts": counts,
        "subgraph_presence_local": subgraph_local,
        "subgraph_presence_global": subgraph_global,
        "antipattern_presence_local": antipattern_local,
        "antipattern_presence_global": antipattern_global,
        "antipattern_family_local": ap_family_local,
        "antipattern_family_global": ap_family_global,
        "subgraph_family_local": sg_family_local,
        "subgraph_family_global": sg_family_global,
    }


def print_aggregate_stats(stats: dict[str, Any]) -> None:
    logger.info(f"=== Aggregate statistics across {stats['n_traces']} traces ===")
    logger.info("-- Metric means --")
    for k, v in stats["metric_means"].items():
        logger.info(f"  {k:<45s} {v:.4f}" if v is not None else f"  {k:<45s} N/A")
    logger.info("-- Count mean / total --")
    for k, d in stats["counts"].items():
        mean_s = f"{d['mean']:.2f}" if d["mean"] is not None else "N/A"
        tot_s = str(d["total"]) if d["total"] is not None else "N/A"
        logger.info(f"  {k:<35s}  {mean_s:>10s}  {tot_s:>10s}")


@dataclass(frozen=True)
class ModelLevelAggregate:
    model: str
    env: str
    level: str
    annotated_dir: Path
    aggregate_stats_path: Path
    aggregate_stats: dict[str, Any]


@dataclass(frozen=True)
class TraceNodeRow:
    model: str
    env: str
    level: str
    file: str
    input_file: str
    message_count: int
    counts: dict[str, int]


def iter_model_level_aggregates(root: Path) -> list[ModelLevelAggregate]:
    aggregates: list[ModelLevelAggregate] = []
    for model_dir in sorted(root.iterdir()):
        if (
            not model_dir.is_dir()
            or model_dir.name.startswith(".")
            or model_dir.name == "analysis"
        ):
            continue
        for env_dir in sorted(model_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            for level_dir in sorted(env_dir.iterdir()):
                if not level_dir.is_dir():
                    continue
                annotated_dir = level_dir / "annotated"
                agg_path = annotated_dir / "analysis" / "aggregate_stats.json"
                if not annotated_dir.is_dir() or not agg_path.is_file():
                    continue
                aggregates.append(
                    ModelLevelAggregate(
                        model=model_dir.name,
                        env=env_dir.name,
                        level=level_dir.name,
                        annotated_dir=annotated_dir,
                        aggregate_stats_path=agg_path,
                        aggregate_stats=safe_read_json(agg_path),
                    )
                )
    return aggregates


def resolve_input_file(annotated_doc: dict[str, Any], annotated_path: Path) -> Path:
    raw_input = annotated_doc.get("input_file")
    if raw_input:
        candidate = Path(raw_input)
        if not candidate.is_absolute():
            candidate = SCRIPT_DIR / candidate
        if candidate.is_file():
            return candidate
    stem = annotated_path.name.removesuffix(".annotated.json")
    fallback = annotated_path.parent.parent / f"{stem}.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Could not resolve original trace for {annotated_path}")


def collect_trace_node_rows(
    aggregates: list[ModelLevelAggregate],
) -> list[TraceNodeRow]:
    rows: list[TraceNodeRow] = []
    for agg in aggregates:
        for ap in sorted(agg.annotated_dir.glob("*.annotated.json")):
            doc = safe_read_json(ap)
            try:
                input_path = resolve_input_file(doc, ap)
            except FileNotFoundError:
                logger.warning(f"Skipping {ap.name}: cannot resolve input file")
                continue
            input_doc = safe_read_json(input_path)
            messages = input_doc.get("messages", []) or []
            try:
                _rel_input = input_path.relative_to(SCRIPT_DIR)
            except ValueError:
                _rel_input = input_path
            rows.append(
                TraceNodeRow(
                    model=agg.model,
                    env=agg.env,
                    level=agg.level,
                    file=ap.name,
                    input_file=str(_rel_input),
                    message_count=len(messages),
                    counts=count_nodes_by_type_annotated(doc.get("nodes", []) or []),
                )
            )
    return rows


def _group_trace_rows(
    trace_rows: list[TraceNodeRow], key: str, value: str | None = None
) -> list[TraceNodeRow]:
    if key == "overall":
        return trace_rows
    if key == "model":
        return [r for r in trace_rows if r.model == value]
    if key == "env":
        return [r for r in trace_rows if r.env == value]
    if key == "level":
        return [r for r in trace_rows if r.level == value]
    if key == "model_env_level":
        model, env, level = value.split("/", maxsplit=2)
        return [
            r
            for r in trace_rows
            if r.model == model and r.env == env and r.level == level
        ]
    raise ValueError(f"Unsupported grouping key: {key}")


def _group_aggregates(
    aggregates: list[ModelLevelAggregate], key: str, value: str | None = None
) -> list[ModelLevelAggregate]:
    if key == "overall":
        return aggregates
    if key == "model":
        return [a for a in aggregates if a.model == value]
    if key == "env":
        return [a for a in aggregates if a.env == value]
    if key == "level":
        return [a for a in aggregates if a.level == value]
    if key == "model_env_level":
        model, env, level = value.split("/", maxsplit=2)
        return [
            a
            for a in aggregates
            if a.model == model and a.env == env and a.level == level
        ]
    raise ValueError(f"Unsupported grouping key: {key}")


def summarize_node_probabilities(rows: list[TraceNodeRow]) -> dict[str, Any]:
    total_messages = sum(r.message_count for r in rows)
    summary: dict[str, Any] = {
        "n_traces": len(rows),
        "message_count": {
            "total": total_messages,
            "mean": _mean([float(r.message_count) for r in rows]),
        },
        "node_probability_per_message": {},
    }
    for field in NODE_COUNT_FIELDS:
        per_trace = [_safe_div(r.counts.get(field, 0), r.message_count) for r in rows]
        total_nodes = sum(r.counts.get(field, 0) for r in rows)
        summary["node_probability_per_message"][field] = {
            "mean_per_trace": _mean(per_trace),
            "pooled": _safe_div(total_nodes, total_messages),
            "total_nodes": total_nodes,
            "total_messages": total_messages,
        }
    return summary


def summarize_metrics_and_subgraphs(
    aggregates: list[ModelLevelAggregate],
) -> dict[str, Any]:
    total_traces = sum(
        int(a.aggregate_stats.get("n_traces", 0) or 0) for a in aggregates
    )
    metric_means: dict[str, float | None] = {}
    for field in METRIC_FIELDS:
        weighted_sum = 0.0
        weight_total = 0
        for a in aggregates:
            n_tr = int(a.aggregate_stats.get("n_traces", 0) or 0)
            val = a.aggregate_stats.get("metric_means", {}).get(field)
            if val is None:
                continue
            weighted_sum += float(val) * n_tr
            weight_total += n_tr
        metric_means[field] = _safe_div(weighted_sum, weight_total)

    subgraph_presence_global: dict[str, Any] = {}
    for sg in SUBGRAPH_NAMES:
        bc = sum(
            int(
                a.aggregate_stats.get("subgraph_presence_global", {})
                .get(sg, {})
                .get("count", 0)
                or 0
            )
            for a in aggregates
        )
        rt = sum(
            int(
                a.aggregate_stats.get("subgraph_presence_global", {})
                .get(sg, {})
                .get("raw_total", 0)
                or 0
            )
            for a in aggregates
        )
        subgraph_presence_global[sg] = {
            "count": bc,
            "fraction": _safe_div(bc, total_traces),
            "raw_total": rt,
            "raw_mean": _safe_div(rt, total_traces),
        }

    antipattern_local: dict[str, Any] = {}
    antipattern_global: dict[str, Any] = {}
    for ap_name in ANTIPATTERN_NAMES:
        lc = sum(
            int(
                a.aggregate_stats.get("antipattern_presence_local", {})
                .get(ap_name, {})
                .get("count", 0)
                or 0
            )
            for a in aggregates
        )
        lrt = sum(
            int(
                a.aggregate_stats.get("antipattern_presence_local", {})
                .get(ap_name, {})
                .get("raw_total", 0)
                or 0
            )
            for a in aggregates
        )
        gc = sum(
            int(
                a.aggregate_stats.get("antipattern_presence_global", {})
                .get(ap_name, {})
                .get("count", 0)
                or 0
            )
            for a in aggregates
        )
        grt = sum(
            int(
                a.aggregate_stats.get("antipattern_presence_global", {})
                .get(ap_name, {})
                .get("raw_total", 0)
                or 0
            )
            for a in aggregates
        )
        antipattern_local[ap_name] = {
            "count": lc,
            "fraction": _safe_div(lc, total_traces),
            "raw_total": lrt,
            "raw_mean": _safe_div(lrt, total_traces),
        }
        antipattern_global[ap_name] = {
            "count": gc,
            "fraction": _safe_div(gc, total_traces),
            "raw_total": grt,
            "raw_mean": _safe_div(grt, total_traces),
        }

    ap_family_local: dict[str, Any] = {}
    ap_family_global: dict[str, Any] = {}
    for fam_name in ANTIPATTERN_FAMILY_NAMES:
        lc = sum(
            int(
                a.aggregate_stats.get("antipattern_family_local", {})
                .get(fam_name, {})
                .get("count", 0)
                or 0
            )
            for a in aggregates
        )
        lrt = sum(
            int(
                a.aggregate_stats.get("antipattern_family_local", {})
                .get(fam_name, {})
                .get("raw_total", 0)
                or 0
            )
            for a in aggregates
        )
        gc = sum(
            int(
                a.aggregate_stats.get("antipattern_family_global", {})
                .get(fam_name, {})
                .get("count", 0)
                or 0
            )
            for a in aggregates
        )
        grt = sum(
            int(
                a.aggregate_stats.get("antipattern_family_global", {})
                .get(fam_name, {})
                .get("raw_total", 0)
                or 0
            )
            for a in aggregates
        )
        ap_family_local[fam_name] = {
            "count": lc,
            "fraction": _safe_div(lc, total_traces),
            "raw_total": lrt,
            "raw_mean": _safe_div(lrt, total_traces),
        }
        ap_family_global[fam_name] = {
            "count": gc,
            "fraction": _safe_div(gc, total_traces),
            "raw_total": grt,
            "raw_mean": _safe_div(grt, total_traces),
        }

    sg_family_global: dict[str, Any] = {}
    for fam_name in SUBGRAPH_FAMILY_NAMES:
        bc = sum(
            int(
                a.aggregate_stats.get("subgraph_family_global", {})
                .get(fam_name, {})
                .get("count", 0)
                or 0
            )
            for a in aggregates
        )
        rt = sum(
            int(
                a.aggregate_stats.get("subgraph_family_global", {})
                .get(fam_name, {})
                .get("raw_total", 0)
                or 0
            )
            for a in aggregates
        )
        sg_family_global[fam_name] = {
            "count": bc,
            "fraction": _safe_div(bc, total_traces),
            "raw_total": rt,
            "raw_mean": _safe_div(rt, total_traces),
        }

    return {
        "n_traces": total_traces,
        "metric_means": metric_means,
        "subgraph_presence_global": subgraph_presence_global,
        "antipattern_presence_local": antipattern_local,
        "antipattern_presence_global": antipattern_global,
        "antipattern_family_local": ap_family_local,
        "antipattern_family_global": ap_family_global,
        "subgraph_family_global": sg_family_global,
    }


def build_group_summary(
    trace_rows: list[TraceNodeRow],
    aggregates: list[ModelLevelAggregate],
    key: str,
    value: str | None = None,
) -> dict[str, Any]:
    return {
        **summarize_node_probabilities(_group_trace_rows(trace_rows, key, value)),
        **summarize_metrics_and_subgraphs(_group_aggregates(aggregates, key, value)),
    }


def build_cross_model_summary(root: Path) -> dict[str, Any]:
    aggregates = iter_model_level_aggregates(root)
    if not aggregates:
        raise FileNotFoundError(f"No aggregate_stats.json found under {root}")

    trace_rows = collect_trace_node_rows(aggregates)
    models = sorted({a.model for a in aggregates})
    envs = sorted({a.env for a in aggregates})
    levels = sorted({a.level for a in aggregates})

    try:
        _rel_root = root.relative_to(SCRIPT_DIR)
    except ValueError:
        _rel_root = root
    summary: dict[str, Any] = {
        "root": str(_rel_root),
        "node_probability_definition": "node_count / number_of_messages_in_original_trace",
        "groupings": {
            "by_model_env_level": {},
            "by_model": {},
            "by_env": {},
            "by_level": {},
            "overall": build_group_summary(trace_rows, aggregates, "overall"),
        },
    }

    for a in aggregates:
        key = f"{a.model}/{a.env}/{a.level}"
        if key not in summary["groupings"]["by_model_env_level"]:
            summary["groupings"]["by_model_env_level"][key] = build_group_summary(
                trace_rows, aggregates, "model_env_level", key
            )

    for model in models:
        summary["groupings"]["by_model"][model] = build_group_summary(
            trace_rows, aggregates, "model", model
        )
    for env in envs:
        summary["groupings"]["by_env"][env] = build_group_summary(
            trace_rows, aggregates, "env", env
        )
    for level in levels:
        summary["groupings"]["by_level"][level] = build_group_summary(
            trace_rows, aggregates, "level", level
        )

    summary["trace_node_rows"] = [
        {
            "model": r.model,
            "env": r.env,
            "level": r.level,
            "file": r.file,
            "input_file": r.input_file,
            "message_count": r.message_count,
            **r.counts,
            **{
                f"{field}_per_message": _safe_div(
                    r.counts.get(field, 0), r.message_count
                )
                for field in NODE_COUNT_FIELDS
            },
        }
        for r in trace_rows
    ]
    return summary


def build_aggregation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Reasoning annotation analysis",
        "",
        f"- Node probability definition: {summary['node_probability_definition']}",
        "",
    ]

    def add_section(title: str, grouping: dict[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for name, data in grouping.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(
                f"- Traces: {data['n_traces']} | Total messages: {data['message_count']['total']} | Mean messages/trace: {data['message_count']['mean']:.2f}"
            )
            lines.append("")
            lines.append("#### Node probability per message")
            lines.append("")
            lines.append("| field | mean/trace | pooled |")
            lines.append("| --- | ---: | ---: |")
            for field, stats in data["node_probability_per_message"].items():
                m = (
                    "N/A"
                    if stats["mean_per_trace"] is None
                    else f"{stats['mean_per_trace']:.4f}"
                )
                p = "N/A" if stats["pooled"] is None else f"{stats['pooled']:.4f}"
                lines.append(f"| {field} | {m} | {p} |")
            lines.append("")
            lines.append("#### Metric means")
            lines.append("")
            lines.append("| metric | mean |")
            lines.append("| --- | ---: |")
            for metric, val in data["metric_means"].items():
                lines.append(f"| {metric} | {'N/A' if val is None else f'{val:.4f}'} |")
            lines.append("")
            lines.append("#### Global subgraph presence")
            lines.append("")
            lines.append("| subgraph | count | fraction | raw_total | raw_mean |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for sg, stats in data["subgraph_presence_global"].items():
                f_text = (
                    "N/A" if stats["fraction"] is None else f"{stats['fraction']:.4f}"
                )
                rt = stats.get("raw_total", "N/A")
                rm = (
                    "N/A"
                    if stats.get("raw_mean") is None
                    else f"{stats['raw_mean']:.4f}"
                )
                lines.append(f"| {sg} | {stats['count']} | {f_text} | {rt} | {rm} |")
            lines.append("")
            if "antipattern_presence_global" in data:
                lines.append("#### Anti-pattern presence (global)")
                lines.append("")
                lines.append(
                    "| anti-pattern | count | fraction | raw_total | raw_mean |"
                )
                lines.append("| --- | ---: | ---: | ---: | ---: |")
                for ap_name, stats in data["antipattern_presence_global"].items():
                    f_text = (
                        "N/A"
                        if stats["fraction"] is None
                        else f"{stats['fraction']:.4f}"
                    )
                    rt = stats.get("raw_total", "N/A")
                    rm = (
                        "N/A"
                        if stats.get("raw_mean") is None
                        else f"{stats['raw_mean']:.4f}"
                    )
                    lines.append(
                        f"| {ap_name} | {stats['count']} | {f_text} | {rt} | {rm} |"
                    )
                lines.append("")
            if "antipattern_presence_local" in data:
                lines.append("#### Anti-pattern presence (local)")
                lines.append("")
                lines.append(
                    "| anti-pattern | count | fraction | raw_total | raw_mean |"
                )
                lines.append("| --- | ---: | ---: | ---: | ---: |")
                for ap_name, stats in data["antipattern_presence_local"].items():
                    f_text = (
                        "N/A"
                        if stats["fraction"] is None
                        else f"{stats['fraction']:.4f}"
                    )
                    rt = stats.get("raw_total", "N/A")
                    rm = (
                        "N/A"
                        if stats.get("raw_mean") is None
                        else f"{stats['raw_mean']:.4f}"
                    )
                    lines.append(
                        f"| {ap_name} | {stats['count']} | {f_text} | {rt} | {rm} |"
                    )
                lines.append("")
            if "antipattern_family_global" in data:
                lines.append("#### Anti-pattern family presence (global)")
                lines.append("")
                lines.append("| family | count | fraction | raw_total | raw_mean |")
                lines.append("| --- | ---: | ---: | ---: | ---: |")
                for fam_name, stats in data["antipattern_family_global"].items():
                    f_text = (
                        "N/A"
                        if stats["fraction"] is None
                        else f"{stats['fraction']:.4f}"
                    )
                    rt = stats.get("raw_total", "N/A")
                    rm = (
                        "N/A"
                        if stats.get("raw_mean") is None
                        else f"{stats['raw_mean']:.4f}"
                    )
                    lines.append(
                        f"| {fam_name} | {stats['count']} | {f_text} | {rt} | {rm} |"
                    )
                lines.append("")
            if "antipattern_family_local" in data:
                lines.append("#### Anti-pattern family presence (local)")
                lines.append("")
                lines.append("| family | count | fraction | raw_total | raw_mean |")
                lines.append("| --- | ---: | ---: | ---: | ---: |")
                for fam_name, stats in data["antipattern_family_local"].items():
                    f_text = (
                        "N/A"
                        if stats["fraction"] is None
                        else f"{stats['fraction']:.4f}"
                    )
                    rt = stats.get("raw_total", "N/A")
                    rm = (
                        "N/A"
                        if stats.get("raw_mean") is None
                        else f"{stats['raw_mean']:.4f}"
                    )
                    lines.append(
                        f"| {fam_name} | {stats['count']} | {f_text} | {rt} | {rm} |"
                    )
                lines.append("")
            if "subgraph_family_global" in data:
                lines.append("#### Subgraph family presence (global)")
                lines.append("")
                lines.append("| family | count | fraction | raw_total | raw_mean |")
                lines.append("| --- | ---: | ---: | ---: | ---: |")
                for fam_name, stats in data["subgraph_family_global"].items():
                    f_text = (
                        "N/A"
                        if stats["fraction"] is None
                        else f"{stats['fraction']:.4f}"
                    )
                    rt = stats.get("raw_total", "N/A")
                    rm = (
                        "N/A"
                        if stats.get("raw_mean") is None
                        else f"{stats['raw_mean']:.4f}"
                    )
                    lines.append(
                        f"| {fam_name} | {stats['count']} | {f_text} | {rt} | {rm} |"
                    )
                lines.append("")

    add_section("By model + env + level", summary["groupings"]["by_model_env_level"])
    add_section("By model", summary["groupings"]["by_model"])
    add_section("By env", summary["groupings"]["by_env"])
    add_section("By level", summary["groupings"]["by_level"])
    add_section("Overall", {"overall": summary["groupings"]["overall"]})
    return "\n".join(lines).rstrip() + "\n"


def discover_trace_files(root: Path) -> list[tuple[str, str, str, Path]]:
    """Discover JSON traces organized as model/env/level/*.json under root."""
    files: list[tuple[str, str, str, Path]] = []
    for model_dir in sorted(root.iterdir()):
        if (
            not model_dir.is_dir()
            or model_dir.name.startswith(".")
            or model_dir.name in {"analysis", "__pycache__"}
        ):
            continue
        for env_dir in sorted(model_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            for level_dir in sorted(env_dir.iterdir()):
                if not level_dir.is_dir() or level_dir.name in {
                    "annotated",
                    "analysis",
                }:
                    continue
                for json_file in sorted(level_dir.glob("*.json")):
                    if json_file.name.endswith(".annotated.json"):
                        continue
                    files.append(
                        (model_dir.name, env_dir.name, level_dir.name, json_file)
                    )
    return files


def discover_annotated_dirs(root: Path) -> list[Path]:
    """Find all annotated/ directories under root/model/env/level/."""
    dirs: list[Path] = []
    for model_dir in sorted(root.iterdir()):
        if (
            not model_dir.is_dir()
            or model_dir.name.startswith(".")
            or model_dir.name in {"analysis", "__pycache__"}
        ):
            continue
        for env_dir in sorted(model_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            for level_dir in sorted(env_dir.iterdir()):
                if not level_dir.is_dir():
                    continue
                ann = level_dir / "annotated"
                if ann.is_dir() and list(ann.glob("*.annotated.json")):
                    dirs.append(ann)
    return dirs


async def process_file_async(
    in_path: Path,
    out_dir: Path,
    model: str,
    window: int,
    overlap: int,
    max_nodes_per_window: int,
    strict_support: bool,
    dry_run: bool,
) -> dict[str, Any]:
    data = safe_read_json(in_path)
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{in_path.name}: missing or invalid 'messages' list.")

    clean_messages: list[dict[str, Any]] = []
    for _i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", ""))
        content = m.get("content", "")
        if content is None:
            content = ""
        clean_messages.append({"role": role, "content": str(content)})

    qc_warnings: list[str] = []

    nodes, wA = await extract_nodes_pass_a(
        messages=clean_messages,
        model=model,
        window=window,
        overlap=overlap,
        max_nodes_per_window=max_nodes_per_window,
        dry_run=dry_run,
    )
    qc_warnings.extend(wA)

    if strict_support and not dry_run:
        nodes = [
            n
            for n in nodes
            if validate_support_quotes(clean_messages, n.get("support", []))[0]
        ]

    edges, wB = await extract_edges_pass_b(
        messages=clean_messages,
        nodes=nodes,
        model=model,
        window=window,
        overlap=overlap,
        dry_run=dry_run,
    )
    qc_warnings.extend(wB)

    if strict_support and not dry_run:
        node_ids = {n["node_id"] for n in nodes}
        edges = [
            e
            for e in edges
            if validate_support_quotes(clean_messages, e.get("support", []))[0]
            and e.get("src") in node_ids
            and e.get("dst") in node_ids
        ]

    metrics = compute_metrics(nodes, edges) if not dry_run else {}

    try:
        _rel_in = in_path.relative_to(SCRIPT_DIR)
    except ValueError:
        _rel_in = in_path
    result = {
        "input_file": str(_rel_in),
        "provenance": {
            "model": model,
            "window": window,
            "overlap": overlap,
            "max_nodes_per_window": max_nodes_per_window,
            "strict_support": strict_support,
            "dry_run": dry_run,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "nodes": nodes,
        "edges": edges,
        "metrics": metrics,
        "qc": {"warnings": qc_warnings},
    }

    out_path = out_dir / f"{in_path.stem}.annotated.json"
    safe_write_json(out_path, result)
    return result


async def annotate_all(
    trace_files: list[tuple[str, str, str, Path]],
    model: str,
    concurrency: int,
    window: int,
    overlap: int,
    max_nodes_per_window: int,
    strict_support: bool,
    dry_run: bool,
    force: bool,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    ok = 0
    failed = 0
    skipped = 0

    async def _process_one(model_name: str, env: str, level: str, fp: Path) -> None:
        nonlocal ok, failed, skipped
        out_dir = fp.parent / "annotated"
        out_path = out_dir / f"{fp.stem}.annotated.json"
        if out_path.exists() and not force:
            skipped += 1
            return
        file_retries = 3
        for file_attempt in range(1, file_retries + 1):
            async with semaphore:
                try:
                    await process_file_async(
                        in_path=fp,
                        out_dir=out_dir,
                        model=model,
                        window=window,
                        overlap=overlap,
                        max_nodes_per_window=max_nodes_per_window,
                        strict_support=strict_support,
                        dry_run=dry_run,
                    )
                    ok += 1
                    logger.info(f"[OK] {model_name}/{env}/{level}/{fp.name}")
                    return
                except Exception as e:
                    if file_attempt < file_retries:
                        logger.warning(
                            f"[RETRY {file_attempt}/{file_retries}] {model_name}/{env}/{level}/{fp.name}: {e}"
                        )
                        await asyncio.sleep(min(2**file_attempt, 30))
                    else:
                        failed += 1
                        logger.error(
                            f"[FAIL] {model_name}/{env}/{level}/{fp.name}: {e}"
                        )

    if not dry_run:
        _load_litellm()

    tasks = [_process_one(m, e, line, fp) for m, e, line, fp in trace_files]
    await asyncio.gather(*tasks)
    logger.info(f"Annotation done. ok={ok} failed={failed} skipped={skipped}")


def run_analysis_for_dir(annotated_dir: Path, shade_motifs: bool = True) -> None:
    files = sorted(annotated_dir.glob("*.annotated.json"))
    if not files:
        return

    out_dir = annotated_dir / "analysis"
    ensure_dir(out_dir)

    rows: list[dict[str, Any]] = []
    docs: list[tuple[Path, dict[str, Any]]] = []

    for fp in files:
        doc = safe_read_json(fp)
        docs.append((fp, doc))
        rows.append(summarize_trace(doc, fp))

    agg_stats = compute_aggregate_stats(rows)
    safe_write_json(out_dir / "aggregate_stats.json", agg_stats)
    print_aggregate_stats(agg_stats)

    if HAS_MPL:
        agg_plots = out_dir / "aggregate_plots"
        ensure_dir(agg_plots)
        plot_stacked_node_type_fractions(
            rows, agg_plots / "node_type_fractions_stacked.png"
        )

        tl_dir = out_dir / "timelines"
        ensure_dir(tl_dir)
        for fp, doc in docs:
            plot_timeline(
                doc, tl_dir / f"{fp.stem}.timeline.png", shade_motifs=shade_motifs
            )

    logger.info(f"Analysis done for {annotated_dir}")


async def _run_pipeline(
    root: str,
    model: str,
    concurrency: int,
    window: int,
    overlap: int,
    max_nodes_per_window: int,
    strict_support: bool,
    dry_run: bool,
    force: bool,
    shade_motifs: bool,
    skip_annotate: bool,
) -> None:
    root_path = Path(root).resolve()

    if not skip_annotate:
        trace_files = discover_trace_files(root_path)
        if trace_files:
            logger.info(f"Discovered {len(trace_files)} trace files")
            await annotate_all(
                trace_files=trace_files,
                model=model,
                concurrency=concurrency,
                window=window,
                overlap=overlap,
                max_nodes_per_window=max_nodes_per_window,
                strict_support=strict_support,
                dry_run=dry_run,
                force=force,
            )
        else:
            logger.warning("No trace files found to annotate")

    annotated_dirs = discover_annotated_dirs(root_path)
    if annotated_dirs:
        logger.info(f"Running analysis for {len(annotated_dirs)} annotated directories")
        for ann_dir in annotated_dirs:
            run_analysis_for_dir(ann_dir, shade_motifs=shade_motifs)
    else:
        logger.warning("No annotated directories found for analysis")

    try:
        summary = build_cross_model_summary(root_path)
        out_dir = root_path / "analysis"
        ensure_dir(out_dir)
        safe_write_json(out_dir / "annotation_summary.json", summary)
        md_path = out_dir / "annotation_summary.md"
        md_path.write_text(build_aggregation_markdown(summary), encoding="utf-8")
        logger.info(f"Cross-model aggregation written to {out_dir}")
    except FileNotFoundError as e:
        logger.warning(f"Skipping aggregation: {e}")


def main(
    root: str = str(SCRIPT_DIR),
    model: str = "anthropic/claude-sonnet-4-6",
    concurrency: int = DEFAULT_CONCURRENCY,
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
    max_nodes_per_window: int = DEFAULT_MAX_NODES_PER_WINDOW,
    strict_support: bool = False,
    dry_run: bool = False,
    force: bool = False,
    shade_motifs: bool = False,
    skip_annotate: bool = False,
) -> None:
    """Unified reasoning analysis pipeline: annotate, analyze, and aggregate.

    Processes trace files organized as `<root>/<model>/<env>/<level>/*.json`
    through three stages: LLM-based annotation, per-directory analysis with
    aggregate statistics and plots, and cross-model aggregation.

    Only annotation is optional (via `skip_annotate`). Analysis and
    aggregation always run on existing annotated files.

    Args:
        root: Root directory containing `model/env/level/` folders with
            JSON trace files. Defaults to the directory containing this
            script.
        model: LiteLLM model identifier used for annotation LLM calls
            (e.g. `"anthropic/claude-sonnet-4-6"`, `"openai/gpt-4o"`).
        concurrency: Maximum number of trace files to annotate in parallel.
            Increase for faster throughput; decrease to avoid rate limits.
        window: Sliding window size in number of messages passed to the LLM
            for each annotation call.
        overlap: Number of messages overlapping between consecutive windows,
            ensuring context continuity across boundaries.
        max_nodes_per_window: Upper limit on nodes extracted per window in
            Pass A.  Caps the LLM output to prevent runaway extraction.
        strict_support: When True, discard nodes and edges whose support
            quotes cannot be found verbatim in the original messages.
        dry_run: When True, discover and validate input files without making
            any LLM calls. Useful for checking directory layout.
        force: When True, re-annotate files even if an `.annotated.json`
            output already exists.
        shade_motifs: When True, shade H-T-E-J-U motif time spans in the
            generated timeline plots.
        skip_annotate: Skip the LLM annotation step entirely and run only
            analysis and aggregation on existing annotated files.
    """
    asyncio.run(
        _run_pipeline(
            root=root,
            model=model,
            concurrency=concurrency,
            window=window,
            overlap=overlap,
            max_nodes_per_window=max_nodes_per_window,
            strict_support=strict_support,
            dry_run=dry_run,
            force=force,
            shade_motifs=shade_motifs,
            skip_annotate=skip_annotate,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
