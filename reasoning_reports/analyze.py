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

SCRIPT_DIR = Path(__file__).resolve().parent

NODE_TYPES = ["H", "T", "E", "J", "C", "N"]
NODE_TYPES_SET = set(NODE_TYPES)
EDGE_RELATIONS = [
    "tests",
    "observes",
    "informs",
    "updates_to",
    "competes_with",
    "contradicts",
]
EDGE_RELATIONS_SET = set(EDGE_RELATIONS)

# Allowed (relation, src_type, dst_type) combinations for edges.
# "tests" edges are bidirectional (H<->T, J<->T).
# Same-type edges are bidirectional by definition (H<->H, J<->J).
ALLOWED_EDGE_TYPE_COMBOS: set[tuple[str, str, str]] = {
    # tests (bidirectional)
    ("tests", "H", "T"),
    ("tests", "T", "H"),
    ("tests", "J", "T"),
    ("tests", "T", "J"),
    # observes
    ("observes", "T", "E"),
    # updates_to (same type, bidirectional)
    ("updates_to", "H", "H"),
    # competes_with (same type, bidirectional)
    ("competes_with", "H", "H"),
    # contradicts
    ("contradicts", "E", "H"),
    ("contradicts", "J", "H"),
    # informs
    ("informs", "E", "H"),
    ("informs", "E", "J"),
    ("informs", "E", "C"),
    ("informs", "J", "C"),
    ("informs", "J", "H"),
    ("informs", "J", "J"),
}

# The description dicts are the single source of truth for pattern names.
# Change a key here and it propagates to SUBGRAPH_NAMES, families, matchers, etc.

SUBGRAPH_DESCRIPTIONS: dict[str, str] = {
    "refutation_driven_belief_revision": (
        "Evidence triggers a belief update to a new hypothesis "
        "[H -tests-> T -observes-> E -informs-> J, H -updates_to-> H2]."
    ),
    "fixed_hypothesis_test_tuning": (
        "Hypothesis is held fixed while tests are iteratively adjusted "
        "[H -tests-> T -observes-> E -informs-> J -tests-> T2]."
    ),
    "explore_then_test_transition": (
        "Exploration precedes hypothesis formation, which then drives testing "
        "[T -observes-> E ... H ... H -tests-> T]."
    ),
    "hypothesis_reranking": (
        "Competing hypotheses are compared as new evidence arrives "
        "[H1 -competes_with- H2, both tested]."
    ),
    "evidence_led_hypothesis_generation": (
        "Evidence is observed first; a hypothesis is formed afterward "
        "[E -informs-> J ... H ... H -tests-> T]."
    ),
    "convergent_multi_test_evidence": (
        "One hypothesis is evaluated via multiple independent tests "
        "[H -tests-> T1/T2/T3..., each -> E]."
    ),
    "precommitted_test_plan": (
        "A commitment is stated before evidence collection begins "
        "[C before E; then H -tests-> T]."
    ),
    "evidence_guided_test_redesign": (
        "A judgment motivates a new test, which then produces new evidence "
        "[J -tests-> T -observes-> E]."
    ),
}

SUBGRAPH_NAMES = list(SUBGRAPH_DESCRIPTIONS.keys())

(
    SG_REFUTATION_DRIVEN_BELIEF_REVISION,
    SG_FIXED_HYPOTHESIS_TEST_TUNING,
    SG_EXPLORE_THEN_TEST_TRANSITION,
    SG_HYPOTHESIS_RERANKING,
    SG_EVIDENCE_LED_HYPOTHESIS_GENERATION,
    SG_CONVERGENT_MULTI_TEST_EVIDENCE,
    SG_PRECOMMITTED_TEST_PLAN,
    SG_EVIDENCE_GUIDED_TEST_REDESIGN,
) = SUBGRAPH_NAMES

ANTIPATTERN_DESCRIPTIONS: dict[str, str] = {
    "untested_claim": ("Hypothesis never linked to a test [H with no tests]."),
    "evidence_non_uptake": (
        "Evidence collected but never used [E with no informs to J or H]."
    ),
    "unsupported_judgment": (
        "Judgment made without supporting evidence [J with no E via informs]."
    ),
    "stalled_revision": (
        "Revised hypothesis never tested "
        "[H target of updates_to with no outgoing tests]."
    ),
    "contradiction_without_repair": (
        "Contradiction unresolved by any update or alternative "
        "[E -contradicts-> H, no updates_to/competes_with]."
    ),
    "premature_commitment": (
        "Hypothesis committed without intermediate testing [H -> C with no T]."
    ),
    "uninformative_test": ("Test produces no observed evidence [T with no E]."),
    "fixed_belief_trace": (
        "No hypothesis revision in the entire trace " "[No updates_to edges in trace]."
    ),
    "disconnected_evidence": ("Evidence node with no edges [Isolated E]."),
    "one_sided_confirmation": (
        "Commitment reached without considering contradicting evidence "
        "[H -> C with support, no contradicts]."
    ),
}

ANTIPATTERN_NAMES = list(ANTIPATTERN_DESCRIPTIONS.keys())

(
    AP_UNTESTED_CLAIM,
    AP_EVIDENCE_NON_UPTAKE,
    AP_UNSUPPORTED_JUDGMENT,
    AP_STALLED_REVISION,
    AP_CONTRADICTION_WITHOUT_REPAIR,
    AP_PREMATURE_COMMITMENT,
    AP_UNINFORMATIVE_TEST,
    AP_FIXED_BELIEF_TRACE,
    AP_DISCONNECTED_EVIDENCE,
    AP_ONE_SIDED_CONFIRMATION,
) = ANTIPATTERN_NAMES

ANTIPATTERN_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        AP_UNTESTED_CLAIM,
        AP_CONTRADICTION_WITHOUT_REPAIR,
        AP_ONE_SIDED_CONFIRMATION,
    ],
    "evidence_handling": [
        AP_EVIDENCE_NON_UPTAKE,
        AP_DISCONNECTED_EVIDENCE,
        AP_UNSUPPORTED_JUDGMENT,
        AP_UNINFORMATIVE_TEST,
    ],
    "experimental_strategy": [
        AP_STALLED_REVISION,
        AP_FIXED_BELIEF_TRACE,
        AP_PREMATURE_COMMITMENT,
    ],
}
ANTIPATTERN_FAMILY_NAMES = list(ANTIPATTERN_FAMILIES.keys())

SUBGRAPH_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        SG_REFUTATION_DRIVEN_BELIEF_REVISION,
        SG_HYPOTHESIS_RERANKING,
        SG_EVIDENCE_LED_HYPOTHESIS_GENERATION,
    ],
    "evidence_handling": [
        SG_CONVERGENT_MULTI_TEST_EVIDENCE,
        SG_EXPLORE_THEN_TEST_TRANSITION,
    ],
    "experimental_strategy": [
        SG_FIXED_HYPOTHESIS_TEST_TUNING,
        SG_PRECOMMITTED_TEST_PLAN,
        SG_EVIDENCE_GUIDED_TEST_REDESIGN,
    ],
}
SUBGRAPH_FAMILY_NAMES = list(SUBGRAPH_FAMILIES.keys())

DEFAULT_WINDOW = 20
DEFAULT_OVERLAP = 5
DEFAULT_MAX_NODES_PER_WINDOW = 100
DEFAULT_CONCURRENCY = 8


PASS_A_SYSTEM = """You are a careful annotator. You MUST only extract information explicitly present in the provided messages.
Rules:
- Do NOT invent hidden thoughts or implied steps.
- Do not borrow any external knowledge or make assumptions beyond the text.
- Do not judge or correct the content, only label supported nodes.
- For messages with several nodes, you MUST follow the order in which they appear in the text to assign message indices.
- If uncertain, omit the node rather than guessing.
- Output JSON only, matching the required schema.
"""

PASS_A_INSTRUCTIONS = """Extract 0..k nodes from the provided message window.

Every non-Observation message must be assigned with at least one node. It is possible that some messages will have multiple nodes (e.g., a message that both states a hypothesis and describes a test). Avoid repeated nodes of the **same type** within a single message, unless the text explicitly supports multiple distinct instances.

Node types:
H = Hypothesis: a candidate explanation, or a working assumption about the system. It should be a revisable claim, proposal, or the current best guess about the answer. Information in task definitions or environment descriptions do not count as hypotheses (H).
E = Evidence: all Observation messages must be assigned with and only with an Evidence (E) node. Only Observation messages can be Evidence (E) nodes.
T = Test: any information-seeking action, including experiments, evaluations, or lookups. Both the intention and the concrete tool call qualify as tests (T). What matters is that the system is seeking new scientific information to evaluate a hypothesis (H) or a judgment (J). Only the tool calls of a message can be assigned as Test (T), and only if it is not Neutral (N).
N = Neutral: for boilerplate operations like writing or copying files, or non-scientific tool calls. Only the tool calls of a message can be assigned as Neutral (N).
J = Judgment: an interpretation of test results (Observation) that goes beyond the literal repetition of the raw output. If the agent restates an observation while adding any evaluative, comparative, or inferential content, even brief it is a judgment (J).

Constraints for nodes:
- Only label what is explicitly present in text.
- Every node must have support quotes (exact substrings) and msg indices.
- If you normalize text, still cite original quote(s).

Pseudo-nodes (not explicitly stated but can be inferred):
C = Commitment: if from the actions of the agents or system it can be inferred that they have reached an implicit commitment to an answer that is not yet fully supported by evidence, and that they are **refusing to revise it**, then create a pseudo-node labeled C. This is a special pseudo-node that captures the commitment even if it is not explicitly stated.

Constrains for pseudo-nodes:
- Only Commitment (C) can be a pseudo-node.
- For the quote support of a pseudo-node, you can cite the text that implies the commitment, even if it is not explicit. You are allowed to add a brief explanation to clarify the implication, but it must be concise and directly tied to the quote.


Return JSON with keys:
{
  "nodes": [
    {
      "node_id": "N1",
      "type": "H|T|E|J|U|V|C",
      "time": <int message index of earliest support>,
      "text": <normalized short node text>,
      "support": [{"msg_idx": <int>, "quote": <exact substring from that message>}],
    }, ...
  ]
}
"""

PASS_B_SYSTEM = """You are a careful annotator. You MUST only add edges supported by explicit text.
Rules:
- You may only connect nodes provided to you.
- Do not borrow any external knowledge or make assumptions beyond the text.
- Do not judge or correct the content, only label supported edges.
- Every edge MUST include at least one support quote with message indices.
- If uncertain, omit the edge rather than guessing.
- Output JSON only, matching the required schema.
"""

PASS_B_INSTRUCTIONS = """Given the message window and a list of extracted nodes (with node_id and text), extract supported edges among these nodes.

Only the following edge types are allowed, any other combination is forbidden:
- tests: Allowed only between: H -> T, J -> T.
    H -> T: the Test (T) directly addresses the Hypothesis' claim (H), or attempts to falsify or verify it.
    J -> T: the Test (T) is designed in response to the Judgment (J), or the Judgment (J) motivates the test design.
- observes: Allowed only between: T -> E.
    T -> E: the Evidence (E) is a direct result of the Test (T).
- updates_to: Allowed only between: H -> H.
    H -> H: the later Hypothesis (H) is a revision of the earlier one, based on the nodes in between.
- competes_with: Allowed only between: H -> H.
    H -> H: the two Hypotheses (H) are alternative explanations that are directly compared or evaluated against each other. Both hypotheses should be plausible and co-existing at the same time. If another hypothesis is introduced later as a revision of the first one, then it should be connected with updates_to instead of competes_with.
- contradicts: Allowed only between: E -> H, J -> H.
    E -> H: the Evidence (E) contradicts the claim of the Hypothesis (H).
    J -> H: the Judgment (J) contradicts the claim of the Hypothesis (H).
- informs: Allowed only between: E -> H, E -> J, E -> C, J -> C, J -> H, J -> J.
    E -> H: the Evidence (E) provides information relevant to the claim of the Hypothesis (H).
    E -> J: the Judgment (J) is an interpretation of the Evidence (E).
    E -> C: the Commitment (C) is informed by the Evidence (E) but not necessarily in an explicit way.
    J -> H: the Judgment (J) provides information relevant to the claim of the Hypothesis (H).
    J -> J: the later Judgment (J) is a refinement, an extension, or a combination of one or serveral earlier judgments (J).
    J -> C: the Commitment (C) is informed by the Judgment (J) but not necessarily in an explicit way.

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

_TIKZ_STYLE_DEFS = r"""\definecolor{nodefill}{HTML}{DFE3E8}%
\tikzset{%
  rnode/.style={circle, draw, thick, minimum size=5mm, inner sep=1pt,
                font=\scriptsize\bfseries, fill=nodefill},%
  rlbl/.style={font=\tiny, midway, above},%
  rlblb/.style={font=\tiny, midway, below},%
  missing/.style={dashed, gray},%
  missingnode/.style={rnode, dashed, gray, text=gray, fill=none},%
}%
"""


def _make_tikz(body: str) -> str:
    """Wrap TikZ node/edge commands in a tikzpicture environment."""
    indented = body.strip().replace("\n", "\n  ")
    return (
        r"\begin{tikzpicture}"
        r"[>=Stealth, baseline=(current bounding box.center), node distance=7mm]"
        "\n  " + indented + "\n"
        r"\end{tikzpicture}"
    )


_TIKZ_SUBGRAPH_PATTERNS: dict[str, str] = {
    SG_REFUTATION_DRIVEN_BELIEF_REVISION: _make_tikz(
        r"""\node[rnode] (h1) {H};
\node[rnode, right=of h1] (t) {T};
\node[rnode, right=of t] (e) {E};
\node[rnode, right=of e] (j) {J};
\node[rnode, right=of j] (h2) {H$_2$};
\draw[->] (h1) -- node[rlbl] {tests} (t);
\draw[->] (t) -- node[rlbl] {obs.} (e);
\draw[->] (e) -- node[rlbl] {inf.} (j);
\draw[->] (h1) to[bend right=40] node[rlblb] {upd.} (h2);"""
    ),
    SG_FIXED_HYPOTHESIS_TEST_TUNING: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, right=of h] (t) {T};
\node[rnode, right=of t] (e) {E};
\node[rnode, right=of e] (j) {J};
\draw[->] (h) -- node[rlbl] {tests} (t);
\draw[->] (t) -- node[rlbl] {obs.} (e);
\draw[->] (e) -- node[rlbl] {inf.} (j);
\draw[->, bend left=50] (j) to node[rlblb] {tests} (t);"""
    ),
    SG_EXPLORE_THEN_TEST_TRANSITION: _make_tikz(
        r"""\node[rnode] (t1) {T};
\node[rnode, right=of t1] (e) {E};
\node[right=4mm of e, draw=none, font=\scriptsize] (dots) {\ldots};
\node[rnode, right=4mm of dots] (h) {H};
\node[rnode, right=of h] (t2) {T};
\draw[->] (t1) -- node[rlbl] {obs.} (e);
\draw[->] (h) -- node[rlbl] {tests} (t2);"""
    ),
    SG_HYPOTHESIS_RERANKING: _make_tikz(
        r"""\node[rnode] (h1) {H$_1$};
\node[rnode, right=15mm of h1] (h2) {H$_2$};
\node[rnode, below left=5mm and 0mm of h1] (t1) {T};
\node[rnode, below right=5mm and 0mm of h2] (t2) {T};
\draw[<->, dashed] (h1) -- node[rlbl] {competes} (h2);
\draw[->] (h1) -- node[left, font=\tiny] {tests} (t1);
\draw[->] (h2) -- node[right, font=\tiny] {tests} (t2);"""
    ),
    SG_EVIDENCE_LED_HYPOTHESIS_GENERATION: _make_tikz(
        r"""\node[rnode] (e) {E};
\node[rnode, right=of e] (j) {J};
\node[right=4mm of j, draw=none, font=\scriptsize] (dots) {\ldots};
\node[rnode, right=4mm of dots] (h) {H};
\node[rnode, right=of h] (t) {T};
\draw[->] (e) -- node[rlbl] {inf.} (j);
\draw[->] (h) -- node[rlbl] {tests} (t);"""
    ),
    SG_CONVERGENT_MULTI_TEST_EVIDENCE: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, right=10mm of h, yshift=7mm] (t1) {T$_1$};
\node[rnode, right=10mm of h] (t2) {T$_2$};
\node[rnode, right=10mm of h, yshift=-7mm] (t3) {T$_3$};
\node[rnode, right=10mm of t2] (e) {E};
\draw[->] (h) -- (t1);
\draw[->] (h) -- (t2);
\draw[->] (h) -- (t3);
\draw[->] (t1) -- (e);
\draw[->] (t2) -- (e);
\draw[->] (t3) -- (e);"""
    ),
    SG_PRECOMMITTED_TEST_PLAN: _make_tikz(
        r"""\node[rnode] (c) {C};
\node[right=4mm of c, draw=none, font=\scriptsize] (dots) {\ldots};
\node[rnode, right=4mm of dots] (h) {H};
\node[rnode, right=of h] (t) {T};
\node[rnode, right=of t] (e) {E};
\draw[->] (h) -- node[rlbl] {tests} (t);
\draw[->] (t) -- node[rlbl] {obs.} (e);"""
    ),
    SG_EVIDENCE_GUIDED_TEST_REDESIGN: _make_tikz(
        r"""\node[rnode] (j) {J};
\node[rnode, right=of j] (t) {T};
\node[rnode, right=of t] (e) {E};
\draw[->] (j) -- node[rlbl] {tests} (t);
\draw[->] (t) -- node[rlbl] {obs.} (e);"""
    ),
}


_TIKZ_ANTIPATTERN_PATTERNS: dict[str, str] = {
    AP_UNTESTED_CLAIM: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[missingnode, right=of h] (t) {T};
\draw[->, missing] (h) -- node[rlbl, text=gray] {tests} (t);
\draw[red, thick] (t.north west) -- (t.south east);
\draw[red, thick] (t.north east) -- (t.south west);"""
    ),
    AP_EVIDENCE_NON_UPTAKE: _make_tikz(
        r"""\node[rnode] (e) {E};
\node[missingnode, right=of e] (j) {J};
\draw[->, missing] (e) -- node[rlbl, text=gray] {inf.} (j);
\draw[red, thick] (j.north west) -- (j.south east);
\draw[red, thick] (j.north east) -- (j.south west);"""
    ),
    AP_UNSUPPORTED_JUDGMENT: _make_tikz(
        r"""\node[missingnode] (e) {E};
\node[rnode, right=of e] (j) {J};
\draw[->, missing] (e) -- node[rlbl, text=gray] {inf.} (j);
\draw[red, thick] (e.north west) -- (e.south east);
\draw[red, thick] (e.north east) -- (e.south west);"""
    ),
    AP_STALLED_REVISION: _make_tikz(
        r"""\node[rnode] (h1) {H};
\node[rnode, right=of h1] (h2) {H$_2$};
\node[right=7mm of h2, draw=none, font=\scriptsize, text=gray] (none) {$\varnothing$};
\draw[->] (h1) -- node[rlbl] {upd.} (h2);
\draw[->, missing] (h2) -- (none);"""
    ),
    AP_CONTRADICTION_WITHOUT_REPAIR: _make_tikz(
        r"""\node[rnode] (e) {E};
\node[rnode, right=of e] (h) {H};
\node[missingnode, right=of h] (h2) {H$_2$};
\draw[->] (e) -- node[rlbl] {contr.} (h);
\draw[->, missing] (h) -- node[rlbl, text=gray] {upd.} (h2);
\draw[red, thick] (h2.north west) -- (h2.south east);
\draw[red, thick] (h2.north east) -- (h2.south west);"""
    ),
    AP_PREMATURE_COMMITMENT: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, right=of h] (c) {C};
\node[missingnode, below=5mm of h] (t) {T};
\draw[->] (h) -- (c);
\draw[->, missing] (h) -- (t);
\draw[red, thick] (t.north west) -- (t.south east);
\draw[red, thick] (t.north east) -- (t.south west);"""
    ),
    AP_UNINFORMATIVE_TEST: _make_tikz(
        r"""\node[rnode] (t) {T};
\node[missingnode, right=of t] (e) {E};
\draw[->, missing] (t) -- node[rlbl, text=gray] {obs.} (e);
\draw[red, thick] (e.north west) -- (e.south east);
\draw[red, thick] (e.north east) -- (e.south west);"""
    ),
    AP_FIXED_BELIEF_TRACE: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, right=of h] (t) {T};
\node[rnode, right=of t] (e) {E};
\node[rnode, right=of e] (j) {J};
\node[missingnode, below=5mm of h] (h2) {H$_2$};
\draw[->] (h) -- (t);
\draw[->] (t) -- (e);
\draw[->] (e) -- (j);
\draw[->, missing] (h) -- node[left, font=\tiny, text=gray] {upd.} (h2);
\draw[red, thick] (h2.north west) -- (h2.south east);
\draw[red, thick] (h2.north east) -- (h2.south west);"""
    ),
    AP_DISCONNECTED_EVIDENCE: _make_tikz(
        r"""\node[rnode] (e) {E};
\node[missingnode, left=of e] (t) {T};
\node[missingnode, right=of e] (j) {J};
\draw[->, missing] (t) -- (e);
\draw[->, missing] (e) -- (j);
\draw[red, thick] (t.north west) -- (t.south east);
\draw[red, thick] (t.north east) -- (t.south west);
\draw[red, thick] (j.north west) -- (j.south east);
\draw[red, thick] (j.north east) -- (j.south west);"""
    ),
    AP_ONE_SIDED_CONFIRMATION: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, below left=5mm and 1mm of h] (es) {E};
\node[rnode, right=12mm of h] (c) {C};
\node[missingnode, below right=5mm and 1mm of h] (ec) {E$_{\!c}$};
\draw[->] (es) -- (h);
\draw[->] (h) -- (c);
\draw[->, missing] (ec) -- node[right, font=\tiny, text=gray] {contr.} (h);
\draw[red, thick] (ec.north west) -- (ec.south east);
\draw[red, thick] (ec.north east) -- (ec.south west);"""
    ),
}


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
    temperature: float = 0.0,
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
    # Always prepend the first two messages (task / system context)
    preamble_end = min(2, len(messages))
    for i in range(preamble_end):
        role = messages[i].get("role", "")
        content = messages[i].get("content", "")
        if content is None:
            content = ""
        content = str(content)
        lines.append(f"[{i}] role={role}\n{content}\n")
    # For windows that don't start at the beginning, indicate omitted messages
    actual_start = max(start, preamble_end)
    if actual_start > preamble_end:
        lines.append(f"[... messages {preamble_end}-{actual_start - 1} omitted ...]\n")
    for i in range(actual_start, end):
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


def _nodes_of_type(ntype: str, node_type_map: dict[str, str]) -> list[str]:
    return [nid for nid, t in node_type_map.items() if t == ntype]


def _has_edge(
    src: str, dst: str, relation: str, out_edges: dict[str, list[dict[str, Any]]]
) -> bool:
    for e in out_edges.get(src, []):
        if e.get("relation") == relation and e.get("dst") == dst:
            return True
    return False


def _has_informs_link(
    a: str, b: str, out_edges: dict[str, list[dict[str, Any]]]
) -> bool:
    return _has_edge(a, b, "informs", out_edges) or _has_edge(
        b, a, "informs", out_edges
    )


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
                    if not _has_informs_link(ev, j, out_edges):
                        continue
                    for h2 in _neighbours(
                        h1, "updates_to", "H", out_edges, node_type_map
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
            count += 1
    return count


def _match_ml_make_it_work(node_type_map, _node_by_id, out_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        matched = False
        for t1 in _neighbours(h, "tests", "T", out_edges, node_type_map):
            for ev in _neighbours(t1, "observes", "E", out_edges, node_type_map):
                for j in _nodes_of_type("J", node_type_map):
                    if not _has_informs_link(ev, j, out_edges):
                        continue
                    j_to_Ts = _neighbours(j, "tests", "T", out_edges, node_type_map)
                    if not j_to_Ts:
                        continue
                    h_to_Hs = _neighbours(
                        h, "updates_to", "H", out_edges, node_type_map
                    )
                    if h_to_Hs:
                        continue
                    matched = True
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
            if not _has_informs_link(e0, j0, out_edges):
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
    for j in _nodes_of_type("J", node_type_map):
        for t in _neighbours(j, "tests", "T", out_edges, node_type_map):
            if _neighbours(t, "observes", "E", out_edges, node_type_map):
                count += 1
                break
    return count


_SUBGRAPH_MATCHERS = {
    SG_REFUTATION_DRIVEN_BELIEF_REVISION: _match_popperian,
    SG_FIXED_HYPOTHESIS_TEST_TUNING: _match_ml_make_it_work,
    SG_EXPLORE_THEN_TEST_TRANSITION: _match_exploratory_to_confirmatory,
    SG_HYPOTHESIS_RERANKING: _match_bayesian,
    SG_EVIDENCE_LED_HYPOTHESIS_GENERATION: _match_abductive,
    SG_CONVERGENT_MULTI_TEST_EVIDENCE: _match_triangulation,
    SG_PRECOMMITTED_TEST_PLAN: _match_preregistered,
    SG_EVIDENCE_GUIDED_TEST_REDESIGN: _match_active_learning,
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
    has_HH_updates = _typed_edge_exists("H", "H", "updates_to", node_by_id, out_edges)
    has_JT_tests = _typed_edge_exists("J", "T", "tests", node_by_id, out_edges)
    has_HH_competes = _typed_edge_exists(
        "H", "H", "competes_with", node_by_id, out_edges
    )
    fan_out_H_T = _fan_out_global("H", "T", "tests", node_by_id, out_edges)
    has_EJ_informs = _typed_edge_exists(
        "E", "J", "informs", node_by_id, out_edges
    ) or _typed_edge_exists("J", "E", "informs", node_by_id, out_edges)

    results: dict[str, int] = {}
    results[SG_REFUTATION_DRIVEN_BELIEF_REVISION] = int(
        has_HT_tests and has_TE_observes and has_HH_updates and n_H >= 2
    )
    results[SG_FIXED_HYPOTHESIS_TEST_TUNING] = int(
        n_H <= 1
        and n_T >= 2
        and has_HT_tests
        and has_TE_observes
        and has_EJ_informs
        and not has_HH_updates
    )
    results[SG_EXPLORE_THEN_TEST_TRANSITION] = int(
        t_first_T is not None
        and t_first_H is not None
        and t_first_T < t_first_H
        and has_HT_tests
    )
    results[SG_HYPOTHESIS_RERANKING] = int(n_H >= 2 and has_HH_competes)
    results[SG_EVIDENCE_LED_HYPOTHESIS_GENERATION] = int(
        t_first_E is not None and t_first_H is not None and t_first_E < t_first_H
    )
    results[SG_CONVERGENT_MULTI_TEST_EVIDENCE] = int(fan_out_H_T >= 3)
    results[SG_PRECOMMITTED_TEST_PLAN] = int(
        t_first_C is not None and t_first_E is not None and t_first_C < t_first_E
    )
    results[SG_EVIDENCE_GUIDED_TEST_REDESIGN] = int(has_JT_tests and has_TE_observes)
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
            e.get("relation") == "informs"
            and node_type_map.get(e.get("dst")) in {"J", "H"}
            for e in out_edges.get(ev, [])
        )
        if not used:
            used = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) in {"J", "H"}
                for e in in_edges.get(ev, [])
            )
        if not used:
            count += 1
    return count


def _ap_judgment_without_evidence(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for j in _nodes_of_type("J", node_type_map):
        has_e = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(j, [])
        )
        if not has_e:
            has_e = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(j, [])
            )
        if not has_e:
            count += 1
    return count


def _ap_dead_end_update(node_type_map, _node_by_id, out_edges, in_edges):
    # Revised hypotheses (targets of updates_to) with no outgoing tests
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        is_revised = any(e.get("relation") == "updates_to" for e in in_edges.get(h, []))
        if not is_revised:
            continue
        has_test = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not has_test:
            count += 1
    return count


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
                    and node_type_map.get(src) == "H"
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
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(h, [])
        )
        if not links_c:
            links_c = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) == "C"
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


def _ap_no_belief_revision(_node_type_map, _node_by_id, out_edges, _in_edges):
    # No updates_to edges anywhere in the trace
    has_update = any(
        e.get("relation") == "updates_to"
        for edges_list in out_edges.values()
        for e in edges_list
    )
    return 0 if has_update else 1


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
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(h, [])
        )
        if not committed:
            committed = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) == "C"
                for e in in_edges.get(h, [])
            )
        if not committed:
            continue
        has_support = False
        for ie in in_edges.get(h, []):
            src = ie.get("src")
            if ie.get("relation") == "informs" and node_type_map.get(src) == "J":
                for je in in_edges.get(src, []):
                    if (
                        je.get("relation") == "informs"
                        and node_type_map.get(je.get("src")) == "E"
                    ):
                        has_support = True
                        break
            if ie.get("relation") == "informs" and node_type_map.get(src) == "E":
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
    AP_UNTESTED_CLAIM: _ap_untested_hypothesis,
    AP_EVIDENCE_NON_UPTAKE: _ap_evidence_ignored,
    AP_UNSUPPORTED_JUDGMENT: _ap_judgment_without_evidence,
    AP_STALLED_REVISION: _ap_dead_end_update,
    AP_CONTRADICTION_WITHOUT_REPAIR: _ap_unresolved_contradiction,
    AP_PREMATURE_COMMITMENT: _ap_hypothesis_to_commitment_shortcut,
    AP_UNINFORMATIVE_TEST: _ap_test_without_evidence,
    AP_FIXED_BELIEF_TRACE: _ap_no_belief_revision,
    AP_DISCONNECTED_EVIDENCE: _ap_orphan_evidence,
    AP_ONE_SIDED_CONFIRMATION: _ap_confirmation_only,
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

    # Count updates_to edges for fixed_belief_trace check
    n_updates_to = sum(1 for e in edges if e.get("relation") == "updates_to")

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
            e.get("relation") == "informs"
            and node_type_map.get(e.get("dst")) in {"J", "H"}
            for e in out_edges.get(ev, [])
        )
        if not used:
            used = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) in {"J", "H"}
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
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(h, [])
        )
        if not links_c:
            links_c = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) == "C"
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

    j_without_e = 0
    for j in _nodes_of_type("J", node_type_map):
        has_e = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(j, [])
        )
        if not has_e:
            has_e = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(j, [])
            )
        if not has_e:
            j_without_e += 1

    # Stalled revision: revised H (target of updates_to) with no outgoing tests
    dead_revised = 0
    for h in _nodes_of_type("H", node_type_map):
        is_revised = any(e.get("relation") == "updates_to" for e in in_edges.get(h, []))
        if not is_revised:
            continue
        has_test = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not has_test:
            dead_revised += 1
    ap_local_counts = detect_antipatterns_local(nodes, edges)

    binary: dict[str, bool] = {
        AP_UNTESTED_CLAIM: (n_H - h_tested) > 0,
        AP_EVIDENCE_NON_UPTAKE: (n_E - e_used) > 0,
        AP_UNSUPPORTED_JUDGMENT: j_without_e > 0,
        AP_STALLED_REVISION: dead_revised > 0,
        AP_CONTRADICTION_WITHOUT_REPAIR: n_unresolved > 0,
        AP_PREMATURE_COMMITMENT: h_to_c_untested > 0,
        AP_UNINFORMATIVE_TEST: (n_T - t_with_ev) > 0,
        AP_FIXED_BELIEF_TRACE: n_updates_to == 0,
        AP_DISCONNECTED_EVIDENCE: e_orphan > 0,
        AP_ONE_SIDED_CONFIRMATION: ap_local_counts.get(AP_ONE_SIDED_CONFIRMATION, 0)
        > 0,
    }
    counts: dict[str, int] = {
        AP_UNTESTED_CLAIM: n_H - h_tested,
        AP_EVIDENCE_NON_UPTAKE: n_E - e_used,
        AP_UNSUPPORTED_JUDGMENT: j_without_e,
        AP_STALLED_REVISION: dead_revised,
        AP_CONTRADICTION_WITHOUT_REPAIR: n_unresolved,
        AP_PREMATURE_COMMITMENT: h_to_c_untested,
        AP_UNINFORMATIVE_TEST: n_T - t_with_ev,
        AP_FIXED_BELIEF_TRACE: 1 if n_updates_to == 0 else 0,
        AP_DISCONNECTED_EVIDENCE: e_orphan,
        AP_ONE_SIDED_CONFIRMATION: ap_local_counts.get(AP_ONE_SIDED_CONFIRMATION, 0),
    }
    return binary, counts


def summarize_trace(doc: dict[str, Any], file_path: Path) -> dict[str, Any]:
    nodes = doc.get("nodes", []) or []
    edges = doc.get("edges", []) or []

    row: dict[str, Any] = {
        "file": file_path.name,
        "input_file": doc.get("input_file", ""),
        "model": doc.get("provenance", {}).get("model", ""),
        "nodes_total": len(nodes),
        "edges_total": len(edges),
    }

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


def compute_aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

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
        "subgraph_presence_local": subgraph_local,
        "subgraph_presence_global": subgraph_global,
        "antipattern_presence_local": antipattern_local,
        "antipattern_presence_global": antipattern_global,
        "antipattern_family_local": ap_family_local,
        "antipattern_family_global": ap_family_global,
        "subgraph_family_local": sg_family_local,
        "subgraph_family_global": sg_family_global,
    }


@dataclass(frozen=True)
class ModelLevelAggregate:
    model: str
    env: str
    level: str
    annotated_dir: Path
    aggregate_stats_path: Path
    aggregate_stats: dict[str, Any]
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


def summarize_subgraphs_and_antipatterns(
    aggregates: list[ModelLevelAggregate],
) -> dict[str, Any]:
    total_traces = sum(
        int(a.aggregate_stats.get("n_traces", 0) or 0) for a in aggregates
    )

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
        "subgraph_presence_global": subgraph_presence_global,
        "antipattern_presence_local": antipattern_local,
        "antipattern_presence_global": antipattern_global,
        "antipattern_family_local": ap_family_local,
        "antipattern_family_global": ap_family_global,
        "subgraph_family_global": sg_family_global,
    }


def build_group_summary(
    aggregates: list[ModelLevelAggregate],
    key: str,
    value: str | None = None,
) -> dict[str, Any]:
    return summarize_subgraphs_and_antipatterns(
        _group_aggregates(aggregates, key, value)
    )


def build_cross_model_summary(root: Path) -> dict[str, Any]:
    aggregates = iter_model_level_aggregates(root)
    if not aggregates:
        raise FileNotFoundError(f"No aggregate_stats.json found under {root}")

    models = sorted({a.model for a in aggregates})
    envs = sorted({a.env for a in aggregates})
    levels = sorted({a.level for a in aggregates})

    try:
        _rel_root = root.relative_to(SCRIPT_DIR)
    except ValueError:
        _rel_root = root
    summary: dict[str, Any] = {
        "root": str(_rel_root),
        "groupings": {
            "by_model_env_level": {},
            "by_model": {},
            "by_env": {},
            "by_level": {},
            "overall": build_group_summary(aggregates, "overall"),
        },
    }

    for a in aggregates:
        key = f"{a.model}/{a.env}/{a.level}"
        if key not in summary["groupings"]["by_model_env_level"]:
            summary["groupings"]["by_model_env_level"][key] = build_group_summary(
                aggregates, "model_env_level", key
            )

    for model in models:
        summary["groupings"]["by_model"][model] = build_group_summary(
            aggregates, "model", model
        )
    for env in envs:
        summary["groupings"]["by_env"][env] = build_group_summary(
            aggregates, "env", env
        )
    for level in levels:
        summary["groupings"]["by_level"][level] = build_group_summary(
            aggregates, "level", level
        )

    return summary


def build_aggregation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Reasoning annotation analysis",
        "",
    ]

    def add_section(title: str, grouping: dict[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for name, data in grouping.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"- Traces: {data['n_traces']}")
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


def _postprocess_observation_nodes(
    nodes: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Enforce that observation messages have exactly one E node each.

    If the previous message contains a Neutral (N) node, the observation
    node is assigned N instead of E.  Also removes E nodes that appear at
    non-observation messages.
    """
    warnings: list[str] = []

    # Identify observation message indices
    obs_indices: set[int] = set()
    for i, msg in enumerate(messages):
        content = str(msg.get("content", "") or "")
        if content.startswith("Observation:"):
            obs_indices.add(i)

    if not obs_indices:
        return nodes, warnings

    # Group nodes by time (message index)
    nodes_by_time: dict[int, list[dict[str, Any]]] = {}
    for n in nodes:
        t = n.get("time")
        if isinstance(t, int):
            nodes_by_time.setdefault(t, []).append(n)

    def _has_neutral_at(msg_idx: int) -> bool:
        return any(n.get("type") == "N" for n in nodes_by_time.get(msg_idx, []))

    # Compute max node id for generating new ones
    max_id = max(
        (
            int(n["node_id"][1:])
            for n in nodes
            if n.get("node_id", "").startswith("N") and n["node_id"][1:].isdigit()
        ),
        default=0,
    )

    obs_covered: set[int] = set()
    result: list[dict[str, Any]] = []

    for n in nodes:
        t = n.get("time")
        ntype = n.get("type")

        if isinstance(t, int) and t in obs_indices:
            if t not in obs_covered:
                # Keep first node but ensure correct type
                prev_neutral = t > 0 and _has_neutral_at(t - 1)
                target_type = "N" if prev_neutral else "E"
                if ntype != target_type:
                    warnings.append(
                        f"Changed node {n.get('node_id')} at observation message {t} "
                        f"from {ntype} to {target_type}."
                    )
                    node = dict(n)
                    node["type"] = target_type
                    result.append(node)
                else:
                    result.append(n)
                obs_covered.add(t)
            else:
                warnings.append(
                    f"Removed extra node {n.get('node_id')} ({ntype}) "
                    f"at observation message {t}: only one node per observation."
                )
        elif ntype == "E":
            warnings.append(
                f"Removed Evidence node {n.get('node_id')} at message {t}: "
                f"not an Observation message."
            )
        else:
            result.append(n)

    # Add missing nodes for uncovered observation messages
    for obs_idx in sorted(obs_indices - obs_covered):
        max_id += 1
        content = str(messages[obs_idx].get("content", "") or "")
        prev_neutral = obs_idx > 0 and _has_neutral_at(obs_idx - 1)
        target_type = "N" if prev_neutral else "E"
        new_node = {
            "node_id": f"N{max_id}",
            "type": target_type,
            "time": obs_idx,
            "text": normalize_whitespace(content[:200]),
            "support": [{"msg_idx": obs_idx, "quote": content[:500]}],
        }
        result.append(new_node)
        warnings.append(
            f"Added {target_type} node N{max_id} for observation message {obs_idx} "
            f"(no node was previously assigned)."
        )

    result.sort(key=lambda n: (n.get("time", 0), n.get("node_id", "")))
    return result, warnings


def _filter_invalid_edges(
    edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove edges whose (relation, src_type, dst_type) is not in ALLOWED_EDGE_TYPE_COMBOS."""
    warnings: list[str] = []
    node_type_map = {n["node_id"]: n.get("type") for n in nodes if "node_id" in n}

    valid_edges: list[dict[str, Any]] = []
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        rel = e.get("relation")
        src_type = node_type_map.get(src)
        dst_type = node_type_map.get(dst)

        if src_type is None or dst_type is None:
            warnings.append(f"Removed edge {src}->{dst} ({rel}): unknown node type.")
            continue

        if (rel, src_type, dst_type) not in ALLOWED_EDGE_TYPE_COMBOS:
            warnings.append(
                f"Removed edge {src}({src_type})->{dst}({dst_type}) ({rel}): "
                f"disallowed type combination."
            )
            continue

        valid_edges.append(e)

    return valid_edges, warnings


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

    if not dry_run:
        nodes, wObs = _postprocess_observation_nodes(nodes, clean_messages)
        qc_warnings.extend(wObs)

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

    if not dry_run:
        edges, wEdge = _filter_invalid_edges(edges, nodes)
        qc_warnings.extend(wEdge)

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


def run_analysis_for_dir(annotated_dir: Path) -> None:
    files = sorted(annotated_dir.glob("*.annotated.json"))
    if not files:
        return

    out_dir = annotated_dir / "analysis"
    ensure_dir(out_dir)

    rows: list[dict[str, Any]] = []

    for fp in files:
        doc = safe_read_json(fp)
        rows.append(summarize_trace(doc, fp))

    agg_stats = compute_aggregate_stats(rows)
    safe_write_json(out_dir / "aggregate_stats.json", agg_stats)

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
    skip_annotate: bool,
    files: list[str] | None = None,
) -> None:
    root_path = Path(root).resolve()

    if files:
        # Annotate only the explicitly listed files
        resolved = [Path(f.strip()).resolve() for f in files]
        missing = [p for p in resolved if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"File(s) not found: {[str(p) for p in missing]}")
        # Build (model, env, level, path) tuples; use "_" placeholders
        # when the file is not inside the standard hierarchy.
        trace_files: list[tuple[str, str, str, Path]] = []
        for p in resolved:
            try:
                rel = p.relative_to(root_path)
                parts = rel.parts  # model/env/level/file.json
                if len(parts) >= 4:
                    trace_files.append((parts[0], parts[1], parts[2], p))
                else:
                    trace_files.append(("_", "_", "_", p))
            except ValueError:
                trace_files.append(("_", "_", "_", p))

        logger.info(f"Annotating {len(trace_files)} explicitly listed file(s)")
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
        logger.info("Done. Skipping analysis/aggregation for explicit file mode.")
        return

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
            run_analysis_for_dir(ann_dir)
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

    write_pattern_definitions_latex(root_path / "analysis" / "results" / "tables")


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in *text*."""
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(ch, f"\\{ch}")
    text = text.replace("~", "\\textasciitilde{}")
    return text.replace("^", "\\textasciicircum{}")


def _pretty_name(raw: str) -> str:
    """Turn a snake_case identifier into a readable title."""
    text = raw.replace("_", " ")
    if not text:
        return raw
    result = text[0].upper() + text[1:].lower()
    return re.sub(r"/(.)", lambda m: "/" + m.group(1).upper(), result)


def _split_description(desc: str) -> tuple[str, str]:
    """Split a description of the form 'prose [graph notation].' into a (prose, graph) tuple."""
    m = re.search(r"\[([^\]]+)\]\s*\.?\s*$", desc)
    if m:
        graph = m.group(1)
        prose = desc[: m.start()].rstrip().rstrip(".")
        return prose, graph
    return desc.rstrip("."), ""


# Display order for the LaTeX definitions table.
# Each entry is (merge_label | None, [pattern_keys]).
# When *merge_label* is not None, \multirow groups the constituent rows.
_SUBGRAPH_TABLE_ORDER: list[tuple[str | None, list[str]]] = [
    (None, [SG_REFUTATION_DRIVEN_BELIEF_REVISION]),
    (
        "Data-first hypothesis",
        [SG_EXPLORE_THEN_TEST_TRANSITION, SG_EVIDENCE_LED_HYPOTHESIS_GENERATION],
    ),
    (None, [SG_HYPOTHESIS_RERANKING]),
    (None, [SG_CONVERGENT_MULTI_TEST_EVIDENCE]),
    (
        "Iterative test refinement",
        [SG_FIXED_HYPOTHESIS_TEST_TUNING, SG_EVIDENCE_GUIDED_TEST_REDESIGN],
    ),
    (None, [SG_PRECOMMITTED_TEST_PLAN]),
]

_ANTIPATTERN_TABLE_ORDER: list[tuple[str | None, list[str]]] = [
    ("Untested hypothesis", [AP_UNTESTED_CLAIM, AP_PREMATURE_COMMITMENT]),
    ("Unused evidence", [AP_EVIDENCE_NON_UPTAKE, AP_DISCONNECTED_EVIDENCE]),
    (None, [AP_UNSUPPORTED_JUDGMENT]),
    (None, [AP_CONTRADICTION_WITHOUT_REPAIR]),
    (None, [AP_UNINFORMATIVE_TEST]),
    ("Absent/Stalled revision", [AP_STALLED_REVISION, AP_FIXED_BELIEF_TRACE]),
    (None, [AP_ONE_SIDED_CONFIRMATION]),
]


def _emit_group(
    merge_name: str | None,
    keys: list[str],
    descriptions: dict[str, str],
    tikz_patterns: dict[str, str] | None = None,
) -> list[str]:
    rows: list[str] = []
    n = len(keys)
    for i, key in enumerate(keys):
        prose, graph_text = _split_description(descriptions[key])
        prose = _latex_escape(prose)
        if tikz_patterns and key in tikz_patterns:
            graph = tikz_patterns[key]
        else:
            graph = _latex_escape(graph_text)
        if n == 1:
            name = _pretty_name(key)
            rows.append(rf"{name} & {graph} & {prose} \\")
        else:
            if i == 0:
                rows.append(
                    rf"\multirow{{{n}}}{{=}}{{{merge_name}}} & {graph} & {prose} \\"
                )
            else:
                rows.append(rf" & {graph} & {prose} \\")
    return rows


def build_productive_motifs_latex() -> str:
    r"""Return a LaTeX tabularx table with definitions of productive motifs.

    Column layout: Topic (X), Graph (TikZ picture), Description (X).
    Related patterns share a single merged row label via multirow.
    Each graph cell contains an inline TikZ diagram.
    """
    lines: list[str] = []
    lines.append(_TIKZ_STYLE_DEFS)
    lines.append(r"\begin{tabularx}{\textwidth}{p{2.2cm}cX}")
    lines.append(r"\toprule")
    lines.append(r"Pattern & Graph & Description \\")
    lines.append(r"\midrule")

    for idx, (merge_name, keys) in enumerate(_SUBGRAPH_TABLE_ORDER):
        if idx:
            lines.append(r"\midrule[0.015em]")
        lines.extend(
            _emit_group(
                merge_name, keys, SUBGRAPH_DESCRIPTIONS, _TIKZ_SUBGRAPH_PATTERNS
            )
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def build_reasoning_breakdowns_latex() -> str:
    r"""Return a LaTeX tabularx table with definitions of reasoning breakdowns.

    Column layout: Topic (X), Graph (TikZ picture), Description (X).
    Related breakdowns share a single merged row label via multirow.
    Each graph cell contains an inline TikZ diagram.
    """
    lines: list[str] = []
    lines.append(_TIKZ_STYLE_DEFS)
    lines.append(r"\begin{tabularx}{\textwidth}{XcX}")
    lines.append(r"\toprule")
    lines.append(r"Pattern & Graph & Description \\")
    lines.append(r"\midrule")

    for idx, (merge_name, keys) in enumerate(_ANTIPATTERN_TABLE_ORDER):
        if idx:
            lines.append(r"\midrule[0.03em]")
        lines.extend(
            _emit_group(
                merge_name, keys, ANTIPATTERN_DESCRIPTIONS, _TIKZ_ANTIPATTERN_PATTERNS
            )
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def write_pattern_definitions_latex(
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write the pattern-definition LaTeX tables to out_dir.

    Produces two files:

    - productive_motifs.tex: Productive Motifs table.
    - reasoning_breakdowns.tex: Reasoning Breakdowns table.

    If out_dir is None, defaults to <script_dir>/../analysis/results/tables/.
    """
    if out_dir is None:
        out_dir = SCRIPT_DIR / "analysis" / "results" / "tables"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motifs_path = out_dir / "productive_motifs.tex"
    motifs_path.write_text(build_productive_motifs_latex(), encoding="utf-8")
    logger.info(f"Productive Motifs LaTeX table written to {motifs_path}")

    breakdowns_path = out_dir / "reasoning_breakdowns.tex"
    breakdowns_path.write_text(build_reasoning_breakdowns_latex(), encoding="utf-8")
    logger.info(f"Reasoning Breakdowns LaTeX table written to {breakdowns_path}")

    return motifs_path, breakdowns_path


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
    skip_annotate: bool = False,
    files: list[str] | None = None,
) -> None:
    """Unified reasoning analysis pipeline: annotate, analyze, and aggregate.

    Processes trace files organized as `<root>/<model>/<env>/<level>/*.json`
    through three stages: LLM-based annotation, per-directory analysis with
    pattern/antipattern detection, and cross-model aggregation.

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
        skip_annotate: Skip the LLM annotation step entirely and run only
            analysis and aggregation on existing annotated files.
        files: Optional list of specific JSON trace file paths to annotate.
            When provided, only these files are annotated (analysis and
            aggregation are skipped). Annotated output is written next to
            each file in an `annotated/` subdirectory.
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
            skip_annotate=skip_annotate,
            files=files,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)

# ["claude_sonnet_45/afm/level_2/afm_experiment_level_2-8.json", "gpt_4o/ml/level_1/ml_oxides-13.json", "gpt_4o/wetlab/level_2/qualysis_lvl2_08-70.json", "claude_sonnet_45/retrosynthesis/level_3/make_5_lvl3-43.json", "claude_sonnet_45/spectra/level_1/10_15227_orgsyn_096_0036-19.json ", "claude_sonnet_45/ml/level_1/ml_sulphides-10.json"]
