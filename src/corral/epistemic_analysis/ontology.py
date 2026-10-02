"""Ontology for epistemic trace analysis."""

from __future__ import annotations

NODE_TYPES = ["H", "T", "E", "J", "C", "N", "F"]


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


PASS_A_SYSTEM = """You are a careful annotator. You MUST only extract information explicitly present in the provided messages.
Rules:
- Do NOT invent hidden thoughts or implied steps.
- Do not borrow any external knowledge or make assumptions beyond the text.
- Do not judge or correct the content, only label supported nodes.
- For messages with several nodes, you MUST follow the order in which they appear in the text to assign message indices.
- If uncertain, omit the node rather than guessing.
- Output JSON only, matching the required schema.
"""


NODE_CRITERIA = {
    "H": 'Hypothesis: a candidate explanation, or a working assumption about the system. It should be a revisable claim, proposal, suggestion, or the current best guess about the answer. Information in task definitions or environment descriptions do not count as hypotheses (H). Partial or full reiteration of the task descriptionn does not count as hypothesis (H). Correcting a typo in a tool argument does not count as a Hypothesis (H). We consider a Hypothesis (H) to be present if the agent states a claim similar to "I think the answer is X", "This suggests that X might be the solution", "The most likely explanation is X", "X could be the case if...", "It seems that the answer can be X", etc.',
    "E": "Evidence: all Observation messages must be assigned with and only with an Evidence (E) node. Only Observation messages can be Evidence (E) nodes, with the exception of the task description message.",
    "N": "Neutral: a fallback for boilerplate operations like writing or copying files, non-scientific tool calls, or runtime notices. Assign N only when no other node type is assigned to the same span. Unlabelled runtime budget-exhaustion notices receive N during postprocessing.",
    "T": "Test: any information-seeking action, including experiments, evaluations, or lookups. Both the intention and the concrete tool call qualify as one test (T) - each tool call is one test; what matters is that the system is seeking new scientific information to evaluate a hypothesis (H) or a judgment (J). If both the intention/plan to run a test and the actual tool call are present within the same message, count only the tool call as the test (T) node. If both T and N are selected for a tool call (with the tag <action>), keep T. If no node type is assigned to a tool call, use N.",
    "J": "Judgment: an interpretation of test results (Observation) that goes beyond the literal repetition of the raw output. If the agent restates an observation while adding any evaluative, comparative, or inferential content, even brief it is a judgment (J).",
    "F": "Final Answer: Only the message that contains a final submisssion (with action `submit_answer`), must be a assigned with a Final Answer (F) node.",
    "C": "Commitment: if from the actions of the agents or system it can be inferred that they have reached an implicit commitment to an answer that is not yet fully supported by evidence, and that they are **refusing to revise it**, then create a pseudo-node labeled C. This is a special pseudo-node that captures the commitment even if it is not explicitly stated. It is possible that the Final Answer (F) node is also a Commitment (C) if the Final Answer (F) is not fully supported by evidence.",
}

PASS_A_INSTRUCTIONS = f"""Extract 0..k nodes from the provided message window.

Every non-Observation message must be assigned with at least one node. It is possible that some messages will have multiple nodes (e.g., a message that both states a hypothesis and describes a test). Avoid repeated nodes of the **same type** within a single message, unless the text explicitly supports multiple distinct instances.

Node types:
H = {NODE_CRITERIA["H"]}
E = {NODE_CRITERIA["E"]}
N = {NODE_CRITERIA["N"]}
T = {NODE_CRITERIA["T"]}
J = {NODE_CRITERIA["J"]}
F = {NODE_CRITERIA["F"]}

Constraints for nodes:
- Only label what is explicitly present in text.
- Every node must have support quotes (exact substrings) and msg indices.
- If you normalize text, still cite original quote(s).

Pseudo-nodes (not explicitly stated but can be inferred):
C = {NODE_CRITERIA["C"]}

Constrains for pseudo-nodes:
- Only Commitment (C) can be a pseudo-node.
- For the quote support of a pseudo-node, you can cite the text that implies the commitment, even if it is not explicit. You are allowed to add a brief explanation to clarify the implication, but it must be concise and directly tied to the quote.


Return JSON with keys:
{{
  "nodes": [
    {{
      "node_id": "N1",
      "type": "H|T|E|J|C|F",
      "time": <int message index of earliest support>,
      "text": <normalized short node text>,
      "support": [{{"msg_idx": <int>, "quote": <exact substring from that message>}}],
    }}, ...
  ]
}}
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
