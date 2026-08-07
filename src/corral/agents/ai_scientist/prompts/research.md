Design {{count}} {{node_type}} research proposal(s) that maximally reduce the
uncertainty preventing a correct final answer.

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Parent checkpoint:
{{parent}}

Branch workspaces, one per proposal in proposal order:
{{branch_workspaces}}

Global research journal (including sibling evidence):
{{journal}}

Allowed tools:
{{tools}}

Prefer discriminating or falsifying experiments: when hypotheses remain
compatible with current observations, choose an observation expected to differ
between them. A DEBUG node must address the concrete failed action. Use at most
{{max_actions}} sequential actions, listed tools only, and JSON object strings
for arguments. Put any tool output path under the proposal's assigned branch
workspace. Return exactly {{count}} proposal(s) when possible.
