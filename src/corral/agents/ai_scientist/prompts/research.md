Design {{count}} {{node_type}} research proposal(s) that maximally reduce the
uncertainty preventing a correct final answer.

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Current experimental substage and its specific objectives:
{{substage}}

Parent checkpoint:
{{parent}}

Existing child experiments from this checkpoint:
{{siblings}}

Proposal slots in this expansion:
{{proposal_slots}}

Execution-relative workspaces, one per proposal in proposal order:
{{branch_workspaces}}

Global research journal (including sibling evidence):
{{journal}}

Allowed tools:
{{tools}}

Prefer discriminating or falsifying experiments: when hypotheses remain
compatible with current observations, choose an observation expected to differ
between them. Define one experiment goal and observable success criteria per
proposal. Do not repeat an existing child, and make requested slots test
materially different uncertainties. Do not precompute tool actions. A bounded
worker will choose up to
{{max_actions}} actions adaptively from actual observations. A DEBUG node must
address the concrete failed action. The experiment must use listed tools only
and keep output paths relative to the assigned trial workspace. Return exactly
{{count}} proposal(s) when possible.
