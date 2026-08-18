Design {{count}} {{node_type}} proposal(s) that optimize the experimental
procedure, not merely the wording of a conclusion.

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

All reliable evidence:
{{journal}}

Allowed tools:
{{tools}}

Vary only meaningful parameters identified in the formulation or justified by
observations. If the formulation lists no explicit tunable parameter, optimize
a meaningful experimental or procedural choice such as sampling, controls,
measurement settings, or analysis thresholds. Define one experiment goal and
observable success criteria, and
state what comparison the change enables. Do not repeat an existing child, and
make requested slots materially different parameter comparisons. Do not
precompute tool actions. A
bounded worker will choose up to {{max_actions}} listed-tool actions one at a
time from actual observations. Keep output paths relative to the assigned trial
workspace. Return exactly {{count}} proposal(s) when possible.
