Design {{count}} {{node_type}} preliminary experiment proposal(s).

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

Trial-relative workspaces, one per proposal in proposal order:
{{branch_workspaces}}

Research journal:
{{journal}}

Allowed tools:
{{tools}}

Each proposal must define one scientifically distinct experiment: its
hypothesis, rationale, experiment goal, and observable success criteria. Do not
repeat an existing child experiment, and make every requested proposal slot
materially different from the other slots in hypothesis or measurement. Do not
precompute tool actions. A bounded worker will choose up to {{max_actions}}
actions one at a time after observing each result. The experiment must be
executable using only the listed tools. When outputs are needed, keep their
paths relative to the assigned trial workspace. A DEBUG proposal must diagnose
the parent's actual failure; a REFINE proposal must improve a successful route.
Return exactly {{count}} proposal(s) when possible.
