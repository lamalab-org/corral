Design one verification proposal of type {{node_type}}.

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Current experimental substage and its specific objectives:
{{substage}}

Best parent checkpoint:
{{parent}}

Existing child experiments from this checkpoint:
{{siblings}}

Proposal slot in this expansion:
{{proposal_slots}}

Execution-relative workspace:
{{branch_workspaces}}

Global research journal:
{{journal}}

Allowed tools:
{{tools}}

For ABLATION, remove or change an assumption carrying the conclusion. For
REPLICATION, independently repeat the decisive measurement. For COUNTERFACTUAL,
seek an observation that would falsify the current conclusion. For AGGREGATION,
define an experiment goal that reconciles repeated or conflicting evidence
already in the journal and requires no environment action. For every other
type, define one experiment goal and observable success criteria. Do not repeat
an existing child experiment from this checkpoint. Do not
precompute tool actions. A bounded worker will choose up to {{max_actions}}
listed-tool actions adaptively. Keep output paths relative to the assigned trial
workspace.
