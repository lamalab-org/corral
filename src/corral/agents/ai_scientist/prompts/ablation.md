Design one verification proposal of type {{node_type}}.

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Best parent checkpoint:
{{parent}}

Branch workspace:
{{branch_workspaces}}

Global research journal:
{{journal}}

Allowed tools:
{{tools}}

For ABLATION, remove or change an assumption carrying the conclusion. For
REPLICATION, independently repeat the decisive measurement. For COUNTERFACTUAL,
seek an observation that would falsify the current conclusion. For AGGREGATION,
use no tool actions and reconcile repeated or conflicting evidence already in
the journal. Non-aggregation proposals may use at most {{max_actions}}
sequential actions. Use listed tools only and JSON object strings for arguments.
Put any tool output path under the assigned branch workspace.
