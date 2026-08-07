Design {{count}} {{node_type}} proposal(s) that optimize the experimental
procedure, not merely the wording of a conclusion.

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Parent checkpoint:
{{parent}}

Branch workspaces, one per proposal in proposal order:
{{branch_workspaces}}

All reliable evidence:
{{journal}}

Allowed tools:
{{tools}}

Vary only meaningful parameters identified in the formulation or justified by
observations. State what comparison the change enables. Use at most
{{max_actions}} sequential actions per proposal, listed tools only, and JSON
object strings for arguments. Put any tool output path under the proposal's
assigned branch workspace. Return exactly {{count}} proposal(s) when
possible.
