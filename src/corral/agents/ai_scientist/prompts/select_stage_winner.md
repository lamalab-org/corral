Select the strongest checkpoint for the boundary of main stage {{stage}} by
comparing the candidates directly. Do not use or speculate about a benchmark
score.

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Inherited seed node id:
{{seed_node_id}}

Candidates with observations, critic evaluations, and visual feedback:
{{candidates}}

Global evidence journal:
{{journal}}

Choose exactly one candidate id. Compare experimental validity, task-specific
progress, strength and reproducibility of evidence, observed dynamics, visual
feedback, and unresolved failures. Prefer the inherited seed when later-stage
experiments do not actually improve it. Return the selected id, a concise
reason, and a short listwise comparison. Never select an id outside the given
candidates.
