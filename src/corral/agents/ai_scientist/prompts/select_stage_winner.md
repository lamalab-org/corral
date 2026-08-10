Select the strongest checkpoint for the boundary of main stage {{stage}} by
comparing the candidates directly. Do not use or speculate about a benchmark
score.

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Inherited seed node id:
{{seed_node_id}}

Candidates with observations, objectively parsed measured outcomes when
available, critic evaluations, and visual feedback:
{{candidates}}

Global evidence journal:
{{journal}}

Choose exactly one candidate id. Compare experimental validity, task-specific
progress, strength and reproducibility of evidence, observed dynamics, visual
feedback, and unresolved failures. When comparable measured outcomes are
present, treat their declared direction as the objective anchor; never replace
them with an invented scalar. Prefer the inherited seed when later-stage
experiments do not actually improve it. Return the selected id, a concise
reason, and a short listwise comparison. Never select an id outside the given
candidates.
