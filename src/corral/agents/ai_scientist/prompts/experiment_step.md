Carry out the single scientific experiment represented by this tree node.

Task:
{{task_prompt}}

Task formulation:
{{formulation}}

Experiment node:
{{node}}

Prior partial checkpoint being continued:
{{prior_checkpoint}}

Reliable evidence from the global journal:
{{journal}}

This experiment's trajectory so far, including actual tool observations:
{{trajectory}}

Allowed tools:
{{tools}}

You have {{remaining_actions}} tool action(s) left in this node. Choose the next
step using the observations above; do not precompute later steps. If zero tool
actions remain, you MUST return `decision="finish"`; requesting another action
will mark the experiment partial. Return
`decision="act"` with exactly one listed tool, its JSON object-string
arguments, purpose, and expected information when another measurement would
advance this experiment. Return `decision="finish"`, no action fields, and a
brief evidence-grounded conclusion when the experiment is complete or no
useful action remains. Keep output paths relative to the node's branch
workspace. Do not call scoring or answer-submission tools.
