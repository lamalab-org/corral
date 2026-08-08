Evaluate this experiment as evidence, without access to or speculation about a
benchmark score.

Task:
{{task_prompt}}

Formulation:
{{formulation}}

Experiment node and actual observations:
{{node}}

Reliable evidence known before this node:
{{journal}}

Visual artifacts produced by this experiment (empty when none are available):
{{visual_artifacts}}

Score validity, task progress, evidence strength, information gain, and
consistency independently from 0 to 1. Failed or invalid tool actions must
reduce validity, but successful partial observations may still have information
value. Record only conclusions supported by actual observations, explicitly
record contradicted claims and unresolved questions, and recommend whether to
continue, refine, debug, finalize, or abandon. Never invent a result.
When image content is attached, inspect axes, legends, convergence, anomalies,
and visible agreement with the success criteria. Put concise observations in
`visual_feedback`; treat a plot as evidence only for what is actually visible.
