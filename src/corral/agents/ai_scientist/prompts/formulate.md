Formulate the scientific task before any experiment is run.

Task:
{{task_prompt}}

Allowed Corral tools:
{{tools}}

Optional worked examples:
{{examples}}

Identify the objective, exact required answer and format, known and unknown
information, at least three genuinely different candidate hypotheses when the
task permits them, observable quantities, possible experiments, and explicit
success criteria. List tunable experimental parameters only when changing them
could materially improve the procedure; otherwise return an empty list. Do not
perform novelty search and do not assume access to anything outside the tools.
When the experiments expose a meaningful task-internal scalar objective, define
one `measured_metric` with its exact result-field name, direction, and unit. Do
not use, request, or infer a Corral benchmark score. Leave the metric null when
no defensible scalar exists.
