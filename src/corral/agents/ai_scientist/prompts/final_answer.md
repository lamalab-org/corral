Produce only the submission-ready answer to the benchmark task.

Task and required output format:
{{task_prompt}}

Global research journal:
{{journal}}

Highest-priority experiment nodes:
{{best_nodes}}

Canonical scored workspace:
{{canonical_workspace}}

Use only collected evidence. Resolve contradictions where the observations
permit it, preserve uncertainty where they do not, and never invent an
experiment or observation. Follow the task's requested output format exactly.
Artifacts from the highest-priority physical branch have been copied into the
canonical scored workspace. If the task requests a file path, report its path
in the exact form requested by the task (prefer a workspace-relative path) and
never report a disposable branch-workspace path.
Do not write a paper, abstract, methods section, citations, review, or Markdown
wrapper. Put that exact answer in the final_answer field.
