Answer the user's request using the collected evidence.

Task and required output format:
{{task_prompt}}

Global research journal:
{{journal}}

Highest-priority experiment nodes:
{{best_nodes}}

Final output workspace:
{{canonical_workspace}}

Resolve contradictions based on the collected evidence.
Follow the task's requested output format exactly.
Artifacts from the highest-priority physical branch have been copied into the final output workspace.
If the task requests a file path, report its path in the exact form requested by the task (prefer a canonical `/workspace/...` path) and never report a disposable branch-workspace path.
Do not write a paper, abstract, methods section, citations, review, or Markdown wrapper.
Put that exact answer in the final_answer field.
