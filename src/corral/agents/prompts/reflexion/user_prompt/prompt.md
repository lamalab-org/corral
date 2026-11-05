---
uuid: f07ff386-ddc0-45a1-92f9-8fa098d2fac3
name: user_prompt
namespace: reflexion
description: Default user prompt for reflexion agent that instructs the agent to generate
  reflections based on previous attempts.
version: 1
tags:
- reflexion
- agent
- user-prompt
variables:
- task_description
- trajectory_summary
created_at: '2025-11-13T16:22:58.604863+00:00'
updated_at: '2025-11-13T16:22:58.604863+00:00'
---

Please review my previous attempt and suggest how to approach the task next.

Task:
{{task_description}}

Previous attempt (key steps):
{{trajectory_summary}}

Please provide a brief reflection (2-5 sentences) that:
1. Identifies what worked, what went wrong, and what remains unresolved
2. Suggests a specific strategy for the next attempt based on the observed results
3. Is actionable and directly applicable to solving this task
