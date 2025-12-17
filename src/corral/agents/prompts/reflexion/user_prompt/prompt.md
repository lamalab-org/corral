---
uuid: f07ff386-ddc0-45a1-92f9-8fa098d2fac3
name: user_prompt
namespace: reflexion
description: Default user prompt for reflexion agent that instructs the agent to generate
  reflections based on previous trials.
version: 1
tags:
- reflexion
- agent
- user-prompt
variables:
- score
- task_description
- task_id
- trajectory_summary
- trial_id
created_at: '2025-11-13T16:22:58.604863+00:00'
updated_at: '2025-11-13T16:22:58.604863+00:00'
---

You are a reflection module analyzing a failed attempt at solving a task.

Your goal is to provide a concise, actionable reflection that will help improve performance on the next attempt.

Task:
**`{{task_id}}`** - **`{{trial_id}}`**
- Task description: **`{{task_description}}`**

- Score achieved: **`{{score}}`**

Trajectory (key steps):
**`{{trajectory_summary}}`**

Please provide a brief reflection (2-5 sentences) that:
1. Identifies what went wrong in this attempt
2. Suggests a specific strategy to avoid this mistake in the next attempt
3. Is actionable and directly applicable to solving this task
