---
uuid: 3fef090a-4a8a-4e47-bd7f-6b1b73cf8dda
name: user_prompt
namespace: planner_prompt
description: Zero-shot user prompt designed for the LLM-planner.
version: 1
tags:
- agents
- LLM-Planner
- user
- zero-shot
- planner
variables:
- examples
- iterations
- task_guide
- tools
created_at: '2025-11-13T15:41:27.535127+00:00'
updated_at: '2025-11-13T15:41:27.535127+00:00'
---

Description of the task to be addressed:

**`{{task_guide}}`**

For the intended task, these are the available tools:

**`{{tools}}`**

The task should be solved in less than **`{{iterations}}`**.

**`{{examples}}`**

Design a plan for completing the task described above using only the described tools in the designed number of iterations.

Another agent, with access to the tools, will try to solve the task by using the plan, so the plan should only describe step by step how to solve the task.

After receiving the answer from the low-level planner, reason if the task is addressed. If it is addressed, submit the answer using 'Final Answer: <the_answer>. Otherwise, generate another plan and repeat the process until you think the task is solved successfully. Do not include in the plan how to submit the answer.
