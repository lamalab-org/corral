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

The task should be solved in less than **`{{iterations}}`** iterations.

**`{{examples}}`**

You are a high-level planner agent. Your job is to create a plan for another agent to solve the task described above.

Design the plan for completing the task described above in the designated number of iterations. The low-level agent executing the plan will have access to the tools mentioned above. The plan should clearly specify the sequence of steps to be taken, including which tools to use and how to use them effectively.

That plan will be executed by a low-level planner agent. Ensure that the plan is clear, concise, and actionable.

The low level agent will provide you with the results of executing the plan, including the steps taken, the outcomes and the final answer.

After receiving the answer from the low-level agent, reason whether the task is addressed.

If you believe the task is addressed, submit the answer using 'Final Answer: <the_answer>.

If you believe the task is not addressed, provide corrections to the plan or draft a new plan to better address the task. This process may be repeated multiple times, until you believe the task was successfully addressed by the low-level agent.
