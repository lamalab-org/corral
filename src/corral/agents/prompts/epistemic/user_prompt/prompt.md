---
uuid: 8c8f7b09-0e21-4ff8-85da-68a6fe0e2879
name: user_prompt
namespace: epistemic
description: Epistemic agent prompt with visible action explanations and reasoning-only turns.
version: 1
tags:
- agents
- tool-calling
- epistemic
variables:
- surrender_instructions
- task_guide
created_at: '2025-11-13T15:41:27.537950+00:00'
updated_at: '2025-11-13T15:41:27.537950+00:00'
---

Task Guide:

**`{{task_guide}}`**

Before every tool call, including `submit_answer`, write a concise reasoning
summary in visible assistant text in the same response, before the tool call.
Connect the relevant task facts or observations to your decision: explain why
this action is useful, which question or uncertainty it addresses, and how its
result could affect your conclusion. If you call multiple tools in one response,
identify each tool and explain the purpose of each call before making the calls.
The explanation must be visible text so it is captured in the trajectory; an
internal reasoning field alone does not fulfill this requirement.

A tool call is not required on every turn. You may use a turn only to explain
your reasoning, review evidence, compare hypotheses, or revise your plan. Such
turns continue the task and count toward the iteration limit. Choose a tool call
when it will advance the task.

When ready to finish, first explain why the evidence supports your answer and
why it is sufficient to submit, including any remaining uncertainty. Then call
`submit_answer` in the same response with the complete answer in the task's
required format. Keep the explanation in the assistant text, separate from the
tool's `answer` argument. Only `submit_answer` completes the task; text alone
does not.

**`{{surrender_instructions}}`**
