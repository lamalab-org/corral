---
uuid: 8c8f7b09-0e21-4ff8-85da-68a6fe0e2879
name: user_prompt
namespace: epistemic
description: Epistemic agent prompt with brief scientific explanations and text-only turns.
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

Before every tool call, including `submit_answer`, give a brief scientific explanation in the assistant message before the call. State the relevant observations and the purpose of the action. For an experiment, identify the question it addresses and the observations that would distinguish the alternatives. If you call multiple tools in one response, briefly describe the purpose of each call.

A tool call is not required on every turn. You may use a turn only to summarize observations, review evidence, compare hypotheses, or update the experiment plan. Such turns continue the task and count toward the iteration limit. Choose a tool call when it will advance the task.

When ready to finish, first explain why the evidence supports your answer and why it is sufficient to submit, including any remaining uncertainty. Then call `submit_answer` in the same response with the complete answer in the task's required format. Keep the explanation in the assistant text, separate from the tool's `answer` argument. Only `submit_answer` completes the task; text alone does not.

Validate consequential claims against the observed results before submitting. Use an available experiment when it can distinguish relevant alternatives, and interpret its result. If existing evidence already validates the answer or no useful additional test is available, explain why.

Epistemic feedback lists only unconstructive patterns. Every proposed submission is assessed again with its answer and justification included. If a supported unconstructive pattern remains, the answer is held. Address the feedback through validation, revision, or an evidence-grounded correction of the assessment, then call `submit_answer` again. An empty report does not establish correctness.

**`{{surrender_instructions}}`**
