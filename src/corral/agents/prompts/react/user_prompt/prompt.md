---
uuid: b05ea11f-c0a7-49cc-90cf-16d7399551d7
name: user_prompt
namespace: react
description: Very basic REACT prompt. Intended for use with LLM agents.
version: 1
tags:
- agents
- react
variables:
- examples
- surrender_instructions
- task_guide
created_at: '2025-11-13T15:41:27.533078+00:00'
updated_at: '2025-11-13T15:41:27.533078+00:00'
---

Task Guide: **`{{task_guide}}`**

**`{{examples}}`**

Think about what to do next and respond in the following format:

<thought>[your reasoning]</thought>
<action>[tool name]</action>
<action_input>[tool arguments as JSON]</action_input>. For tool calls without arguments, use `<action_input>{}</action_input>`

When the task is complete, call the `submit_answer` tool using the same action
format and pass the complete answer in its `answer` argument. Do not return a
final answer as plain text or with final-answer tags.

**`{{surrender_instructions}}`**

The code expects the action XML tags to be used exactly as shown. If this
format is not followed, the action parser will not work.
