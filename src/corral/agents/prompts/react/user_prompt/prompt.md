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

If you have the final answer, respond with:
<thought>[your reasoning]</thought>
<final_answer>[answer]</final_answer>

**`{{surrender_instructions}}`**

The code expects the XML tags to be used exactly as shown. If this format is not followed, the regex parsing will not work.
