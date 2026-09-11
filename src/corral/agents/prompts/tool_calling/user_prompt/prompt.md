---
uuid: c891017a-33ba-4b71-a639-67aa0e5e1d94
name: user_prompt
namespace: tool_calling
description: Tool Calling user prompt (without tool description). Intended for use
  with tool calling agents.
version: 1
tags:
- agents
- tool-calling
variables:
- surrender_instructions
- task_guide
created_at: '2025-11-13T15:41:27.537950+00:00'
updated_at: '2025-11-13T15:41:27.537950+00:00'
---

Task Guide:

**`{{task_guide}}`**

You must think what to do next. You can use some of the tools available in the system.

When the task is complete, call the `submit_answer` tool with the complete
answer. Do not return a final answer as plain text.

**`{{surrender_instructions}}`**
