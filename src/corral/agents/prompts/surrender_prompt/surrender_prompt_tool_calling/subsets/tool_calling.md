---
uuid: bdb2dcc2-f61d-449f-8589-1cfc03f0b3a4
name: surrender_prompt_tool_calling
namespace: surrender_prompt
description: surrender instructions for ToolCalling agent
version: 1
tags:
- agents
- tool-calling
- surrender
variables: []
subset: tool_calling
created_at: '2025-11-13T15:41:27.537389+00:00'
updated_at: '2025-11-13T15:41:27.537389+00:00'
---

If you cannot complete this request, or you conclude that it cannot be done
with the information and tools available, call `submit_answer` with
`{"answer": "SURRENDER"}` instead of an answer.
