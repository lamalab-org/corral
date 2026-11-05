---
uuid: 1ee256eb-f466-4dba-b99e-608732f8e4b9
name: surrender_prompt_react
namespace: surrender_prompt
description: surrender instructions for ReAct agent with XML format
version: 1
tags:
- agents
- react
- surrender
variables: []
subset: react
created_at: '2025-11-13T15:41:27.536794+00:00'
updated_at: '2025-11-13T15:41:27.536794+00:00'
---

If you cannot complete this request, or you conclude that it cannot be done
with the information and tools available, submit this action instead of an
answer:
<thought>[why the request cannot be completed]</thought>
<action>submit_answer</action>
<action_input>{"answer": "SURRENDER"}</action_input>
