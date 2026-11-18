"""Private subprocess entry point for the pinned Denario/CMBagent SDKs.

Compatibility hooks live only in this process. They add environment access and
observability; native planning, routing, execution and review stay in the SDK.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4

from corral.agents.denario._bridge import Bridge
from corral.agents.denario._workflow import NativeWorkflow, completed_context

ENVIRONMENT_INSTRUCTIONS = """
Solve the supplied task and preserve its constraints. Obtain observations from
the task environment; never fabricate data or infer hidden task state.
The engineer and researcher can call the supplied environment tools directly.
Use their published JSON schemas for exact tool names and arguments. Use your
native code executor for Python calculations and file generation.
Files submitted by path must be written under the task workspace.
When an answer is ready, call propose_answer to record a candidate. You may
replace it as the analysis develops. This does not end the workflow: complete
the native plan, including the researcher's interpretation and final results.
The final researcher must check and update the candidate against those results.
After Denario finishes and saves its results, the adapter submits the latest
candidate. A prose response alone does not propose an answer.
"""


def install_native_hooks(bridge: Bridge):
    """Connect environment tools and observe the SDK without replacing its loop."""
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    from autogen.coding import LocalCommandLineCodeExecutor

    cmb = importlib.import_module("cmbagent.cmbagent")
    utils = importlib.import_module("cmbagent.utils")
    workflow = NativeWorkflow(bridge)
    call_context: ContextVar[str | None] = ContextVar("denario_tool_call", default=None)

    def model_config(_model, _api_keys):
        return {
            "model": bridge.client_model,
            "api_type": "openai",
            "api_key": bridge.token,
            "base_url": bridge.url + "/v1",
        }

    cmb.get_model_config = utils.get_model_config = model_config
    for name in utils.default_agent_llm_configs:
        utils.default_agent_llm_configs[name] = model_config(None, None)
    utils.default_llm_config_list[:] = [model_config(None, None)]

    original_register = cmb.register_functions_to_agents

    def register(instance):
        original_register(instance)
        for role in ("engineer", "researcher"):
            agent = instance.get_agent_from_name(role)
            for schema in workflow.tools:
                name = schema["function"]["name"]
                if name in agent.function_map:
                    raise ValueError(
                        f"Corral tool conflicts with a native Denario tool: {name}"
                    )

                agent.update_tool_signature(schema, is_remove=False)
                agent.register_function(
                    {name: workflow.function(instance, name, call_context)}
                )

    cmb.register_functions_to_agents = register
    original_solve = cmb.CMBAgent.solve

    @wraps(original_solve)
    def solve(instance, *args, **kwargs):
        result = original_solve(instance, *args, **kwargs)
        if instance.mode == "planning_and_control_context_carryover":
            workflow.final_context = instance.final_context.to_dict()
            bridge.event("native_step_finished", context=workflow.final_context)
        return result

    cmb.CMBAgent.solve = solve
    original_reply = ConversableAgent.generate_reply

    @wraps(original_reply)
    def generate_reply(agent, *args, **kwargs):
        previous, bridge.agent = bridge.agent, agent.name
        try:
            return original_reply(agent, *args, **kwargs)
        finally:
            bridge.agent = previous

    ConversableAgent.generate_reply = generate_reply
    original_async_reply = ConversableAgent.a_generate_reply

    @wraps(original_async_reply)
    async def a_generate_reply(agent, *args, **kwargs):
        previous, bridge.agent = bridge.agent, agent.name
        try:
            return await original_async_reply(agent, *args, **kwargs)
        finally:
            bridge.agent = previous

    ConversableAgent.a_generate_reply = a_generate_reply
    original_group_append = GroupChat.append

    @wraps(original_group_append)
    def append_group_message(group, message, speaker):
        original_group_append(group, message, speaker)
        # Persist the broadcast once, before delivery to any recipient. The
        # SDK sends this same message object separately to every other agent.
        bridge.event(
            "message",
            sender=speaker.name,
            recipient="groupchat",
            recipients=[agent.name for agent in group.agents if agent != speaker],
            message=group.messages[-1],
        )

    GroupChat.append = append_group_message
    original_append = ConversableAgent._append_oai_message

    @wraps(original_append)
    def append_message(agent, message, role, conversation_id, is_sending):
        appended = original_append(agent, message, role, conversation_id, is_sending)
        broadcast = (
            isinstance(agent, GroupChatManager)
            and agent.groupchat.messages
            and message is agent.groupchat.messages[-1]
        )
        if appended and is_sending and not broadcast:
            # Both sync and async sends pass here. Capture SDK transformations
            # and handoffs, not just the provider output that preceded them.
            # An event envelope keeps broadcasts from counting as new actions.
            bridge.event(
                "message",
                sender=agent.name,
                recipient=conversation_id.name,
                message=agent._oai_messages[conversation_id][-1],
            )
        return appended

    ConversableAgent._append_oai_message = append_message
    original_execute = ConversableAgent.execute_function
    original_async_execute = ConversableAgent.a_execute_function
    task_tools = {tool["function"]["name"] for tool in bridge.config["tools"]}

    def native_start(agent_name, func_call, call_id):
        bridge.rpc(
            "messages",
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "name": agent_name,
                    "sdk_native": True,
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": func_call}
                    ],
                }
            ],
        )

    def native_result(func_call, call_id, content, success, started):
        bridge.rpc(
            "messages",
            messages=[
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_call["name"],
                    "content": str(content),
                    "sdk_native": True,
                    "is_error": not success,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
            ],
        )

    def begin_function(agent, func_call, call_id):
        if func_call["name"] not in task_tools:
            # Corral deduplicates provider proposals and execution starts by ID.
            # SDK-generated calls may have no provider proposal or call ID.
            native_start(f"denario.{agent.name}", func_call, call_id)
        return call_context.set(call_id), time.monotonic()

    def finish_function(func_call, call_id, content, success, started):
        if func_call["name"] not in task_tools:
            native_result(func_call, call_id, content, success, started)

    @wraps(original_execute)
    def execute_function(agent, func_call, call_id=None, verbose=False):
        call_id = call_id or str(uuid4())
        token, started = begin_function(agent, func_call, call_id)
        try:
            success, result = original_execute(agent, func_call, call_id, verbose)
        except Exception as exc:
            finish_function(
                func_call, call_id, f"{type(exc).__name__}: {exc}", False, started
            )
            raise
        else:
            finish_function(func_call, call_id, result["content"], success, started)
            return success, result
        finally:
            call_context.reset(token)

    ConversableAgent.execute_function = execute_function

    @wraps(original_async_execute)
    async def a_execute_function(agent, func_call, call_id=None, verbose=False):
        call_id = call_id or str(uuid4())
        token, started = begin_function(agent, func_call, call_id)
        try:
            success, result = await original_async_execute(
                agent, func_call, call_id, verbose
            )
        except Exception as exc:
            finish_function(
                func_call, call_id, f"{type(exc).__name__}: {exc}", False, started
            )
            raise
        else:
            finish_function(func_call, call_id, result["content"], success, started)
            return success, result
        finally:
            call_context.reset(token)

    ConversableAgent.a_execute_function = a_execute_function
    original_code = LocalCommandLineCodeExecutor.execute_code_blocks

    @wraps(original_code)
    def execute_code(executor, code_blocks):
        call_id = str(uuid4())
        func_call = {
            "name": "denario.execute_code",
            "arguments": json.dumps(
                {
                    "blocks": [
                        {"language": block.language, "code": block.code}
                        for block in code_blocks
                    ]
                }
            ),
        }
        native_start("denario.executor", func_call, call_id)
        started = time.monotonic()
        try:
            result = original_code(executor, code_blocks)
        except Exception as exc:
            native_result(
                func_call, call_id, f"{type(exc).__name__}: {exc}", False, started
            )
            raise
        native_result(func_call, call_id, result.output, result.exit_code == 0, started)
        return result

    LocalCommandLineCodeExecutor.execute_code_blocks = execute_code
    _configure_plot_models(bridge)
    return workflow


def _configure_plot_models(bridge):
    """Route the old SDK's fixed VLM choices through the configured model."""
    from openai import OpenAI

    vlm = importlib.import_module("cmbagent.vlm_utils")
    # This release selects its OpenAI branch using a two-name allowlist. The
    # gateway applies the actual configured model, including newer model names.
    vlm.vlm_model = "gpt-4o"

    def plot_debugger(task_context, vlm_analysis, problems, executed_code):
        # Upstream hardcodes Gemini for this optional helper. Keep its inputs
        # and output schema, using the same provider as the rest of this run.
        result = OpenAI().chat.completions.parse(
            model=bridge.config["model"],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a plot debugging expert. Analyze the code and provide "
                        "concrete, targeted fixes for the identified problems. Focus on "
                        "root causes; reference code lines or sections where possible. "
                        "Group related problems into a single fix when appropriate.\n"
                        f"TASK CONTEXT:\n{task_context}\nVLM ANALYSIS:\n{vlm_analysis}\n"
                        f"PROBLEMS:\n{json.dumps(problems)}\nEXECUTED CODE:\n{executed_code}"
                    ),
                }
            ],
            response_format=vlm.PlotDebuggerResponse,
        )
        parsed = result.choices[0].message.parsed
        return parsed.fixes if parsed else []

    vlm.call_external_plot_debugger = plot_debugger
    # functions.py imports this symbol by value when the SDK loads.
    importlib.import_module(
        "cmbagent.functions"
    ).call_external_plot_debugger = plot_debugger


def run(config, bridge):
    try:
        from denario import Denario
        from denario.llm import LLM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Native Denario requires the 'denario' extra on Python 3.12/3.13. "
            "Build the sandbox with --agent denario --build-sandbox-image."
        ) from exc

    workflow = install_native_hooks(bridge)
    den = Denario(project_dir=config["project_dir"])
    den.set_data_description(
        ENVIRONMENT_INSTRUCTIONS
        + "\nTask:\n"
        + config["task"]
        + "\nAvailable environment tools (JSON schemas):\n"
        + json.dumps(workflow.tools, ensure_ascii=False)
    )
    # An LLM instance avoids Denario 1.0.1's outdated named-model registry.
    llm = LLM(
        # The pinned SDK chooses its client from the model name. Use its
        # OpenAI-compatible client for all gateway providers; the gateway
        # replaces this internal alias with the exact user-selected model.
        name=bridge.client_model,
        max_output_tokens=128000,
        temperature=config["model_settings"].get("temperature"),
    )
    bridge.set_phase("idea")
    den.get_idea_fast(llm=llm, iterations=config["denario_iterations"])
    bridge.set_phase("method")
    den.get_method_fast(llm=llm)
    bridge.set_phase("experiment")
    den.get_results(
        engineer_model=llm,
        researcher_model=llm,
        planner_model=llm,
        plan_reviewer_model=llm,
        orchestration_model=llm,
        formatter_model=llm,
        max_n_steps=config["max_plan_steps"],
        max_n_attempts=config["max_n_attempts"],
    )
    if not completed_context(workflow.final_context):
        raise RuntimeError("Denario returned without completing the native plan")
    bridge.rpc("done", context=workflow.final_context)


def main(*, writer=None):
    reader, writer = sys.stdin, writer or sys.stdout
    config = json.loads(reader.readline())
    os.chdir(config["project_dir"])
    # Reserve stdout for IPC; SDK text remains available in the worker log.
    sys.stdout = sys.stderr
    bridge = Bridge(config, reader, writer)
    bridge.start()
    try:
        run(config, bridge)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        bridge.rpc("error", error=f"{type(exc).__name__}: {exc}")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
