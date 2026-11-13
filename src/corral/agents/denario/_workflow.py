"""Connect external tools to the pinned SDK's native control context."""

from __future__ import annotations

import json
from copy import deepcopy
from inspect import unwrap
from uuid import uuid4

from corral.agents.denario._bridge import EnvironmentToolError


def completed_context(context):
    """A saved report alone does not prove that native control finished."""
    steps = context.get("number_of_steps_in_plan", 0)
    return (
        steps > 0
        and context.get("current_plan_step_number") == steps
        and context.get("current_status") == "completed"
        and context.get("n_attempts", 0) < context.get("max_n_attempts", 0)
    )


class NativeWorkflow:
    def __init__(self, bridge):
        self.bridge = bridge
        self.final_context = {}
        self.tools = deepcopy(bridge.config["tools"])
        if any(t["function"]["name"] == "propose_answer" for t in self.tools):
            raise ValueError("Environment tool conflicts with Denario's propose_answer")
        for tool in self.tools:
            function = tool["function"]
            if function["name"] == "submit_answer":
                function.update(
                    name="propose_answer",
                    description=(
                        "Record or replace the candidate final answer. This does not "
                        "end the experiment. Complete the native plan and researcher "
                        "results; the adapter submits the latest candidate only after "
                        "Denario finishes and saves its results."
                    ),
                )

    def invoke(self, instance, name, arguments, call_id):
        from autogen.agentchat.group import AgentTarget, ReplyResult

        # AutoPattern shares this context with all agents and its tool executor.
        context = instance.get_agent_from_name("engineer").context_variables
        if context["n_attempts"] >= context["max_n_attempts"]:
            # AG2 can execute several calls from one message before handing off.
            # Do not dispatch further state-changing tools after exhaustion.
            message = (
                "Tool not executed: the native per-step failure budget is exhausted."
            )
            if name != "propose_answer":
                self.bridge.rpc(
                    "messages",
                    messages=[
                        {
                            "role": "assistant",
                            "content": None,
                            "sdk_native": True,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "name": name,
                            "tool_call_id": call_id,
                            "content": message,
                            "sdk_native": True,
                            "is_error": True,
                            "not_executed": True,
                        },
                    ],
                )
            return ReplyResult(
                message=message,
                target=AgentTarget(instance.get_agent_from_name("terminator")),
                context_variables=context,
            )
        if name == "propose_answer":
            response = self.bridge.rpc(
                "proposal",
                arguments=arguments,
                id=call_id,
                role=context["agent_for_sub_task"],
                step=context["current_plan_step_number"],
            )
            if not response["success"]:
                raise EnvironmentToolError(response)
            return response["result"]
        try:
            return self.bridge.tool(name, arguments, call_id)
        except EnvironmentToolError as exc:
            if exc.status in {"invalid_args", "invalid_tool"}:
                # These are pre-execution mistakes, not failed experiments.
                raise
            formatter = instance.get_agent_from_name("executor_response_formatter")
            transfer = unwrap(formatter.function_map["post_execution_transfer"])
            previous = context["n_attempts"]
            result = transfer(
                next_agent_suggestion="control",
                context_variables=context,
                execution_status="failure",
                fix_suggestion=str(exc),
            )
            # The upstream helper only counts code-producing roles. External
            # tools can fail in researcher steps too; use the same shared budget.
            if context["n_attempts"] == previous:
                context["n_attempts"] += 1
            exhausted = context["n_attempts"] >= context["max_n_attempts"]
            if exhausted:
                result.target = AgentTarget(instance.get_agent_from_name("terminator"))
            result.message += (
                f"\nEnvironment tool {name} failed: {exc}\n"
                "The action was attempted once and may have changed the environment. "
                "Inspect its state before deciding whether to retry."
            )
            self.bridge.event(
                "environment_tool_recovery",
                tool_call_id=call_id,
                tool=name,
                status=exc.status,
                error=str(exc),
                step=context["current_plan_step_number"],
                n_attempts=context["n_attempts"],
                max_n_attempts=context["max_n_attempts"],
                target="terminator" if exhausted else "control",
            )
            return result

    def function(self, instance, name, call_context):
        def invoke(**arguments):
            return self.invoke(
                instance, name, arguments, call_context.get() or str(uuid4())
            )

        return invoke
