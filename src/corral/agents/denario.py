"""Denario planning followed by Corral's native tool-calling execution loop."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from corral.agents.schema import AgentOutcome
from corral.agents.tool_calling import ToolCallingAgent
from corral.core.errors import concise_error_message

if TYPE_CHECKING:
    from corral.agents.session import AgentSession


class DenarioAgent(ToolCallingAgent):
    """Use Denario's fast idea/method stages to guide real environment actions.

    This is a hybrid planner/executor, not Denario's native ``cmbagent``
    experiment workflow. ``denario_model`` is a name in Denario's model registry;
    ``model`` and other ToolCallingAgent options configure execution only.
    Usage and the session iteration limit cover execution, not planning.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_endpoint: str | None = None,
        temperature: float = 1.0,
        *,
        denario_model: str = "gpt-4o",
        denario_iterations: int = 2,
        **kwargs: Any,
    ) -> None:
        if not isinstance(denario_model, str) or not denario_model.strip():
            raise ValueError("denario_model must be a non-empty Denario model name")
        if (
            isinstance(denario_iterations, bool)
            or not isinstance(denario_iterations, int)
            or denario_iterations < 1
        ):
            raise ValueError("denario_iterations must be a positive integer")
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            temperature=temperature,
            **kwargs,
        )
        self.denario_model = denario_model
        self.denario_iterations = denario_iterations

    def _generate_plan(self, task: str, tools: list[dict[str, Any]]) -> tuple[str, str]:
        # Import only when planning, keeping the optional SDK out of core imports.
        try:
            from denario import Denario
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "DenarioAgent requires the 'denario' extra (Python 3.12 or 3.13). "
                "Install it from the repository root with `uv sync --extra denario`."
            ) from None
        # Each session gets an isolated project; only its public plan is retained
        # in canonical agent state. Denario must not modify the task workspace.
        with TemporaryDirectory(prefix="corral-denario-") as project_dir:
            den = Denario(project_dir=project_dir)
            den.set_data_description(
                "Plan how to solve the following Corral task. Preserve its goal "
                "and constraints; do not invent a different research question. "
                "Experiments will be performed by an executor using only the "
                "provided environment tools. Do not assume access to local data "
                "files, fabricate observations, or treat a plan as a result.\n\n"
                f"Task:\n{task}\n\n"
                f"Available tools (JSON schemas):\n{json.dumps(tools, ensure_ascii=False)}\n"
            )
            den.get_idea_fast(
                llm=self.denario_model, iterations=self.denario_iterations
            )
            den.get_method_fast(llm=self.denario_model)
            inputs = Path(project_dir) / "input_files"
            idea = (inputs / "idea.md").read_text(encoding="utf-8").strip()
            method = (inputs / "methods.md").read_text(encoding="utf-8").strip()
            if not idea or not method:
                raise ValueError("Denario produced an empty idea or methodology")
            return idea, method

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Restore or generate a plan, then execute it through the same session."""
        metadata = {
            "denario_mode": "planning_with_tool_calling",
            "denario_model": self.denario_model,
            "denario_iterations": self.denario_iterations,
            "usage_scope": "executor_only",
        }
        state = session.get_agent_state("denario") or {}
        idea, method = state.get("idea"), state.get("method")
        if not all(
            isinstance(value, str) and value.strip() for value in (idea, method)
        ):
            prompt = session.prompt
            task = (
                prompt
                if isinstance(prompt, str)
                else json.dumps(prompt, ensure_ascii=False)
            )
            try:
                idea, method = await asyncio.to_thread(
                    self._generate_plan, task, list(session.tools)
                )
            except Exception as exc:
                return AgentOutcome(
                    status="agent_failure",
                    error=f"Denario planning failed: {concise_error_message(exc)}",
                    metadata=metadata,
                )
            await session.set_agent_state("denario", {"idea": idea, "method": method})

        handoff = (
            "Denario proposed the following research plan. Treat it as a "
            "hypothesis, not verified results. Execute it using the actual "
            "Corral tools, adapt to observations and the original task's "
            "constraints, and submit the final answer with submit_answer.\n\n"
            f"Research idea:\n{idea}\n\nMethodology:\n{method}"
        )
        if not any(
            message.get("role") == "user" and message.get("content") == handoff
            for message in session.messages
        ):
            await session.record_message(
                {"role": "user", "content": handoff}, llm_calls=0
            )

        # Reuse the complete native loop, including hooks, surrender, usage,
        # iteration limits and tool submission, without introducing another run.
        outcome = await super().run_session(session)
        return replace(outcome, metadata={**outcome.metadata, **metadata})
