from __future__ import annotations

import re
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING

import litellm
from litellm import completion
from litellm.caching import Cache
from loguru import logger

from .prompts import (
    BASELINESYSTEMPROMPT,
    BASELINEUSERPROMPT,
)
from .schemas import ReActOutput

if TYPE_CHECKING:
    from corral.base import Tool
    from corral.evaluate import BenchmarkInterface

litellm.cache = Cache()


class ModelName(Enum):
    CLAUDE_3_SONNET = "anthropic/claude-3-5-sonnet-latest"


class Temperature(Enum):
    ZERO = 0.0
    LOW = 0.3
    MEDIUM = 0.7
    HIGH = 1.0


class Agent(ABC):
    """Base class for agents"""

    def __init__(
        self,
        model: str = ModelName.CLAUDE_3_SONNET.value,
        temperature: float = Temperature.ZERO.value,
        timeout: int = 60,
        caching: bool = True,
        toolstore: list[Tool] | None = None,
        max_calls: int = 10,
        **model_kwargs,
    ):
        if toolstore is None:
            toolstore = []
        self.llm = {
            "model": model,
            "temperature": temperature,
            "timeout": timeout,
            "caching": caching,
        }
        self.toolstore = toolstore
        self._scratchpad = ""
        self.max_calls = max_calls
        self.model_kwargs = model_kwargs
        self.sys_prompt = BASELINESYSTEMPROMPT.format(guide="")
        self.user_prompt = BASELINEUSERPROMPT

    def format_messages(self):
        return [
            {
                "role": "system",
                "content": self.sys_prompt,
            },
            {
                "role": "user",
                "content": self.user_prompt,
            },
        ]

    def _call(self):
        return completion(
            model=self.llm["model"],
            temperature=self.llm["temperature"],
            messages=self.format_messages(),
            max_tokens=8192,
            caching=self.llm["caching"],
        )

    def _format_prompt(self, task: str, calls_left: int):
        system = self.sys_prompt.format(
            guide=task,
        )
        user = self.user_prompt
        return system, user

    def answer_question(self, interface: BenchmarkInterface, task_id: str) -> str:
        logger.debug("Running Simulation Agent")
        task = interface.get_task_guide(task_id)
        calls = 0
        while calls < self.max_calls:
            calls_left = self.max_calls - calls
            logger.debug(f"Scratchpad: {self._scratchpad}")
            system_prompt, user_prompt = self._format_prompt(
                task,
                calls_left,
            )
            logger.debug(f"Prompt: {system_prompt} {user_prompt}")
            answer = self._call()
            if "Final Answer:" in answer:
                return answer.split("Final Answer:")[1].strip()
            elif "Action:" in answer:
                action_name = re.findall(r"Action: (.*)", answer)[0].strip()
                action_input = re.findall(r"Action Input: (.*)", answer)[0]
                try:
                    observation = self.toolstore.run_tool(action_name, action_input)
                    self._scratchpad += (
                        f"Observation (from running {action_name} with {action_input}): "
                        + observation
                        + "\n"
                    )
                except Exception as e:
                    logger.error(e)
                    observation = "Error in running tool: " + str(e)
                    self._scratchpad += (
                        f"Observation (from running {action_name} with {action_input}): "
                        + observation
                        + "\n"
                    )

            calls += 1
            if calls >= self.max_calls:
                raise Exception("Max calls reached")
        return None


class ConstrainedAgent(ABC, Agent):
    """Base class for agents with constraints"""

    def __init__(
        self,
        model: str = ModelName.CLAUDE_3_SONNET.value,
        temperature: float = Temperature.ZERO.value,
        timeout: int = 60,
        caching: bool = True,
        toolstore: list[Tool] | None = None,
        max_calls: int = 10,
        **model_kwargs,
    ):
        if toolstore is None:
            toolstore = []
        super().__init__(
            model=model,
            temperature=temperature,
            timeout=timeout,
            caching=caching,
            toolstore=toolstore,
            max_calls=max_calls,
            **model_kwargs,
        )

    def _call(self):
        return self.client.chat.completions.create(
            model=self.llm["model"],
            max_retries=3,
            temperature=self.llm["temperature"],
            messages=self.format_messages(),
            max_tokens=8192,
            response_model=ReActOutput,
        )

    def _format_scratchpad(self, response) -> str:
        answer = ""
        thought = response.thought
        answer += f"Thought: {thought}\n"
        if response.action:
            action = response.action.action_type
            answer += f"Action: {action}\n"
            action_input = response.action.action_input
            answer += f"Action Input: {action_input}\n"
        fanswer = response.final_answer
        if fanswer:
            answer += f"Final Answer: {fanswer}\n"
        return answer

    def answer_question(self, interface: BenchmarkInterface, task_id: str) -> str:
        logger.debug("Running Simulation Agent")
        task = interface.get_task_guide(task_id)
        calls = 0
        while calls < self.max_calls:
            calls_left = self.max_calls - calls
            logger.debug(f"Scratchpad: {self._scratchpad}")
            system_prompt, user_prompt = self._format_prompt(
                task,
                calls_left,
            )
            logger.debug(f"Prompt: {system_prompt} {user_prompt}")
            answer = self._call(
                system_prompt,
                user_prompt,
                self.llm["model"],
                self.llm["temperature"],
                **self.model_kwargs,
            )

            _answer = self._format_scratchpad(answer)
            logger.debug(f"Answer: {_answer}")
            self._scratchpad += _answer
            if answer.final_answer:
                return answer.final_answer
            elif answer.action:
                if answer.action.action_type == "Describe Tool":
                    tool_name = answer.action.action_input
                    observation = self.toolstore.get_tool_str(tool_name)
                    self._scratchpad += f"Observation: {observation}\n"
                else:
                    tool_name = answer.action.action_type
                    tool_input = answer.action.action_input
                    try:
                        observation = self.toolstore.run_tool(tool_name, tool_input)
                        logger.debug(f"Observation: {observation}")
                        self._scratchpad += f"Observation (from running {tool_name} with {tool_input}):\n{observation}\n"
                    except Exception as e:
                        logger.error(e)
                        observation = f"Error in running tool: {e!s}"
                        self._scratchpad += f"Observation (from running {tool_name} with {tool_input}):\n{observation}\n"
            calls += 1
            if calls >= self.max_calls:
                raise Exception("Max calls reached")
        return None
