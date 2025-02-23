import re
from abc import ABC, abstractmethod
from enum import Enum

import litellm
from litellm import completion
from litellm.caching.caching import Cache
from loguru import logger

from corral.base import Tool
from corral.evaluate import BenchmarkInterface

from .prompts import (
    BASELINESYSTEMPROMPT,
    BASELINEUSERPROMPT,
)

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
        logger.debug(f"System Prompt: {system}")
        logger.debug(f"calls_left: {calls_left}")
        user = self.user_prompt
        return system, user

    @abstractmethod
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
                if isinstance(answer, str) and "Final Answer:" in answer:
                    return answer.split("Final Answer:")[1].strip()
                elif isinstance(answer, str) and "Action:" in answer:
                    action_name = re.findall(r"Action: (.*)", answer)[0].strip()
                    action_input = re.findall(r"Action Input: (.*)", answer)[0]
                    try:
                        tool = next(
                            tool for tool in self.toolstore if tool.name == action_name
                        )
                        observation = tool.run(action_input)
                        self._scratchpad += (
                            f"Observation (from running {action_name} with {action_input}): "
                            + observation
                            + "\n"
                        )
                    except Exception as e:
                        logger.error(e)
                        observation = f"Error in running tool: {e!s}"
                        self._scratchpad += (
                            f"Observation (from running {action_name} with {action_input}): "
                            + observation
                            + "\n"
                        )
                        self._scratchpad += (
                            f"Observation (from running {action_name} with {action_input}): "
                            + observation
                            + "\n"
                        )

                calls += 1
                if calls >= self.max_calls:
                    raise Exception("Max calls reached")
        return ""
