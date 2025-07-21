# API Specs For Environment, Tools and Agents

This document describes the API for environment, tools and agents to enable consistent contributions.


## Environment API

```
# to host environment URL interface
import uvicorn

# for environment functionality
from corral.base import Environment
from corral.server import create_benchmark_server

# add environment tools
from tools import (
    tool_1, tool_2, tool_3, tool_4)

tool_list = [
    tool_1, tool_2, tool_3, tool_4]

class SampleEnvironment(Environment):
    def __init__(self, task_id: str, question: str, answer: float = None):
        self.question = question
        self.correct_answer = answer
        super().__init__(task_id)

        # Add multiple tools
        for tool in tool_list:
            self.add_tool(tool)

    def get_task_prompt(self) -> str:
        return f"Solve this problem: {self.question}"

    def score(self) -> float:
        """Score based on submitted answer"""
        if self.state.submitted_answer is None:
            return 0.0
        try:
            """ Write the score function for the environment task"""
        except ValueError:
            return 0.0


if __name__ == "__main__":
    # Create environments for different tasks
    environments = {
        "env_1": SampleEnvironment("env_1", "Question 1?", Answer_1),
        "env_2": SampleEnvironment("env_2", "Question 2?", Answer_2)
        "env_3": SampleEnvironment("env_3", "Question 3?", Answer_3),
    }

    # Create and run server
    with app.run():
        app = create_benchmark_server(environments)
        uvicorn.run(app, host="0.0.0.0", port=8000)

```


## Tools API


```
from __future__ import annotations


# import tool utilities
from corral.base import Tool, ToolArgument
from corral.utils import modal_tool, tool



@tool
def tool1(arg1: [str, float, int], arg2: [str, float, int]) -> [str, float, int]:
    """Perform basic math operations.

    Args:
        Describe the arguments here
    """

    result = function(arg1, arg2)

    return result

### Example percentage calculator

@tool
def percentage_calculator(value: float, percentage: float = 100.0) -> float:
    """Calculate percentage of a value.

    Args:
        value: The base value
        percentage: The percentage to calculate (defaults to 100.0)

    Returns:
        float: The calculated result
    """
    return (value * percentage) / 100.0



```


## Agent API - Basic Agent


```

import json

from dotenv import load_dotenv
from prompts import BASELINESYSTEMPROMPT, BASELINEUSERPROMPT

from corral.evaluate import BenchmarkInterface, MatAgentBenchmark

from prompts import BASELINESYSTEMPROMPT, BASELINEUSERPROMPT
from openai import OpenAI


class BaseAgent:
    """General agent implementation"""

    def __init__(self, base_url: str, tools: list(str), args, **kwargs):

        # Base Agent API implementation

        self.client = OpenAI(base_url=base_url, api_key="dummy")

        self.tools = tools


    def solve_task(self, interface: BenchmarkInterface, task_id: str) -> str:
    	    """ Solve Task function to solve task in the environment """
        guide = interface.get_task_guide(task_id)

        system_prompt = BASELINESYSTEMPROMPT.format(guide=guide)

        # Initialize messages with a starter message
        messages = [{"role": "system", "content": system_prompt,
                    "role": "user", "content": BASELINEUSERPROMPT,
                    "role": "task", "content": guide}]

        while True:

            response = self.client.chat.completions.create(
                model='model_name',
                max_tokens=1024,
                messages=messages,
                tools=self.tools)

            message = response.choices[0].message.content

            messages.append({"role": "assistant", "content": message})

            if "FINAL ANSWER:" in message:
                print(f"Final answer: {message.split('FINAL ANSWER:')[1].strip()}")
                return message.split("FINAL ANSWER:")[1].strip()

            if "TOOL CALL:" in message:
                try:
                    print(f"Tool call: {message.split('TOOL CALL:')[1].strip()}")
                    tool_json = message.split("TOOL CALL:")[1].strip()
                    tool_request = json.loads(tool_json)
                    result = interface.execute_tool(
                        task_id, tool_request["tool_name"], tool_request["arguments"]
                    )
                    if result.success:
                        messages.append(
                            {"role": "user", "content": f"Tool result: {result.result}"}
                        )
                    else:
                        messages.append(
                            {"role": "user", "content": f"Error: {result.error}"}
                        )
                except Exception as e:
                    messages.append({"role": "user", "content": f"Error: {e!s}"})


```

## Calling the Benchmark in Full

```

if __name__ == "__main__":
    import os

    # Create components
    interface = BenchmarkInterface()

    # Load Tools from Environment



    # Load Agent and Runner
    agent = BaseAgent(base_url='agent_url')
    runner = MatAgentBenchmark(interface, agent)

    # Run benchmark
    result = runner.bench()
    print("Benchmark completed:")
    print(f"Average score: {result.average_score}")
    print(f"Tasks completed: {result.successful_tasks}/{result.total_tasks}")

    # Print detailed results
    for task_id, task_result in result.task_results.items():
        print(f"\nTask {task_id}:")
        print(f"Score: {task_result.score}")
        print(f"Tool usage: {task_result.tool_statistics}")


```
