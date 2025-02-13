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