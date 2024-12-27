from __future__ import annotations

import uvicorn
from tools import CalculatorTool, UnitConverterTool

from corral.base import Environment, ToolCallStatus
from corral.server import create_benchmark_server


class MathEnvironment(Environment):
    def __init__(self, task_id: str, question: str, answer: float):
        self.question = question
        self.correct_answer = answer
        super().__init__(task_id)

        # Add multiple tools
        self.add_tool(CalculatorTool())
        self.add_tool(UnitConverterTool())

    def get_task_prompt(self) -> str:
        return f"Solve this math problem: {self.question}"

    def score(self) -> float:
        """Score based on submitted answer"""
        if self.state.submitted_answer is None:
            return 0.0
        try:
            submitted_result = float(self.state.submitted_answer)
            return 1.0 if abs(submitted_result - self.correct_answer) < 0.001 else 0.0
        except ValueError:
            return 0.0


if __name__ == "__main__":
    # Create environments for different tasks
    environments = {
        "math_1": MathEnvironment("math_1", "What is 23 + 45?", 68),
        "math_2": MathEnvironment("math_2", "What is 12 * 8?", 96),
        "math_3": MathEnvironment("math_3", "What is 99 * 63 * 999 * 111?", 691614693),
    }

    # Create and run server
    app = create_benchmark_server(environments)

    uvicorn.run(app, host="0.0.0.0", port=8000)
