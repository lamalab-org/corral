from __future__ import annotations

import uvicorn
# from tools import UnitConverterTool, app, calculator, number_converter

from tools import run_bash_command, run_lammps

from corral.base import Environment
from corral.server import create_benchmark_server


class LammpsEnvironment(Environment):
    def __init__(self, task_id: str, question: str, answer: float, tools: list, threshold: float):
        self.question = question
        self.correct_answer = answer
        self.threshold = threshold
        super().__init__(task_id)

        # Add multiple tools
        for tool in tools:
            self.add_tool(tool)
        # self.add_tool(run_lammps)
        # self.add_tool(run_bash_command)

    def get_task_prompt(self) -> str:
        return f"{self.question} Whatever potentials you need to run the simulation, you can find them at /potentials/EAM/. Do not change your working directory, it has already been set. If the task is to give the final output as a scalar, only return the numerical value, without any units."

    def score(self) -> float:
        """Score based on submitted answer"""
        if self.state.submitted_answer is None:
            return 0.0
        try:
            submitted_result = float(self.state.submitted_answer)
            return 1.0 if abs(submitted_result - self.correct_answer) < self.threshold else 0.0
        except ValueError:
            return 0.0


# if __name__ == "__main__":
#     # Create environments for different tasks
#     lattice_prompt = "Simulate an aluminum lattice in lammps with simulation box of size 1x1x1 and periodic boundary conditions. Do not perform energy minimisation. As a final answer print the total number of atoms present in the simulation. Only give the numerical value, nothing else."
#     em_prompt = "Simulate an aluminum lattice in lammps with simulation box of size 1x1x1 and periodic boundary conditions. As the final output, give the total energy in eV, only the scalar value without any units."
#     environments = {
#         "task_1": LammpsEnvironment("task_1", em_prompt, -13.4177872966398)

        # "math_1": MathEnvironment("math_1", "What is 23 + 45?", 68),
        # "math_2": MathEnvironment("math_2", "What is 12 * 8?", 96),
        # "math_3": MathEnvironment("math_3", "What is 99 * 63 * 999 * 111?", 691614693),
        # "math_4": MathEnvironment(
        #     "math_4",
        #     "What is twenty one thousand four hundred and seventy three * twenty one thousand four hundred and seventy three?",
        #     4666829,
        # ),
    # }

    # Create and run server
    # with app.run():
    # app = create_benchmark_server(environments)
    # uvicorn.run(app, host="0.0.0.0", port=8000)
