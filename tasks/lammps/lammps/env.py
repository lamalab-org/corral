import uvicorn
from loguru import logger
from tools import run_lammps

from corral.base import Environment, Tool

import os

from score import *

from corral.io import (
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    MkdirTool,
    CatFilesTool,
    FileInfoTool,
    CopyFileTool
)

from corral.server import create_benchmark_server


class LammpsEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        question: str,
        # answer: float,
        io_tools: dict[str, Tool],
        output: list[dict[str, str]],
        answer_file : str,
        scoring_fn : str
    ):
        self.question = question
        # self.correct_answer = answer
        # self.threshold = threshold
        self.answer_file = answer_file
        self.output = output
        self.scoring_fn = scoring_fn

        # The answer file path can be made configurable.
        # Here we use a virtual file path if using the local ("file") protocol,
        # self.answer_file = os.environ.get(
        #     "MODAL_BASE_IO_PATH", f"/results/final_benchmark/temp_0/claude_37/energy_minimisation/{task_id}/"
        # )

        # Initialize environment with the given task id.
        super().__init__(task_id)

        # Add math-related tools
        self.add_tool(run_lammps)

        # Add I/O tools so that the agent can read/write answer files.
        # (Agents will call these using submissions like "write_file" with appropriate arguments.)
        self.add_tool(io_tools["read_file"])
        self.add_tool(io_tools["write_file"])
        self.add_tool(io_tools["list_files"])
        self.add_tool(io_tools["file_info"])
        self.add_tool(io_tools["copy_file"])
        self.add_tool(io_tools["cat_files"])
        self.add_tool(io_tools["mkdir"])

    def get_task_prompt(self) -> str:
        return f"{self.question} Make sure all the associated files for this task (input files, log files, any other files) are in the {self.answer_file} directory. You can use the available I/O tools (e.g., write_file) make new files .etc. Whatever potentials you need to run the simulation, you can find them at /potentials/. A type of potential can be accessed by /potentials/TYPE where TYPE can be [EAM, TERSOFF] which further contains the exact potential files. Potential files have been taken from the original sources, and hence are correct. Do not try to read the entire potential files at once, they are too large it will crash the program. If the task is to give the final output as a scalar, only return the numerical value, without any units."

    def score(self) -> float:
        """Score based on reading the answer from the designated answer file."""
        
        if self.state.submitted_answer is None:
            return 0.0
        try:       
            if self.scoring_fn == "energy_minimisation":
                return energy_minimisation(self.answer_file, self.state.submitted_answer, self.output)
            elif self.scoring_fn == "check_numerical":
                return check_numerical(self.answer_file, self.state.submitted_answer, self.output[0])
            elif self.scoring_fn == "check_structure":
                return check_structure(self.answer_file, self.state.submitted_answer, self.output[0])
            elif self.scoring_fn == "check_stress_strain":
                return check_stress_strain(self.answer_file, self.state.submitted_answer, self.output[0])
            elif self.scoring_fn == "check_stress_tensor":
                return check_stress_tensor(self.answer_file, self.state.submitted_answer, self.output[0])
        except ValueError:
            return 0.0
    

if __name__ == "__main__":
    import os

    def create_tools(fs_manager: FSManager) -> dict[str, Tool]:
        return {
            "list_files": ListFilesTool(fs_manager),
            "read_file": ReadFileTool(fs_manager),
            "write_file": WriteFileTool(fs_manager),
            "file_info": FileInfoTool(fs_manager),
            "copy_file": CopyFileTool(fs_manager),
            "mkdir": MkdirTool(fs_manager),
            "cat_files": CatFilesTool(fs_manager)
        }

    fs_protocol = os.environ.get("CORRAL_FS_PROTOCOL", "file")
    fs_kwargs = {}  # add more keyword options from config if needed
    fs_manager = FSManager(protocol=fs_protocol, app="simagent")
    io_tools = create_tools(fs_manager)
    # Create environments for different tasks
    # environments = {
    #     "task_1": LammpsEnvironment(
    #         "task_1",
    #         "Simulate a face centered cubic lattice structure for Aluminum using LAMMPS with a lattice constant of 4.05 angstrom. The simulation should use metal units, define a 5x5x5 simulation box, and set periodic boundary conditions to True in all directions. Perform energy minimisation using the Embedded Atom Method (EAM) potential and use conjugate gradient as the minimiser, with force tolerance of 1e-10, energy tolerance of 1e-10, maximum iterations of 1e+5 and maximum evaluations of 1e+5. As the final output, give the total energy after minimisation in eV units.",
    #         -1790.00,
    #         io_tools=io_tools,
    #         threshold=1e-2, 
    #         answer_file="/results/temp/"

    #     ),
    # }

    # with app.run():
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
