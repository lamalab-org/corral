import uvicorn
from loguru import logger
from tools import run_lammps, get_structure_from_mp_text, convert_structure_to_lammps_data, get_potential_metadata, extract_max_stress
from pathlib import Path
from corral.base import Environment, Tool

import os

from pathlib import Path

from score import check_numerical

from corral.io import (
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    MkdirTool,
    CatFilesTool,
    FileInfoTool,
    CopyFileTool,
    GrepTool
)

import os

class LammpsEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        question: str,
        # answer: float,
        output: list[dict[str, str]],
        work_dir : str,
        scoring_fn : str
    ):
        self.question = question
        # self.correct_answer = answer
        # self.threshold = threshold
        self.work_dir = work_dir
        self.output = output
        self.scoring_fn = scoring_fn

        # Initialize environment with the given task id.
        fs_manager = FSManager("file", base_path=work_dir, app="simagent")
        super().__init__(task_id, base_work_dir=work_dir, fs_manager=fs_manager)
        
        # Add math-related tools
        self.add_tool(run_lammps)
        self.add_tool(extract_max_stress)
        self.add_tool(get_structure_from_mp_text)
        self.add_tool(convert_structure_to_lammps_data)
        self.add_tool(get_potential_metadata)

        self._setup_file_tools()

    def _setup_file_tools(self):
        """Setup file tools for current workspace"""
        if self.current_work_dir:
            logger.info(
                f"DEBUG: Setting up FSManager with base_path: {self.current_work_dir}"
            )
            # Create new FSManager for current workspace
            fs_manager = FSManager("file", base_path=self.current_work_dir, app="simagent")

            # Add/update file tools
            self.tools.update(
                {
                    "list_files": ListFilesTool(fs_manager),
                    "read_file": ReadFileTool(fs_manager),
                    "write_file": WriteFileTool(fs_manager),
                    "file_info": FileInfoTool(fs_manager),
                    "cat_files": CatFilesTool(fs_manager),
                    "copy_file": CopyFileTool(fs_manager),
                    "grep_tool" : GrepTool(fs_manager),
                }
            )
            logger.info(
                f"DEBUG: File tools setup complete for workspace: {self.current_work_dir}"
            )
        else:
            logger.warning("DEBUG: No current_work_dir set, skipping file tools setup")
    
    def reset_state(self) -> str:
        """Reset state and update file tools for new workspace"""
        trial_id = super().reset_state()
        # Recreate file tools for new workspace
        self._setup_file_tools()
        return trial_id

    def get_task_prompt(self) -> str:
        prompt = f"{self.question} Make sure all the associated files for this task (input files, log files, any other files) are in the {self.current_work_dir} directory. You can use the available I/O tools (e.g., write_file) make new files .etc. Whatever potentials you need to run the simulation, you can find them at /potentials/. A type of potential can be accessed by /potentials/TYPE where TYPE can be [EAM, TERSOFF, REAXFF] which further contains the exact potential files. Note that in case of reaxff potentials, pair style 'reax/c' has been renamed to 'reaxff' and always use NULL for the control file (cfile), for example, this syntax is correct : pair_style reaxff NULL. If the task is to give the final output as a scalar, only return the numerical value, without any units." 
        # Add workspace info
        # if self.current_work_dir:
        #     prompt += f"\nIMPORTANT: You have access to filesystem tools. All files, if generated, should be saved in your isolated workspace. The isolated workspace is located at {self.current_work_dir}.\n"
            # prompt += f"For this task, the afm image will be saved at C:\\Users\\Admin\\Desktop\\corral\\mat-agent-bench\\tasks\\afm\\afm\\afm_images\\{self.file}_{pointer_}.nid\n"
        logger.info(f"prompt : {prompt}")
        return prompt

    # def get_task_prompt(self) -> str:
    #     self.set_workdir_for_trail()
    #     self.answer_file = Path(self.work_dir).joinpath(self._task_trail_id_name)
    #     # fs_protocol = os.environ.get("CORRAL_FS_PROTOCOL", "file")
    #     # fs_manager = FSManager(protocol=fs_protocol, app="simagent")
    #     # fs_manager.mkdir(os.path.join(self.answer_file), create_parents=True)
    #     return f"{self.question} Make sure all the associated files for this task (input files, log files, any other files) are in the {self.answer_file} directory. You can use the available I/O tools (e.g., write_file) make new files .etc. Whatever potentials you need to run the simulation, you can find them at /potentials/. A type of potential can be accessed by /potentials/TYPE where TYPE can be [EAM, TERSOFF, REAXFF] which further contains the exact potential files. Note that in case of reaxff potentials, pair style 'reax/c' has been renamed to 'reaxff' and always use NULL for the control file (cfile), for example, this syntax is correct : pair_style reaxff NULL. If the task is to give the final output as a scalar, only return the numerical value, without any units."

    def score(self) -> float:
        """Score based on reading the answer from the designated answer file."""
        
        if self.state.submitted_answer is None:
            logger.warning(f"No submission found for task {self.task_id}")
            return 0.0
        try:       
            answer_value = self.state.submitted_answer.strip()
            if self.scoring_fn == "check_numerical":
                score = check_numerical(answer_value, self.output[0])
                return score
            # if self.scoring_fn == "energy_minimisation":
            #     return energy_minimisation(self.answer_file, self.state.submitted_answer, self.output)
            # elif self.scoring_fn == "check_numerical":
            #     return check_numerical(self.answer_file, self.state.submitted_answer, self.output[0])
            # elif self.scoring_fn == "check_structure":
            #     return check_structure(self.answer_file, self.state.submitted_answer, self.output[0])
            # elif self.scoring_fn == "check_stress_strain":
            #     return check_stress_strain(self.answer_file, self.state.submitted_answer, self.output[0])
            # elif self.scoring_fn == "check_stress_tensor":
            #     return check_stress_tensor(self.answer_file, self.state.submitted_answer, self.output[0])
        except ValueError:
            return 0.0
    

# if __name__ == "__main__":
#     import os

#     def create_tools(fs_manager: FSManager) -> dict[str, Tool]:
#         return {
#             "list_files": ListFilesTool(fs_manager),
#             "read_file": ReadFileTool(fs_manager),
#             "write_file": WriteFileTool(fs_manager),
#             "file_info": FileInfoTool(fs_manager),
#             "copy_file": CopyFileTool(fs_manager),
#             "mkdir": MkdirTool(fs_manager),
#             "cat_files": CatFilesTool(fs_manager),
#             # "read_large_file" : ReadLargeFileTool(fs_manager)
#         }

#     fs_protocol = os.environ.get("CORRAL_FS_PROTOCOL", "file")
#     fs_kwargs = {}  # add more keyword options from config if needed
#     fs_manager = FSManager(protocol=fs_protocol, app="simagent")
#     io_tools = create_tools(fs_manager)
    #Create environments for different tasks
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

    # # with app.run():
    # app = create_benchmark_server(environments)
    # uvicorn.run(app, host="0.0.0.0", port=8000)
