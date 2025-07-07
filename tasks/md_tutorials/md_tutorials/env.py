import json
import os
from pathlib import Path

from loguru import logger
from score import (
    check_numerical,
    check_stress_strain,
    check_stress_tensor,
    check_structure,
    energy_minimisation,
)
from tools import (
    convert_structure_to_lammps_data,
    extract_max_stress,
    get_potential_metadata,
    get_structure_from_mp_text,
    run_lammps,
)

from corral.base import Environment
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    GrepTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from corral.server import run_server


class LammpsEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        question: str,
        output: list[dict[str, str]],
        work_dir: str,
        scoring_fn: str,
    ):
        self.question = question
        self.output = output
        self.work_dir = work_dir
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
        if self.work_dir:
            logger.info(f"DEBUG: Setting up FSManager with base_path: {self.work_dir}")
            # Create new FSManager for current workspace
            fs_manager = FSManager("file", base_path=self.work_dir, app="simagent")

            # Add/update file tools
            self.tools.update(
                {
                    "list_files": ListFilesTool(fs_manager),
                    "read_file": ReadFileTool(fs_manager),
                    "write_file": WriteFileTool(fs_manager),
                    "file_info": FileInfoTool(fs_manager),
                    "cat_files": CatFilesTool(fs_manager),
                    "copy_file": CopyFileTool(fs_manager),
                    "grep_tool": GrepTool(fs_manager),
                }
            )
            logger.info(
                f"DEBUG: File tools setup complete for workspace: {self.work_dir}"
            )
        else:
            logger.warning("DEBUG: No work_dir set, skipping file tools setup")

    def reset_state(self) -> str:
        """Reset state and update file tools for new workspace"""
        trial_id = super().reset_state()
        # Recreate file tools for new workspace
        self._setup_file_tools()
        return trial_id

    def get_task_prompt(self) -> str:
        prompt = f"{self.question} Make sure all the associated files for this task (input files, log files, any other files) are in the {self.work_dir} directory. You can use the available I/O tools (e.g., write_file) make new files .etc. Whatever potentials you need to run the simulation, you can find them at /potentials/. A type of potential can be accessed by /potentials/TYPE where TYPE can be [EAM, TERSOFF, REAXFF] which further contains the exact potential files. Note that in case of reaxff potentials, pair style 'reax/c' has been renamed to 'reaxff' and always use NULL for the control file (cfile), for example, this syntax is correct : pair_style reaxff NULL. If the task is to give the final output as a scalar, only return the numerical value, without any units."
        logger.info(f"prompt : {prompt}")
        return prompt

    def score(self) -> float:
        """Score based on reading the answer from the designated answer file."""

        if self.state.submitted_answer is None:
            logger.warning(f"No submission found for task {self.task_id}")
            return 0.0
        try:
            answer_value = self.state.submitted_answer.strip()
            if self.scoring_fn == "check_numerical":
                return check_numerical(answer_value, self.output[0])
            if self.scoring_fn == "energy_minimisation":
                return energy_minimisation(
                    self.work_dir, self.state.submitted_answer, self.output
                )
            elif self.scoring_fn == "check_numerical":
                return check_numerical(self.state.submitted_answer, self.output[0])
            elif self.scoring_fn == "check_structure":
                return check_structure(self.state.submitted_answer, self.output[0])
            elif self.scoring_fn == "check_stress_strain":
                return check_stress_strain(self.state.submitted_answer, self.output[0])
            elif self.scoring_fn == "check_stress_tensor":
                return check_stress_tensor(self.state.submitted_answer, self.output[0])
        except ValueError:
            return 0.0
        return 0.0


if __name__ == "__main__":
    tasks_files_path = Path(__file__).parent / "md_tutorials_tasks"
    tasks_files = tasks_files_path.glob("**/*.json")

    environments = {}
    for task_file in tasks_files:
        try:
            with Path.open(task_file, encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {task_file}: {e}")
            raise
        task_id = data["id"]
        question = data["input"][0]["prompt"]
        output = data["output"]
        scoring_fn = data["scoring_fn"]

        work_dir = os.environ.get("MODAL_BASE_IO_PATH", f"{task_id}/")

        environments[task_id] = LammpsEnvironment(
            task_id=task_id,
            question=question,
            output=output,
            work_dir=work_dir,
            scoring_fn=scoring_fn,
        )
    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    port = int(os.environ.get("CORRAL_PORT", "8000"))
    run_server(environments, host, port)
