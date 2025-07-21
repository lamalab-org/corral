import gc

from loguru import logger
from score import check_image_quality, check_params, check_scalar

# from tools import Image_optimizer, get_structure_from_mp_text
from tools import Code_Executor, Document_Retrieval, Image_Analyzer, Image_optimizer

from corral.base import Environment
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)


class AfmEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        question: str,
        answer: dict[str, float],
        initial_params: dict[str, float],
        scoring_fn: str,
        work_dir: str,
        file: str | None = None,
        pointer: str | None = None,
        gt: float | None = None,
        target_directory: str | None = None,
    ):
        self.question = question
        self.correct_answer = answer
        self.initial_params = initial_params
        self.scoring_fn = scoring_fn
        self.file = file
        self.pointer = pointer
        self.gt = gt
        self.target_directory = target_directory
        # self.target_directory = None

        super().__init__(task_id, base_work_dir=work_dir)

        # Add math-related tools
        self.add_tool(Image_optimizer)
        self.add_tool(Code_Executor)
        self.add_tool(Image_Analyzer)
        self.add_tool(Document_Retrieval)

        self._setup_file_tools()

    def _setup_file_tools(self):
        """Setup file tools for current workspace"""
        if self.current_work_dir:
            logger.info(
                f"DEBUG: Setting up FSManager with base_path: {self.current_work_dir}"
            )
            # Create new FSManager for current workspace
            fs_manager = FSManager("file", base_path=self.current_work_dir)

            # Add/update file tools
            self.tools.update(
                {
                    "list_files": ListFilesTool(fs_manager),
                    "read_file": ReadFileTool(fs_manager),
                    "write_file": WriteFileTool(fs_manager),
                    "file_info": FileInfoTool(fs_manager),
                    "cat_files": CatFilesTool(fs_manager),
                    "copy_file": CopyFileTool(fs_manager),
                }
            )
            logger.info(
                f"DEBUG: File tools setup complete for workspace: {self.current_work_dir}"
            )
        else:
            logger.warning("DEBUG: No current_work_dir set, skipping file tools setup")

    def update_pointer(self) -> None:
        logger.info("updating AFM file pointer")
        pointer_int = int(self.pointer) + 1
        self.pointer = str(pointer_int).zfill(len(self.pointer))
        logger.info(f"current pointer {self.pointer}")

    def reset_params(self) -> None:
        import pythoncom

        pythoncom.CoInitialize()

        import nanosurf

        # Initialize SPM and access subsystems
        # spm = nanosurf.SPM()
        # application = spm.application
        # del spm
        spm = nanosurf.SPM()
        application = spm.application
        scan = application.Scan
        zcontrol = application.ZController
        head = application.ScanHead

        # Access initial parameters
        params = self.initial_params

        # Apply scan parameters (converted to meters and seconds)
        scan.ImageHeight = params["image_height"] * 1e-9  # nm to m
        scan.ImageWidth = params["image_width"] * 1e-9  # nm to m
        scan.Scantime = params["times_per_line"]  # [s]
        scan.Points = params["points_per_line"]
        scan.Rotation = params["rotation"]  # [deg]
        scan.Lines = params["lines_per_frame"]

        # Apply Z-controller parameters
        zcontrol.PGain = params["pgain"]
        zcontrol.IGain = params["igain"]
        zcontrol.DGain = params["dgain"]
        zcontrol.SetPoint = params["setpoint"]
        head.CantileverByGUID = params["tip"]

        del zcontrol
        del scan
        del application
        del spm
        gc.collect()
        pythoncom.CoUninitialize()

        # Optionally log or confirm
        logger.info("AFM parameters have been reset to initial values.")

    def reset_state(self) -> str:
        """Reset state and update file tools for new workspace"""
        trial_id = super().reset_state()
        # Recreate file tools for new workspace
        self._setup_file_tools()
        self.reset_params()
        return trial_id

    def get_task_prompt(self) -> str:
        prompt = f"You are an advanced AI-AFM system with access to the Nanosurf AFM software through its Python API. Solve this task :\n{self.question}\n Images, if generated in the experiment, are stored at {self.target_directory}. If any images are to be saved, use the name {self.file}_ for the nid file in which image will be saved. Note that the Nanosurf software automatically appends an INDEX in the given file name, and hence the final image that will be saved will be {self.file}_{self.pointer}.nid file. In case of numerical answers, give the final scalar output without any units. Do not change parameters except the ones given in the task, as they have been configured by the experimentalists."
        # Add workspace info
        if self.current_work_dir:
            prompt += "\nIMPORTANT: You have access to filesystem tools. All files, if generated, except AFM images will be saved in your isolated workspace."
            # prompt += f"For this task, the afm image will be saved at C:\\Users\\Admin\\Desktop\\corral\\mat-agent-bench\\tasks\\afm\\afm\\afm_images\\{self.file}_{pointer_}.nid\n"
        logger.info(f"prompt : {prompt}")
        return prompt

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            logger.warning(f"No submission found for task {self.task_id}")
            return 0.0

        try:
            # Get and log the raw submission
            answer_value = self.state.submitted_answer.strip()
            logger.info(f"Raw submission for {self.task_id}: {answer_value!r}")

            if self.scoring_fn == "check_params":
                return check_params(self.correct_answer)

            if self.scoring_fn == "check_image":
                score = check_params(self.correct_answer)
                from pathlib import Path

                base_path = Path(self.target_directory)
                file_name = f"{self.file}_{self.pointer}.nid"
                full_path = base_path / file_name
                logger.info(f"checking for path {full_path}")
                exists = full_path.exists()
                logger.info(f"Path exists : {exists}")
                if exists:
                    self.update_pointer()
                return 1.0 if score > 0 and exists else 0.0

            elif self.scoring_fn == "check_scalar":
                score1 = check_scalar(float(self.gt), float(answer_value))
                score2 = check_params(self.correct_answer)
                score = 1.0 if score1 > 0 and score2 > 0 else 0.0

                from pathlib import Path

                base_path = Path(self.target_directory)
                file_name = f"{self.file}_{self.pointer}.nid"
                full_path = base_path / file_name
                logger.info(f"checking for path {full_path}")
                exists = full_path.exists()
                logger.info(f"Path exists : {exists}")
                if exists:
                    self.update_pointer()
                return score

            elif self.scoring_fn == "check_image_quality":
                from pathlib import Path

                base_path = Path(self.target_directory)
                files = list(base_path.glob("*"))
                if not files:
                    return 0.0
                latest_file = max(files, key=lambda f: f.stat().st_ctime)
                return check_image_quality(str(latest_file))

            # elif self.scoring_fn == "binary_image_score":
            #     base_path = self.target_directory
            #     file_name = f"{self.file}_{self.pointer}.nid"
            #     full_path = os.path.join(base_path, file_name)
            #     score = binary_image_score(img1_path, img2_path, metric='ssim', threshold=0.95, resize_to=None)
            #     return score

        except Exception as e:
            logger.error(
                f"Error scoring submission for task {self.task_id}: {e!s}",
                exc_info=True,
            )
            logger.error(f"Submission was: {self.state.submitted_answer!r}")
            return 0.0


# if __name__ == "__main__":

# # Create environments for different tasks
# environments = {
#     "afm_1": AfmEnvironment(
#         "afm_1",
#         "Set the image size to 100x100 nanometer.",
#         [100, 100],
#         "check_image_size",
#         work_dir="./test_directory",
#     ),

# }

# host = os.environ.get("CORRAL_HOST", "0.0.0.0")
# port = int(os.environ.get("CORRAL_PORT", "8000"))
# run_server(environments, host, port)
