import uvicorn
from loguru import logger
# from tools import Image_optimizer, get_structure_from_mp_text
from tools import Image_optimizer, Image_Analyzer, Code_Executor, Document_Retrieval
from pathlib import Path
from score import *
from corral.base import Environment, Tool
from corral.io import (
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    MkdirTool,
    CatFilesTool
)
import os

from corral.server import run_server

import glob

from score import check_params, check_scalar, check_image_quality

class AfmEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        question: str,
        final_params: dict[str, float],
        initial_params : dict[str, float],
        scoring_fn: str,
        work_dir: str,
        gt: float = None
    ):
        self.question = question
        self.final_params = final_params
        self.initial_params = initial_params
        self.scoring_fn = scoring_fn
        self.work_dir = work_dir
        self.gt = gt

        super().__init__(task_id, base_work_dir=work_dir)

        # Add math-related tools
        self.add_tool(Image_optimizer)
        self.add_tool(Code_Executor)
        self.add_tool(Image_Analyzer)
        self.add_tool(Document_Retrieval)

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
        application.SetGalleryHistoryDirectoryPath(self.current_work_dir)
        scan = application.Scan
        zcontrol = application.ZController
        head = application.ScanHead

        # Access initial parameters
        params = self.initial_params

        # Apply scan parameters (converted to meters and seconds)
        scan.ImageHeight = params["image_height"] * 1e-9  # nm to m
        scan.ImageWidth = params["image_width"] * 1e-9    # nm to m
        scan.Scantime = params["times_per_line"]          # [s]
        scan.Points = params["points_per_line"]
        scan.Rotation = params["rotation"]                # [deg]
        scan.Lines = params["lines_per_frame"]
        scan.CenterPosX = params["centre_x"] * 1e-9
        scan.CenterPosY = params["centre_y"] * 1e-9
        zcontrol.PGain = params["pgain"]
        zcontrol.IGain = params["igain"]
        zcontrol.DGain = params["dgain"]
        zcontrol.SetPoint = params["setpoint"]
        head.CantileverByGUID = params["tip"]
    
        logger.info(f"AFM parameters have been reset to initial values. AFM images will be saved at {application.GetGalleryHistoryDirectoryPath}. Corral's current working directory is {self.current_work_dir}.")

        del zcontrol
        del scan
        del application
        del spm
        gc.collect()
        pythoncom.CoUninitialize()

    def reset_state(self) -> str:
        """Reset state and update file tools for new workspace"""
        trial_id = super().reset_state()
        # Recreate file tools for new workspace
        self._setup_file_tools()
        self.reset_params()
        return trial_id
    
    def get_task_prompt(self) -> str:
        prompt = f"You are an advanced AI-AFM system with access to the Nanosurf AFM software through its Python API. Solve this task : \n{self.question}\n In case of numerical answers, give the final scalar output without any units. Do not change parameters except the ones given in the task, as they have been configured by the experimentalists."
        # # Add workspace info
        if self.current_work_dir:
            prompt += f"\nIMPORTANT: You have access to filesystem tools. Any files, including AFM images, if generated, will be automatically saved in your isolated workspace. Your isolated workspace is located at {self.current_work_dir}."
        logger.info(f"prompt : {prompt}")
        return prompt
    
    def score(self) -> float:
        """Score the submitted answer"""
        from score import check_nid_file_exists
        from score import check_params, check_scalar, check_image_quality
        # if not self.state.submitted_answer:
        #     logger.warning(f"No submission found for task {self.task_id}")
        #     return 0.0
        
        try:
            # Get and log the raw submission
            answer_value = None
            if self.state.submitted_answer: 
                answer_value = self.state.submitted_answer.strip()
            logger.info(f"Raw submission for {self.task_id}: {answer_value!r}")

            if self.scoring_fn == "check_indendation":
                score = check_params(self.final_params)
                if check_indentation(self.gt, answer_value):
                    return 1.0
                else:
                    return 0.0

            if self.scoring_fn == "check_image":
                score = check_params(self.final_params)
                if check_nid_file_exists(self.current_work_dir):
                    return score
                else:
                    return 0.0
                
            if self.scoring_fn == "check_params":
                score = check_params(self.final_params)
                return score
            
            if self.scoring_fn == "check_scalar":
                score = check_params(self.final_params)
                if check_scalar(self.gt, answer_value):
                    return score
                else:
                    return 0.0
                
            if self.scoring_fn == "check_image_quality":
                score = check_params(self.final_params)
                logger.info(f"looking files in {self.current_work_dir}")
                nid_files = glob.glob(os.path.join(self.current_work_dir, "*.nid"))
                if not nid_files:
                    return 0.0
                latest_file = max(nid_files, key=os.path.getmtime)
                if check_image_quality(os.path.join(self.current_work_dir, latest_file)):
                    return 1.0
                else:
                    return 0.0


        except Exception as e:
            logger.error(
                f"Error scoring submission for task {self.task_id}: {e!s}",
                exc_info=True,
            )
            logger.error(f"Submission was: {self.state.submitted_answer!r}")
            return 0.0

    




