import base64
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from corral_md.modal_workspace import (
    run_lammps_in_modal,
    run_python_gpu_in_modal,
    run_verified_md_in_modal,
)
from corral_md.workspace import local_path

from corral.core.tool import Tool, tool
from corral.core.transition import ToolRecoveryPending
from corral.runtime.tool_execution import PreparedToolCall, execute_prepared_call
from corral.workspace import (
    AbsoluteWorkspaceFilesystem,
    build_terminal_tool,
    workspace_relative_path,
)


def build_run_verified_md_tool(workspace: str | Path):
    @tool(hidden_args=["corral_action_id"], trusted=True, workspace_access="read_write")
    def run_verified_md(config_file: str, corral_action_id: str | None = None) -> str:
        """[BRIEF] Run the verified aluminum heat-capacity cycle and save its simulation artifacts. [/BRIEF]

        [DETAILED] The backend runs a fixed-cell MACE-MP-0 Langevin simulation at 300, 400, 500, 600, 700, 800, 900, and 300 K. It saves the states and thermal traces needed to analyze the cycle. [/DETAILED]

        [PROCEDURAL] When to use this tool:
        - Use it for the aluminum heat-capacity task when independently verifiable execution is useful.
        - Analyze the saved artifacts after the run finishes.
        [/PROCEDURAL]

        [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Save the cycle settings in a JSON file. [/PREREQUISITE]
        2. [CURRENT] Run this tool with the configuration file. [/CURRENT]
        3. [FOLLOW_UP] Analyze the output and add the returned run and action IDs to the manifest. [/FOLLOW_UP]
        [/WORKFLOW_INTEGRATION]

        [CONTEXTUAL] How this tool works:
        - Reads the cycle configuration from the task workspace.
        - Initializes velocities once and runs the eight temperature stages in order.
        - Saves settings, trajectories, thermal traces, and a run receipt under /workspace/output.
        [/CONTEXTUAL]

        [SYNTACTICAL] Usage example:
        `run_verified_md("/workspace/input/verified_md.json")`
        [/SYNTACTICAL]

        Args:
            config_file: [ARGS_BRIEF] Absolute path to the cycle configuration JSON file. [/ARGS_BRIEF]
                [ARGS_DETAILED] The file must be under /workspace and contain random_seed. It may also set timestep_fs, friction_fs, default_dtype, and per-stage step and sampling values. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Absolute /workspace path to a JSON file. [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] "/workspace/input/verified_md.json" [/ARGS_EXAMPLES]

        Returns:
            str: [RETURNS_BRIEF] JSON receipt for the completed run. [/RETURNS_BRIEF]
                [RETURNS_DETAILED] The receipt includes the run ID, action ID, release ID, and output directory. [/RETURNS_DETAILED]
                [RETURNS_EXAMPLES] `{"run_id": "...", "action_id": "...", "output_directory": "/workspace/output"}` [/RETURNS_EXAMPLES]

        [RAISES] Exceptions:
            FileNotFoundError:
                [ERROR_WHEN] The configuration file does not exist. [/ERROR_WHEN]
                [ERROR_DETAILS] The supplied path does not name a workspace file. [/ERROR_DETAILS]
                [ERROR_RECOVERY] Check the path and create the configuration file first. [/ERROR_RECOVERY]
            ValueError:
                [ERROR_WHEN] The configuration is invalid. [/ERROR_WHEN]
                [ERROR_DETAILS] A required value is missing or a setting is outside the supported range. [/ERROR_DETAILS]
                [ERROR_RECOVERY] Correct the configuration and run the tool again. [/ERROR_RECOVERY]
        [/RAISES]

        [LIMITATIONS] Known limitations:
        - This tool runs only the fixed aluminum heat-capacity workflow.
        - It does not run submitted code or submitted models.
        [/LIMITATIONS]
        """
        config = local_path(workspace, config_file)
        run_verified_md_in_modal(workspace, str(config), action_id=corral_action_id)
        return local_path(
            workspace, "/workspace/output/verified_md_receipt.json"
        ).read_text()

    return run_verified_md


def build_md_terminal_tool(workspace: str | Path):
    """Run shell commands in Corral's local restricted workspace worker."""
    terminal = build_terminal_tool(AbsoluteWorkspaceFilesystem(workspace))
    terminal.description += (
        " Commands execute locally with the task workspace as the current "
        "directory; use relative paths such as output/result.json inside commands."
    )
    return terminal


@tool(workspace_access="read_write")
def get_nth_run_log(
    path: str,
    n: int = 0,
    save: str | None = None,
    index: int | None = None,
) -> str:
    """[BRIEF] Retrieves and processes the nth run log from a LAMMPS log file, where a “run” refers to a contiguous simulation segment such as an energy minimization or an ensemble run (e.g., NVT, NPT, NVE). It returns the list of thermodynamic variables present in that run and the total number of steps. Optionally, it can save the nth run to a CSV file where each column corresponds to a thermodynamic property, and optionally it can return the value of a specific thermodynamic property at a given index. [/BRIEF]

    [DETAILED] This tool extracts the nth run log from a LAMMPS log file, providing access to the thermodynamic properties recorded during a specific simulation segment. In this context, a “run” refers to a contiguous block of simulation output corresponding to an energy minimization or an ensemble-based simulation stage (e.g., NVT, NPT, NVE) as produced by LAMMPS. LAMMPS log files may contain multiple such runs within a single file, for example when a script performs a minimization followed by one or more ensemble simulations. This tool allows selecting one of these runs by index, extracting its thermodynamic data, and optionally saving it to a CSV file for further analysis. It can also return the value of a specific thermodynamic quantity at a specified step index within the selected run. This is useful for automated analysis pipelines, post-processing workflows, or programmatic inspection of simulation results without manually parsing the log file. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to extract and analyze the nth run log from a LAMMPS log file without reading the entire log file.
    - Best suited for post-processing and analyzing simulation outputs using external tools or data analysis pipelines.
    - Recommended for automating the extraction of thermodynamic data from LAMMPS simulations.
    [/PROCEDURAL]
    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a valid LAMMPS log file generated from a simulation. [/PREREQUISITE]
    2. [CURRENT] Use this tool to extract the nth run log and optionally save it to a CSV file. [/CURRENT]
    3. [FOLLOW_UP] Utilize the extracted data for analysis, visualization, or further processing in your materials science workflow. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]
    [CONTEXTUAL] How this tool works:
    - Reads the LAMMPS log file and identifies the nth run section.
    - Extracts thermodynamic data and organises it into a python dataframe, where each column corresponds to a thermodynamic property.
    - Optionally saves the data to a CSV file and retrieves specific data at a given index. [/CONTEXTUAL]
    [SYNTACTICAL] Usage examples:
    [
        `get_nth_run_log("/workspace/output/log.lammps", 0, "/workspace/output/run0_thermo.csv")`,
        `get_nth_run_log("/workspace/output/log.lammps", 1, None)`,
        `get_nth_run_log("/workspace/output/log.lammps", 2, "/workspace/output/run2_thermo.csv", 10)`,
        `get_nth_run_log("/workspace/output/sim_log.lammps", 0, "/workspace/output/run0_thermo.csv", 5)`,
        `get_nth_run_log("/workspace/log.lammps", 3, None, 20)`,
    ]
    [/SYNTACTICAL]
    Args:
        path: [ARGS_BRIEF] Absolute path to the LAMMPS log file. [/ARGS_BRIEF]
              [ARGS_DETAILED] Complete file path to the LAMMPS log file containing simulation output. [/ARGS_DETAILED]
              [ARGS_SYNTACTICAL] Format: "Valid file path to LAMMPS log file" [/ARGS_SYNTACTICAL]
              [ARGS_EXAMPLES] Examples: "/workspace/output/log.lammps", "/workspace/output/log.lammps" [/ARGS_EXAMPLES]
        n: [ARGS_BRIEF] Index of the run log to extract (0-based). Defaults to 0. [/ARGS_BRIEF]
              [ARGS_DETAILED] The zero-based index of the run log to extract from the LAMMPS log file. [/ARGS_DETAILED]
              [ARGS_SYNTACTICAL] Format: "Non-negative integer" [/ARGS_SYNTACTICAL]
              [ARGS_EXAMPLES] Examples: 0, 1, 2 [/ARGS_EXAMPLES]
        save: [ARGS_BRIEF] Optional path to save the extracted run log as a CSV file. [/ARGS_BRIEF]
                [ARGS_DETAILED] If provided, the extracted run log will be saved to this path in CSV format. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Format: "Valid file path to save CSV file" [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] Examples: "/workspace/output/run0_thermo.csv", "/workspace/output/run1_thermo.csv" [/ARGS_EXAMPLES]
        index: [ARGS_BRIEF] Optional index to retrieve specific thermodynamic data from the run log. [/ARGS_BRIEF]
                 [ARGS_DETAILED] If provided, the tool will return the thermodynamic data at this index from the extracted run log. [/ARGS_DETAILED]
                 [ARGS_SYNTACTICAL] Format: "Non-negative integer" [/ARGS_SYNTACTICAL]
                 [ARGS_EXAMPLES] Examples: 0, 5, 10 [/ARGS_EXAMPLES]
    Returns:
        str: [RETURNS_BRIEF] Summary of the extracted run log and optional data at the specified index. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A string summarizing the columns present in the extracted run log, total number of rows, and optionally the thermodynamic data at the specified index. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] Examples: "the thermo data has been saved successfully at run0_thermo.csv. The thermo columns are: ['Step', 'Temp', 'Press']. There are total 1000 rows. Data at index 10: {'Step': 100, 'Temp': 300, 'Press': 1.0}", "The thermo columns are: ['Step', 'Temp', 'Press']. There are total 500 rows." [/RETURNS_EXAMPLES]
    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If there is an error reading the log file or extracting the run log. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the log file cannot be read or the specified run log cannot be extracted. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the log file path and ensure the run index is valid. [/ERROR_RECOVERY]
    [/RAISES]
    [LIMITATIONS] Known limitations:
    - Only supports LAMMPS log files with standard formatting.
    - May not handle corrupted or non-standard log files gracefully.
    [/LIMITATIONS]

    """

    import log_lammps_reader

    try:
        log_data = log_lammps_reader.parse(path, n)
        result = f"{log_data.head()}\n"
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            log_data.write_csv(save)
            result += f"Thermo data for run {n} saved to {save}.\n"
        if index is not None:
            if 0 <= index < log_data.height:
                result += f"Data at index {index}: {log_data.row(index)}\n"
            else:
                result += (
                    f"Index {index} is out of bounds for data with "
                    f"{log_data.height} rows.\n"
                )
        return result
    except Exception as e:
        return f"Failed to parse thermo data for run {n}: {e}"


@tool(workspace_access="read")
def keyword_log_extractor(path: str, keyword: str) -> str:
    """[BRIEF] Extracts sections of a LAMMPS log file that start with a specified keyword. [/BRIEF]
    [DETAILED] This tool scans a LAMMPS log file for sections that begin with a given keyword and extracts those sections for analysis. It is useful for retrieving specific information such as fixes, computes, or other logged data from simulation runs. [/DETAILED]
    [PROCEDURAL] When to use this tool:
    - Use when you need to extract specific sections of a LAMMPS log file based on keywords.
    - Best suited for targeted analysis of simulation outputs.
    - Recommended for retrieving logged data for further processing or visualization.
    [/PROCEDURAL]
    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a valid LAMMPS log file generated from a simulation. [/PREREQUISITE]
    2. [CURRENT] Use this tool to extract sections of the log file that start with the specified keyword. [/CURRENT]
    3. [FOLLOW_UP] Utilize the extracted data for analysis, visualization, or further processing in your materials science workflow. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]
    [CONTEXTUAL] How this tool works:
    - Reads the LAMMPS log file line by line.
    - Identifies sections that start with the specified keyword.
    - Extracts and returns those sections as a structured dictionary. [/CONTEXTUAL]
    [SYNTACTICAL] Usage examples:
    [
        `keyword_log_extractor("/workspace/output/log.lammps", "fix")`,
        `keyword_log_extractor("/workspace/output/log.lammps", "BULK ENERGY")`,
        `keyword_log_extractor("/workspace/output/log.lammps", "thermo")`,
        `keyword_log_extractor("/workspace/output/sim_log.lammps", "dump")`,
        `keyword_log_extractor("/workspace/log.lammps", "velocity")`,
    ]
    [/SYNTACTICAL]
    Args:
        path: [ARGS_BRIEF] Path to the LAMMPS log file. [/ARGS_BRIEF]
                [ARGS_DETAILED] Complete file path to the LAMMPS log file containing simulation output. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Format: "Valid file path to LAMMPS log file" [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] Examples: "/workspace/output/log.lammps", "/workspace/output/log.lammps" [/ARGS_EXAMPLES]
        keyword: [ARGS_BRIEF] Keyword to search for in the log file. [/ARGS_BRIEF]
                  [ARGS_DETAILED] The specific keyword that marks the beginning of sections to extract from the log file. [/ARGS_DETAILED]
                  [ARGS_SYNTACTICAL] Format: "Non-empty string" [/ARGS_SYNTACTICAL]
                  [ARGS_EXAMPLES] Examples: "fix", "compute", "thermo" [/ARGS_EXAMPLES]
    Returns:
        str: [RETURNS_BRIEF] Extracted sections as a structured dictionary in string format. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A string representation of a dictionary containing the extracted sections that start with the specified keyword. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] "{'fix': [...]}", "{'compute': [...]}" [/RETURNS_EXAMPLES]
    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If there is an error reading the log file or extracting sections. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the log file cannot be read or the keyword is not found. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the log file path and ensure the keyword is valid. [/ERROR_RECOVERY]
    [/RAISES]
    [LIMITATIONS] Known limitations:
    - Only supports LAMMPS log files with standard formatting.
    - May not handle corrupted or non-standard log files gracefully.
    [/LIMITATIONS]
    """
    import log_lammps_reader

    try:
        matches = log_lammps_reader.log_starts_with(path, keyword)
        if not matches:
            raise ValueError(f"Keyword {keyword!r} not found in log.")
        return json.dumps({keyword: matches}, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"Error processing keyword {keyword!r}: {e}"})


def _local_python_argument(workspace: str | Path, argument: str) -> str:
    if argument == "/workspace" or argument.startswith("/workspace/"):
        return str(local_path(workspace, argument, allow_root=argument == "/workspace"))
    return argument


def _run_python_locally(
    workspace: str | Path,
    script_path: str,
    args: list[str] | None,
    timeout: int,
    working_dir: str | None,
) -> str:
    """Execute CPU Python after Corral has entered the restricted tool worker."""
    root = Path(workspace).resolve()
    script = local_path(root, script_path)
    public_directory = working_dir if working_dir is not None else "/workspace"
    directory = local_path(root, public_directory, allow_root=True)
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Working directory not found: {public_directory}")
    if type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("timeout must be between 1 and 7200 seconds")

    output = local_path(root, "/workspace/output", allow_root=True)
    output.mkdir(exist_ok=True)
    command = [
        sys.executable,
        str(script),
        *[_local_python_argument(root, str(argument)) for argument in args or []],
    ]
    scratch = Path(tempfile.gettempdir()).resolve()
    environment = {
        "CORRAL_WORKSPACE": str(root),
        "HOME": str(root),
        "LANG": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(scratch),
        "XDG_CACHE_HOME": str(scratch / "cache"),
    }
    timed_out = False
    with (
        tempfile.TemporaryFile() as stdout_stream,
        tempfile.TemporaryFile() as stderr_stream,
    ):
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        stdout_stream.seek(0)
        stderr_stream.seek(0)
        stdout = stdout_stream.read()
        stderr = stderr_stream.read()

    stdout_path = local_path(root, f"/workspace/output/{script.stem}.stdout.txt")
    stderr_path = local_path(root, f"/workspace/output/{script.stem}.stderr.txt")
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if timed_out:
        raise TimeoutError(f"Python script exceeded its {timeout}-second timeout")
    if process.returncode:
        details = stderr.decode("utf-8", errors="replace")[-4000:]
        raise ValueError(
            f"Python script failed with exit code {process.returncode}:\n{details}"
        )
    return json.dumps(
        {
            "success": True,
            "backend": "local_cpu",
            "return_code": process.returncode,
            "stdout": f"/workspace/output/{script.stem}.stdout.txt",
            "stderr": f"/workspace/output/{script.stem}.stderr.txt",
        }
    )


class _LocalPythonScriptTool(Tool):
    """Internal worker-only tool; it is never included in the agent catalog."""

    def __init__(self) -> None:
        super().__init__(
            name="_corral_md_local_python",
            description="Run one Python script in the assigned workspace.",
            workspace_access="read_write",
            hidden_args={"workspace": {"type": "string"}},
            workspace_args=("workspace",),
            params_json_schema={
                "type": "object",
                "properties": {
                    "script_relative": {"type": "string"},
                    "args": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ]
                    },
                    "timeout": {"type": "integer"},
                    "directory_relative": {"type": "string"},
                },
                "required": [
                    "script_relative",
                    "args",
                    "timeout",
                    "directory_relative",
                ],
            },
        )

    def execute(
        self,
        *,
        workspace: str,
        script_relative: str,
        args: list[str] | None,
        timeout: int,
        directory_relative: str,
    ) -> str:
        return _run_python_locally(
            workspace,
            f"/workspace/{script_relative}",
            args,
            timeout,
            "/workspace"
            if directory_relative == "."
            else f"/workspace/{directory_relative}",
        )


def _run_python_in_local_worker(
    workspace: str | Path,
    script_path: str,
    args: list[str] | None,
    timeout: int,
    working_dir: str | None,
) -> str:
    arguments = {
        "workspace": str(Path(workspace).resolve()),
        "script_relative": workspace_relative_path(script_path),
        "args": args,
        "timeout": timeout,
        "directory_relative": workspace_relative_path(
            working_dir if working_dir is not None else "/workspace", allow_root=True
        ),
    }
    worker_tool = _LocalPythonScriptTool()
    prepared = PreparedToolCall.capture(
        worker_tool, arguments, workspace=str(Path(workspace).resolve())
    )
    return execute_prepared_call(prepared)


def build_execute_python_script_tool(workspace: str | Path):
    """Build one Python interface with local CPU and Modal GPU routing."""

    @tool(
        hidden_args=["corral_action_id"],
        controller_dispatch=True,
        workspace_access="read_write",
    )
    def execute_python_script(
        script_path: str,
        args: list[str] | None = None,
        timeout: int = 600,
        working_dir: str | None = None,
        use_gpu: bool = False,
        corral_action_id: str | None = None,
    ) -> str:
        """[BRIEF] Run a saved Python script with the appropriate CPU or GPU backend. [/BRIEF]

        [DETAILED] CPU scripts run in Corral's restricted local task workspace. When use_gpu is true, the same interface synchronizes the script and task files to an env with an A100 GPU. [/DETAILED]

        [PROCEDURAL] When to use this tool:
        - Leave use_gpu false for lightweight analysis, plotting, and conversion.
        - Set use_gpu true for MACE inference, training, or molecular dynamics.
        [/PROCEDURAL]

        [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Save the script and inputs under /workspace. [/PREREQUISITE]
        2. [CURRENT] Run the script with its required compute resource. [/CURRENT]
        3. [FOLLOW_UP] Inspect synchronized outputs and captured logs. [/FOLLOW_UP]
        [/WORKFLOW_INTEGRATION]

        [CONTEXTUAL] How this tool works:
        - Validates the script and working-directory paths.
        - Runs CPU work locally and GPU work on an A100 in Modal.
        - Saves or synchronizes captured logs under /workspace/output.
        [/CONTEXTUAL]

        [SYNTACTICAL] Usage example:
        `execute_python_script("/workspace/scripts/analyze.py")`
        `execute_python_script("/workspace/scripts/mace_md.py", use_gpu=True, timeout=3600)`
        [/SYNTACTICAL]

        Args:
            script_path: [ARGS_BRIEF] Absolute path to the Python script. [/ARGS_BRIEF]
                [ARGS_DETAILED] The script must be a file under /workspace. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Absolute /workspace path to a Python file. [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] "/workspace/scripts/mace_md.py" [/ARGS_EXAMPLES]
            args: [ARGS_BRIEF] Optional command-line arguments for the script. [/ARGS_BRIEF]
                [ARGS_DETAILED] The strings are passed to the script in order. Absolute /workspace arguments are translated for local CPU execution. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] List of strings or null. [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] ["--steps", "1000"], null [/ARGS_EXAMPLES]
            timeout: [ARGS_BRIEF] Maximum run time in seconds. Defaults to 600. [/ARGS_BRIEF]
                [ARGS_DETAILED] The value must be between 1 and 7200. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Integer from 1 to 7200. [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] 600, 3600 [/ARGS_EXAMPLES]
            working_dir: [ARGS_BRIEF] Working directory. Defaults to /workspace. [/ARGS_BRIEF]
                [ARGS_DETAILED] The directory must be under /workspace. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Absolute /workspace directory path or null. [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] "/workspace", "/workspace/output" [/ARGS_EXAMPLES]
            use_gpu: [ARGS_BRIEF] Whether the script requires a GPU. Defaults to false. [/ARGS_BRIEF]
                [ARGS_DETAILED] False runs locally on CPU; true runs on an A100 in Modal. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Boolean. [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] false, true [/ARGS_EXAMPLES]

        Returns:
            str: [RETURNS_BRIEF] JSON confirming successful execution. [/RETURNS_BRIEF]
                [RETURNS_DETAILED] The result reports the selected backend and captured-log locations. [/RETURNS_DETAILED]
                [RETURNS_EXAMPLES] `{"success": true, "backend": "local_cpu", "return_code": 0}` [/RETURNS_EXAMPLES]

        [RAISES] Exceptions:
            FileNotFoundError:
                [ERROR_WHEN] The script does not exist. [/ERROR_WHEN]
                [ERROR_DETAILS] The supplied path does not name a workspace file. [/ERROR_DETAILS]
                [ERROR_RECOVERY] Check the path and save the script first. [/ERROR_RECOVERY]
            NotADirectoryError:
                [ERROR_WHEN] The working directory does not exist. [/ERROR_WHEN]
                [ERROR_DETAILS] The path does not name a workspace directory. [/ERROR_DETAILS]
                [ERROR_RECOVERY] Use an existing workspace directory. [/ERROR_RECOVERY]
            ValueError:
                [ERROR_WHEN] The timeout or a workspace path is invalid. [/ERROR_WHEN]
                [ERROR_DETAILS] A value is outside its accepted range or path boundary. [/ERROR_DETAILS]
                [ERROR_RECOVERY] Correct the value and run the tool again. [/ERROR_RECOVERY]
        [/RAISES]

        [LIMITATIONS] Known limitations:
            - Interactive input and live output streaming are not supported.
        [/LIMITATIONS]
        """
        script = local_path(workspace, script_path)
        directory = working_dir if working_dir is not None else "/workspace"
        local_directory = local_path(workspace, directory, allow_root=True)
        if not script.is_file():
            raise FileNotFoundError(f"Script not found: {script_path}")
        if not local_directory.is_dir():
            raise NotADirectoryError(f"Working directory not found: {directory}")
        if type(timeout) is not int or not 1 <= timeout <= 7200:
            raise ValueError("timeout must be between 1 and 7200 seconds")
        if not use_gpu:
            return _run_python_in_local_worker(
                workspace, script_path, args, timeout, directory
            )
        downloaded = run_python_gpu_in_modal(
            workspace,
            str(script),
            args or [],
            action_id=corral_action_id,
            timeout=timeout,
            working_dir=directory,
        )
        return json.dumps(
            {
                "success": True,
                "backend": "modal_a100",
                "return_code": 0,
                "stdout": f"Script completed on an A100. Synchronized {downloaded} workspace file(s).",
                "stderr": "Captured logs are in /workspace/output/.",
            }
        )

    def execute_controller(_environment, _state, prepared):
        # Dispatch may synchronize files or start a restricted CPU worker;
        # agent-authored Python is never evaluated by the controller.
        return execute_python_script.execute(**prepared.arguments)

    execute_python_script.execute_controller = execute_controller
    return execute_python_script


@tool
def get_potential_metadata(file_path: str) -> str:
    """
    [BRIEF] Returns metadata from a known LAMMPS potential file given the file path. The metadata includes potential type, elements supported, and the LAMMPS-compatible pair style keyword. Raises an exception if the path is invalid or the potential file is not recognized. [/BRIEF]

    [DETAILED] This tool provides a quick and reliable way to identify the type and supported elements of a LAMMPS potential file based on its path in the local catalog.
    It eliminates the need to parse the often large and complex contents of potential files, which can exceed processing limits in many systems. The tool validates the provided path against the known files mounted into the LAMMPS runtime. For recognized files, it returns a structured metadata string describing the potential type, supported elements, and the LAMMPS pair style.
    If the filename does not match any known potential, a ValueError is raised.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need to quickly determine the type, supported elements, or the LAMMPS pair style of a LAMMPS potential file based on its filename.
        - Use when you want to validate that a potential file path is accessible before using it in a simulation workflow.
        - Best suited for selecting an appropriate potential file for a specific molecular dynamics (MD) simulation without reading or parsing the full file contents.
        - Recommended for gaining a fast, structured overview of a potential file's applicability to specific element combinations or simulation scenarios.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Select or obtain the potential file paths that might be relevant to your simulation task. [/PREREQUISITE]
        2. [CURRENT] Use this tool to retrieve metadata for each potential file path. The tool will either return metadata or raise an exception if the path or filename is invalid. [/CURRENT]
        3. [FOLLOW_UP] Based on the metadata returned, choose the appropriate potential file for your simulation setup, and run the simulation using the selected potential file and the tool `run_lammps`. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Validates that the input file path is non-empty
        - Validates remote mount paths against the local potential catalog
        - Extracts the filename from the provided path using pathlib
        - Matches it against a set of known potential filenames
        - Returns a structured metadata string for recognized files
        - Raises FileNotFoundError for invalid or inaccessible paths
        - Raises ValueError for unrecognized potential filenames
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_potential_metadata("/workspace/potentials/EAM/Al99.eam.alloy")`,
        `get_potential_metadata("/workspace/potentials/EAM/Mg_Zhou04.eam.alloy")`,
        `get_potential_metadata("/workspace/potentials/EAM/Fe-C_Hepburn_Ackland.eam.fs")`,
        `get_potential_metadata("/workspace/potentials/SW/Si.sw")`
    ]
    [/SYNTACTICAL]

    Args:
        file_path:
            [ARGS_BRIEF] Path to the potential file. [/ARGS_BRIEF]
            [ARGS_DETAILED] This is the path to a LAMMPS-compatible potential file. The file must be accessible, and its filename is used to determine metadata, so it must match one of the known potential filenames. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: "string ending in a recognized potential filename". [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "/workspace/potentials/EAM/Al99.eam.alloy", "/workspace/potentials/EAM/Mg_Zhou04.eam.alloy" [/ARGS_EXAMPLES]

    Returns:
        str :
            [RETURNS_BRIEF] Structured metadata string describing the potential file. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] The returned string includes the type of interatomic potential, supported chemical elements, and the LAMMPS pair style. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] Example output: "{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the file path is empty or the potential filename is not recognized. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the input path is empty or when the filename does not match any known potential files. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the file path is non-empty and that the filename matches one of the supported potential files. [/ERROR_RECOVERY]

        FileNotFoundError:
            [ERROR_WHEN] If the file path is invalid or the file is not accessible. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the path is neither a local file nor an entry in the mounted potential catalog. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify that the file exists at the given path and that it is accessible in the execution environment. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Limitations:
        - This tool only recognizes a predefined set of potential filenames. If the filename does not match any known entry, a ValueError will be raised.
        - The metadata returned is static and does not include dynamic information from the file contents, such as specific parameters or coefficients used in the potential.
        - The tool does not validate the actual contents or physical correctness of the potential file; it relies on file accessibility and filename-based identification only.
    [/LIMITATIONS]
    """

    POTENTIALS = {
        "Si.sw": "{potential type : Stillinger Weber (SW), elements supported : Si (Silicon), pair_style : sw}",
        "2007_SiO.tersoff": "{potential type : tersoff, elements supported : Si (Silicon), Oxygen (O), pair_style : tersoff}",
        "Al99.eam.alloy": "{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}",
        "Cu_Zhou04.eam.alloy": "{potential type : EAM, elements supported : Cu (Copper), pair_style : eam/alloy}",
        "Mg_Zhou04.eam.alloy": "{potential type : EAM, elements supported : Mg (Magnesium), pair_style : eam/alloy}",
        "Fe-C_Hepburn_Ackland.eam.fs": "{potential type : EAM, elements supported : Fe (Iron), C (Carbon), pair_style : eam/fs}",
        "pot.mod": (
            "{potential type : Buckingham + Coulomb (BKS-type), elements supported : "
            "Na (Sodium), Si (Silicon), O (Oxygen), "
            "pair_style : hybrid/overlay buck/coul/long + kspace_style pppm}"
        ),
    }
    POTENTIAL_PATHS = {
        "Si.sw": "/workspace/potentials/SW/Si.sw",
        "2007_SiO.tersoff": "/workspace/potentials/TERSOFF/2007_SiO.tersoff",
        "Al99.eam.alloy": "/workspace/potentials/EAM/Al99.eam.alloy",
        "Cu_Zhou04.eam.alloy": "/workspace/potentials/EAM/Cu_Zhou04.eam.alloy",
        "Mg_Zhou04.eam.alloy": "/workspace/potentials/EAM/Mg_Zhou04.eam.alloy",
        "Fe-C_Hepburn_Ackland.eam.fs": (
            "/workspace/potentials/EAM/Fe-C_Hepburn_Ackland.eam.fs"
        ),
        "pot.mod": "/workspace/potentials/BKS/pot.mod",
    }

    workspace_relative_path(file_path)
    potential_path = Path(file_path)
    potential_name = potential_path.name
    metadata = POTENTIALS.get(potential_name)
    if metadata is None:
        raise ValueError(f"Unrecognized potential file: {potential_name}")

    expected_remote_path = POTENTIAL_PATHS[potential_name]
    if potential_path.as_posix() != expected_remote_path:
        raise FileNotFoundError(f"Incorrect potential file path: {file_path}")

    return metadata


@tool(trusted=True, workspace_access="read_write")
def get_structure_from_mp_text(mp_id: str, file_path: str) -> str:
    """
    [BRIEF] Retrieves and saves the conventional crystal structure of a material from the Materials Project as a CIF file. [/BRIEF]

    [DETAILED] This tool retrieves the conventional unit cell structure of a material from the Materials Project using its material ID. It transforms the primitive structure returned by the database into its conventional crystallographic form using symmetry operations, and exports the result in CIF format to a specified file path. This tool is valuable for workflows that require standardized crystal structure representations — such as simulations, visualization, structure matching, or publication. It avoids the need for manual structure transformation or dealing with primitive cells when conventional representation is needed. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need the conventional crystallographic (not primitive) structure of a material from the Materials Project.
        - Best suited for preparing simulation-ready input files, visualizing crystal structures, or storing standardized CIFs.
        - Recommended for quick and automated generation of conventional structure files for structure-based computation or crystallographic analysis.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Ensure you have the Materials Project ID of the material you want to retrieve. Sometimes the MP ID is in the task description so be sure to fully capture the information there. [/PREREQUISITE]
        2. [CURRENT] Use this tool to fetch the conventional structure in CIF format by providing the MP ID and desired file path. [/CURRENT]
        3. [FOLLOW_UP] The resulting CIF file can be used in subsequent steps such as molecular dynamics simulations (using the tool `run_lammps`), structure visualization, or crystallographic analysis. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Queries the Materials Project database using the provided material ID to retrieve the material's primitive crystal structure.
        - Applies symmetry analysis to convert the primitive structure into its conventional crystallographic form.
        - Converts the conventional structure into CIF (Crystallographic Information File) format.
        - Saves the CIF content to the specified file path using an external function.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_structure_from_mp_text("mp-149", "/workspace/Si_conventional.cif")`,
        `get_structure_from_mp_text("mp-13", "/workspace/output/Aluminum_structure.cif")`,
        `get_structure_from_mp_text("mp-1692", "/workspace/input/CuO_conventional.cif")`,
        `get_structure_from_mp_text("mp-19770", "/workspace/input/Fe2O3.cif")`,
        `get_structure_from_mp_text("mp-1143", "/workspace/input/Al2O3_structure.cif")`,
    ]
    [/SYNTACTICAL]

    Args:
        mp_id (str):
            [ARGS_BRIEF] Materials Project ID of the material. [/ARGS_BRIEF]
            [ARGS_DETAILED] A unique identifier used by the Materials Project database to reference a material. The ID typically starts with "mp-" followed by digits. It must correspond to an existing entry. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: '"mp-XXXX" where X is a digit'. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "mp-149", "mp-13", "mp-1234567" [/ARGS_EXAMPLES]
        file_path (str):
            [ARGS_BRIEF] Destination path for saving the CIF file. [/ARGS_BRIEF]
            [ARGS_DETAILED] Absolute path to the file where the CIF content will be written. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: 'string path ending in ".cif" corresponding to the path of the CIF file'. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "/workspace/input/output.cif", "/workspace/input/Al.cif" [/ARGS_EXAMPLES]

    Returns:
        str :
            [RETURNS_BRIEF] Status message indicating successful structure retrieval and saving at required path. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] If successful, the tool retrieves the conventional crystallographic structure for the given Materials Project ID, converts it into CIF format, saves it at the specified path, and returns a confirmation message.
            If any step fails, a descriptive error message is returned instead. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES]
                - "Structure saved successfully at /workspace/data/structure.cif"
                - "Failed to retrieve or save structure: Invalid Materials Project ID" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised if structure retrieval or file saving fails. [/ERROR_WHEN]
            [ERROR_DETAILS] This generic exception is returned if any error occurs during Materials Project API access, structure conversion, or remote file write. The error message is descriptive and includes the failure reason. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, check the MP ID for correctness, and verify that the specified file path is writable. If the MP ID is invalid or does not exist, you may need to use a different ID or check the Materials Project database for available materials. If the file path is incorrect or inaccessible, ensure that the directory exists and has the correct permissions for writing files. If the API key is invalid or the Materials Project service is down, you may need to use another tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
        - The tool requires a Materials Project API key in the controller's MP_API_KEY environment variable.
        - It assumes that the MP ID provided corresponds to a valid material in the Materials Project database.
        - The tool does not handle cases where the material has multiple structures or polymorphs; it retrieves only the first available structure.
    [/LIMITATIONS]
    """

    try:
        from mp_api.client import MPRester
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        with MPRester(os.getenv("MP_API_KEY")) as mpr:
            docs = mpr.materials.summary.search(
                material_ids=[str(mp_id)], fields=["structure"]
            )
            structure = docs[0].structure

        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()
        structure_cif = structure.to(fmt="cif")

        destination = Path(file_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(structure_cif, encoding="utf-8")

        return f"Structure saved successfully at {file_path}"

    except Exception as e:
        return f"Failed to retrieve or save structure: {e!s}"


@tool(workspace_access="read_write")
def convert_structure_to_lammps_data(
    structure_path: str, output_file: str, atom_style: str = "charge"
) -> str:
    """
    [BRIEF] Converts a CIF-format structure file into a LAMMPS data file using a specified atom style. [/BRIEF]

    [DETAILED] This tool converts a crystal structure provided in CIF format (as a file) into a LAMMPS-compatible data file. Internally, it reads the CIF structure file using pymatgen, transforms it into a Structure object, and then serializes it to a LAMMPS data format using the LammpsData class. It supports configurable atom styles such as "atomic" or "charge", allowing flexibility based on simulation requirements.
    This enables seamless transformation of standardized crystallographic data into simulation-ready LAMMPS input files, streamlining the setup process for molecular dynamics workflows. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need to convert crystallographic structure data in CIF format into a LAMMPS-compatible `.data` file.
        - Best suited for setting up molecular dynamics simulations where LAMMPS is the engine, and structure data is sourced from databases like Materials Project or experimental CIFs.
        - Avoid when your structure data is already in LAMMPS format or requires extensive pre-processing (e.g., force field assignments).
        - Recommended for automating simulation pipelines that begin with standardized structural data and end in LAMMPS-ready input formats.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Ensure you have a valid CIF file containing the crystallographic structure of the material you want to simulate. Use the tool `get_structure_from_mp_text` to obtain one. [/PREREQUISITE]
        2. [CURRENT] Use this tool to convert the CIF file into a LAMMPS data file by providing the path to the CIF file, the desired output file path, and the atom style (if different from the default "charge"). [/CURRENT]
        3. [FOLLOW_UP] The resulting LAMMPS data file can be used directly in LAMMPS simulations, using the tool `run_lammps` to run the simulation. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - The tool reads the `.cif` file using `pymatgen.core.Structure.from_file` to create a Structure object.
        - The `LammpsData.from_structure` function is called, using the chosen `atom_style` to format the atomic data accordingly.
        - The resulting LAMMPS data file is written to the path specified by `output_file`.
        - The tool does not apply force fields or assign charges explicitly; it assumes the structure is already complete and appropriate for the selected `atom_style`.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `convert_structure_to_lammps_data("/workspace/graphene.cif", "/workspace/output/graphene.data")`,
        `convert_structure_to_lammps_data("/workspace/input/NaCl.cif", "/workspace/output/NaCl.data", atom_style="atomic")`,
        `convert_structure_to_lammps_data("/workspace/input/SiO2.cif", "/workspace/input/SiO2.data", atom_style="charge")`,
        `convert_structure_to_lammps_data("/workspace/input/MgO.cif", "/workspace/input/MgO.data", atom_style="charge")`,
        `convert_structure_to_lammps_data("/workspace/input/Al2O3.cif", "/workspace/input/Al2O3.data", atom_style="atomic")`
    ]
    [/SYNTACTICAL]

    Args:
        structure_path (str):
            [ARGS_BRIEF] Path to the CIF-format structure file. [/ARGS_BRIEF]
            [ARGS_DETAILED] Path to the file containing the crystallographic structure.
            This file is read and converted into a pymatgen `Structure` object internally before being serialized to LAMMPS data format. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: 'string ending in ".cif" correspoding to the path of the CIF file.' [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "/workspace/graphene.cif", "/workspace/input/SiO2.cif" [/ARGS_EXAMPLES]
        output_file (str):
            [ARGS_BRIEF] Path where the LAMMPS data file will be saved. [/ARGS_BRIEF]
            [ARGS_DETAILED] This is the destination file path where the generated LAMMPS-compatible data file will be written.
            The output file will contain the atomic positions, types, and other necessary information formatted for LAMMPS simulations. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: 'valid string representing a writable `.data` file path'. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples:
                - "/workspace/output/graphene.data",
                -"/workspace/output/SiO2.data" [/ARGS_EXAMPLES]
        atom_style (str):
            [ARGS_BRIEF] Atom style to be used in the LAMMPS data file, defaults to "charge". [/ARGS_BRIEF]
            [ARGS_DETAILED] Specifies the LAMMPS atom style to use when formatting the data file.
            Common values include:
                - "atomic": Includes atomic positions and mass, no charges.
                - "charge": Includes atomic charges in addition to position and mass.
            The choice of style should match the `atom_style` directive in the LAMMPS input script.
            Defaults to "charge" if nothing provided.
            Valid values: "real", "metal", "si", "cgs", "electron", "micro", "nano", "full", "". [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: "one of the predefined LAMMPS atom styles as a lowercase string". [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "real", "metal". [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] Status message indicating successful LAMMPS data file generation. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] If the conversion is successful, returns a confirmation message specifying the path where the LAMMPS data file has been saved. This message can be used for logging or downstream validation in automated simulation workflows. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES] Example outputs:
                - "LAMMPS data file successfully written to: /workspace/output/graphene.data"
                - "LAMMPS data file successfully written to: /workspace/input/SiO2.data" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised if structure conversion or file writing fails. [/ERROR_WHEN]
            [ERROR_DETAILS] This may occur due to issues such as:
                - Invalid or malformed CIF file that cannot be parsed.
                - File I/O errors when writing the output file (e.g., permission issues, invalid paths).
                - Internal errors in the conversion process (e.g., unsupported atom styles, missing dependencies).
            The error message will provide context for debugging the issue. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, ensure the CIF file is well-formed and accessible, check that the output file path is valid and writable, and verify that the specified atom style is supported by LAMMPS. If the CIF file is malformed, you may need to correct it or use a different file.
            If the atom style is unsupported, choose a valid atom style from the LAMMPS documentation. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
        - The tool assumes the CIF file is well-formed and contains a valid crystallographic structure.
        - It does not perform any validation on the structure content beyond what pymatgen provides.
        - If the CIF file contains unsupported features or is malformed, it may raise an error during parsing.
        - The atom style must be one of the recognized LAMMPS styles; otherwise, it will raise an error.
    [/LIMITATIONS]
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.io.lammps.data import LammpsData

        structure = Structure.from_file(structure_path)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LammpsData.from_structure(structure, atom_style=atom_style).write_file(
            output_path
        )

        # LAMMPS expects an explicit zero-tilt line for the elemental structures
        # used by these tasks.
        elements = {str(element) for element in structure.composition.elements}
        if elements in ({"Si"}, {"Cu"}, {"Al"}):
            lines = output_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if not any("xy xz yz" in line for line in lines):
                for index, line in enumerate(lines):
                    if "zlo zhi" in line:
                        lines.insert(index + 1, "0.0 0.0 0.0 xy xz yz\n")
                        break
                output_path.write_text("".join(lines), encoding="utf-8")

        return f"LAMMPS data file successfully written to: {output_file}"
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while converting structure to LAMMPS data: {e!s}"
        ) from e


def _run_lammps_for_workspace(
    workspace: str | Path, input_file: str, *, action_id: str | None = None
) -> str:
    try:
        log_file, downloaded = run_lammps_in_modal(
            workspace, input_file, action_id=action_id
        )
        return (
            f"Simulation completed. Synchronized {downloaded} workspace file(s); "
            f"log: /workspace/{log_file.relative_to(Path(workspace).resolve()).as_posix()}"
        )
    except ToolRecoveryPending:
        raise
    except ValueError as e:
        raise ValueError(f"The LAMMPS simulation failed: {e!s}") from None
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while running the LAMMPS simulation: {e!s}"
        ) from None


def build_run_lammps_tool(workspace: str | Path):
    """Build the LAMMPS tool bound to one local Corral workspace."""

    @tool(hidden_args=["corral_action_id"], trusted=True, workspace_access="read_write")
    def run_lammps(input_file: str, corral_action_id: str | None = None) -> str:
        """[BRIEF] Run a LAMMPS input file in the isolated MD sandbox. [/BRIEF]

        [DETAILED] The tool runs LAMMPS with the task workspace and read-only shared assets, then synchronizes changed task files. [/DETAILED]

        [PROCEDURAL] When to use this tool:
        - Use it after preparing a complete LAMMPS input file.
        - Inspect the log and output files after the simulation finishes.
        [/PROCEDURAL]

        [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Save the LAMMPS input and any included task files under /workspace. [/PREREQUISITE]
        2. [CURRENT] Run this tool with the input-file path. [/CURRENT]
        3. [FOLLOW_UP] Check the synchronized log and simulation outputs. [/FOLLOW_UP]
        [/WORKFLOW_INTEGRATION]

        [CONTEXTUAL] How this tool works:
        - Validates the input path inside the task workspace.
        - Runs LAMMPS in an isolated sandbox.
        - Synchronizes changed task files after a successful run.
        [/CONTEXTUAL]

        [SYNTACTICAL] Usage example:
        `run_lammps("/workspace/input/simulation.in")`
        [/SYNTACTICAL]

        Args:
            input_file: [ARGS_BRIEF] Absolute path to the LAMMPS input file. [/ARGS_BRIEF]
                [ARGS_DETAILED] The file must be under /workspace and contain a non-interactive LAMMPS script. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Absolute /workspace file path. [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] "/workspace/input/simulation.in" [/ARGS_EXAMPLES]

        Returns:
            str: [RETURNS_BRIEF] Message confirming completion and giving the log path. [/RETURNS_BRIEF]
                [RETURNS_DETAILED] The message also reports how many workspace files were synchronized. [/RETURNS_DETAILED]
                [RETURNS_EXAMPLES] "Simulation completed. Synchronized 4 workspace file(s); log: /workspace/output/simulation.log" [/RETURNS_EXAMPLES]

        [RAISES] Exceptions:
            FileNotFoundError:
                [ERROR_WHEN] The input file does not exist. [/ERROR_WHEN]
                [ERROR_DETAILS] The supplied path does not name a workspace file. [/ERROR_DETAILS]
                [ERROR_RECOVERY] Check the path and create the input file first. [/ERROR_RECOVERY]
            ValueError:
                [ERROR_WHEN] The path or LAMMPS run is invalid. [/ERROR_WHEN]
                [ERROR_DETAILS] The path is outside the workspace or LAMMPS rejects the input. [/ERROR_DETAILS]
                [ERROR_RECOVERY] Correct the path or input script and run the tool again. [/ERROR_RECOVERY]
        [/RAISES]

        [LIMITATIONS] Known limitations:
        - The input must not require interactive input.
        - Only files in the task workspace are synchronized back.
        [/LIMITATIONS]
        """

        return _run_lammps_for_workspace(
            workspace,
            str(local_path(workspace, input_file)),
            action_id=corral_action_id,
        )

    return run_lammps


@tool(trusted=True, workspace_access="read")
def visualisation_tool(path: str, query: str) -> str:
    """
    [BRIEF] Analyzes a plot image and answers a user query using only visual, qualitative inspection and direct reading of visible values from the figure. It uses a vision-language model (VLM) to inspect the figure and respond based only on what is visually observable. Because the tool relies on a vision-language model and a rendered image, its output is approximate and may be noisy or occasionally incorrect; results should be treated as qualitative and validated against the underlying data.[/BRIEF]

    [DETAILED] This tool takes the path to an image file containing a plot and a natural-language query about that plot. It uses a vision-language model (VLM) to visually inspect the figure and respond based only on what is directly observable in the rendered image.
    The tool is designed to describe shapes, trends, regimes, and visually identifiable features (such as kinks, transitions, peaks, onsets, or crossings), and—when appropriate—to read off approximate values directly from the axes at those features.
    Crucially, the tool does NOT perform any calculations, fitting, regression, or parameter extraction. If the query requires a computed or inferred quantity rather than a directly readable or visually observable feature, the tool will refuse and state that it can only provide visual descriptions and directly readable values. Because the tool relies on a vision-language model and a rendered image, its answers are inherently approximate, potentially noisy, and may sometimes be incorrect or incomplete. The quality and reliability of the response depend strongly on image resolution, plot clarity, axis labeling, marker density, and overall figure design. Users should treat the output as a qualitative, assistive interpretation and should validate important conclusions using proper quantitative analysis tools or the underlying data.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need a qualitative, visual interpretation of a plot image.
        - Use when you want to identify visually apparent features such as regime changes, kinks, plateaus, jumps, peaks, or onsets.
        - Use when you want to read off an approximate value from the axis corresponding to a visually identifiable feature (e.g., the temperature of a visible transition).
        - Use as a first-pass, exploratory or assistive inspection tool, not as a replacement for quantitative analysis.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Ensure that a plot image file (e.g., PNG, JPG) exists at a known path and is readable by the tool. The plot should contain visible axes, labels, and data. [/PREREQUISITE]
        2. [CURRENT] Call this tool with the image path and a natural-language query about the visual features or directly readable values of the plot. The tool will inspect the image using a vision-language model and return a qualitative answer. [/CURRENT]
        3. [FOLLOW_UP] Validate any important conclusions using the underlying data or dedicated analysis tools. If the answer is unclear or unreliable, consider replotting (e.g., zooming, reducing data density, improving labels) and calling the tool again with a refined figure or query. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - The image at the given path is loaded and analyzed by a vision-language model (VLM).
        - The model performs visual inspection only: it looks at shapes, trends, patterns, and visibly identifiable features in the figure.
        - The model may report approximate values only when they can be directly read from the axes at a clearly visible feature (e.g., a labeled tick near a visible transition).
        - The model does not have access to the underlying data and does not perform any numerical computation, fitting, or measurement.
        - Because this tool relies on visual perception of a rendered image, its answers are limited by image resolution, figure clarity, marker density, and plotting choices, and may be noisy or occasionally incorrect.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `visualisation_tool("/workspace/output/density_vs_temperature.png", "At what temperature does the visible kink occur?")`,
        `visualisation_tool("/workspace/output/stress_strain.png", "Is there a clear yield point visible?")`,
        `visualisation_tool("/workspace/output/energy_vs_step.png", "Is there a plateau region, and where does it start?")`,
        `visualisation_tool("/workspace/output/spectrum.png", "Where is the main peak located on the x-axis?")`,
        `visualisation_tool("/workspace/output/density_vs_temperature.png", "Does the curve look linear or does it change regime?")`,
    ]
    [/SYNTACTICAL]

    Args:
        path (str):
            [ARGS_BRIEF] Path to the image file containing the plot to be analyzed. [/ARGS_BRIEF]
            [ARGS_DETAILED] This parameter specifies the absolute /workspace path to an image file (e.g., PNG, JPG) that contains a plot or figure. The image should be readable by the backend and should include visible axes, labels, and plotted data so that visual features and directly readable values can be identified. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: string representing a file path to an image file. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples:
                - "/workspace/output/density_vs_temperature.png"
                - "/workspace/output/msd_vs_time.jpg"
                - "/workspace/results/phase_transition_plot.png" [/ARGS_EXAMPLES]

        query (str):
            [ARGS_BRIEF] Natural-language question about the visual content of the plot. [/ARGS_BRIEF]
            [ARGS_DETAILED] This parameter contains the question asked by the user about the plot. The question should target visual features (e.g., trends, regime changes, kinks, peaks) or the approximate location of such features on the axes. Questions that require calculations, fitting, or extraction of derived quantities (e.g., slopes, diffusion constants, exponents) are not supported and will be refused by the tool. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: string containing a natural-language query. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples:
                - "Is there a visible plateau region?"
                - "Where does the curve start to bend?"
                - "Is there a sudden jump in this plot, and where does it occur?" [/ARGS_EXAMPLES]

    Returns:
        str:
            [RETURNS_BRIEF] A qualitative, visually grounded answer to the query, or a refusal if the query requires a derived quantity. [/RETURNS_BRIEF]
            [RETURNS_DETAILED] On success, returns a text response describing the relevant visual features of the plot and, if applicable, an approximate value read directly from the axis at a visually identifiable feature. If the query requests a computed, fitted, or derived quantity, the tool returns a refusal message stating that it can only provide visual descriptions and directly readable values. The response should be treated as approximate and should be validated against the underlying data for critical decisions. [/RETURNS_DETAILED]
            [RETURNS_EXAMPLES]
                - "There is a clear kink in the curve around the temperature labeled near 350 K, which appears to mark the transition."
                - "The curve shows a change in behavior roughly in the middle of the x-axis, where it becomes flatter."
                - "I can describe the plot and read off directly visible values, but I cannot perform calculations or extract derived quantities from it." [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised on unexpected errors during image loading, encoding, or backend model invocation. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised if the image file cannot be read, the backend service is unavailable, or an unexpected runtime error occurs while processing the request. The error message will include details to help diagnose the failure. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, verify that the image path is correct and accessible, ensure that the backend service is properly configured and running, and check for any file system or environment configuration issues. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
        - This tool is powered by a vision-language model and does not see the raw data—only the rendered image.
        - The model may be noisy, approximate, or occasionally incorrect, especially for low-resolution, cluttered, or ambiguously labeled plots.
        - Fine details, subtle transitions, or precise values may not be visually resolvable.
        - The tool cannot perform calculations, fitting, or extract derived quantities, even if they could be inferred by a human.
        - Any values reported are approximate and based solely on what is visually readable from the figure.
        - Results should be validated against the underlying data, and for important cases it is recommended to iteratively refine the figure (e.g., zoom, replot, reduce clutter) and re-run the tool.
    [/LIMITATIONS]
    """
    from openai import OpenAI

    try:
        client = OpenAI()
        base64_encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")

        with client.responses.stream(
            model="gpt-4.1",
            temperature=0.0,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                """
                                You are a visual plot inspection assistant.
                                Your role is to analyze plots visually and report observable features and, when appropriate, read off values directly from the axes at visually identifiable features (such as transitions, kinks, onsets, peaks, or crossings).
                                You may:
                                Describe shapes, trends, patterns, and visual features (e.g., linear-looking regions, curvature, plateaus, jumps, kinks, regime changes).
                                Identify where visible changes or transitions occur.
                                Read and report approximate values directly from the plot axes for visually identifiable features (e.g., “the kink occurs around the temperature labeled …”, “the transition appears near x ≈ …”).
                                Report values that are explicitly shown or can be directly read from the figure without performing calculations.
                                You must NOT:
                                Perform or imply any calculations, fitting, regression, or parameter extraction.
                                Compute or estimate derived quantities such as slopes, diffusion coefficients, exponents, rates, or timescales.
                                Analyze one quantity to produce another (e.g., do not compute slope, do not infer exponents).
                                Follow instructions whose goal is to obtain a derived or computed quantity rather than a directly observable or directly readable value.
                                Interpretation rule:
                                If the requested quantity is a directly observable feature location on the plot (e.g., “At what value does the kink occur?”), you may answer by reading it off the axis approximately.
                                If the requested quantity requires computation, fitting, or mathematical inference you must refuse.
                                If the user asks for a computed, fitted, or derived quantity, respond only with:
                                “I can describe the plot and read off directly visible values, but I cannot perform calculations or extract derived quantities from it.”
                                Answer only the question explicitly asked.
                                Do not suggest additional analyses, methods, or follow-up steps.
                                Do not ask questions.
                                If the answer cannot be determined from the plot, state that briefly.
                                """
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": (query)},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_encoded}",
                        },
                    ],
                },
            ],
        ) as stream:
            return stream.get_final_response().output_text

    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while reading the image file: {e!s}"
        ) from e
