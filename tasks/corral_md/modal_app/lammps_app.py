from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import fsspec
import modal
from envs_tools.lammps import lammps_image
from loguru import logger
from modal import App

simagent_name = os.getenv("SIMAGENT_NAME", "")
if simagent_name and not simagent_name.startswith("-"):
    simagent_name = f"-{simagent_name}"

# Create the app
app = App(f"simagent{simagent_name}")
volume_potential = modal.Volume.from_name("potentials", create_if_missing=True)
volume_sim = modal.Volume.from_name("simulations", create_if_missing=True)
volume_struct = modal.Volume.from_name("structures", create_if_missing=True)
volume_test_files = modal.Volume.from_name("test_files", create_if_missing=True)
volume_image_files = modal.Volume.from_name("image_files", create_if_missing=True)
volume_eval_struct = modal.Volume.from_name("eval_structures", create_if_missing=True)

CPUS = 2

# with volume_potential.batch_upload() as batch:
#     batch.put_directory("./potentials/", "/")

# with volume_struct.batch_upload() as batch:
#     batch.put_directory("./structures/", "/")

# with volume_test_files.batch_upload() as batch:
#     batch.put_directory("./test_files/", "/")

# with volume_image_files.batch_upload() as batch:
#     batch.put_directory("./image_files/", "/")

# with volume_eval_struct.batch_upload() as batch:
#     batch.put_directory("./eval_structures/", "/")


def _run_lammps(
    input_file: str, log_file: str, directory_path: str | None = None, CPUS: int = 1
) -> None:
    """
    Runs a LAMMPS simulation using a specified input file and writes the log output to a given log file.

    Args:
        input_file (str): Path to the LAMMPS input script file.
        log_file (str): Path where the log file output will be stored.
        local_workspace (str | None): Local workspace prefix to rewrite in the
            uploaded input script.
        remote_workspace (str | None): Modal-mounted replacement for that prefix.

    Returns:
        dict: A dictionary containing the log file content and input file content.

    Raises:
        ValueError: If the LAMMPS simulation fails.
    """
    import os
    import subprocess

    lmp_command = "/root/lammps/build/lmp"
    original_cwd = Path.cwd()
    if directory_path:
        os.chdir(directory_path)
    try:
        # sanitize_lammps_input_inplace(input_file)
        command = [
            "mpirun",
            "--allow-run-as-root",
            "--bind-to",
            "core",
            "--map-by",
            "core",
            "-np",
            str(CPUS),
            lmp_command,
            "-in",
            input_file,
            "-log",
            log_file,
        ]
        subprocess.run(command, shell=False, check=True, capture_output=True, text=False)
        # log_file = Path(log_file)
        with Path(log_file).open("rb") as log_f:
            log_content = log_f.read()
        text = log_content.decode("utf-8", errors="ignore")
        with Path(log_file).open("w", encoding="utf-8") as dst:
            dst.write(text)
    except subprocess.CalledProcessError:
        import log_lammps_reader

        log_path = Path(log_file)
        if log_path.exists():
            raw = log_path.read_bytes()
            log_path.write_text(raw.decode("utf-8", errors="ignore"), encoding="utf-8")

        try:
            error_log = log_lammps_reader.log_starts_with(log_file, "ERROR")
        except Exception:
            error_log = "Could not read error log"

        raise ValueError(f"LAMMPS simulation failed: {error_log}") from None
    finally:
        # Sanitize log file regardless of success or failure
        log_path = Path(log_file)
        if log_path.exists():
            raw = log_path.read_bytes()
            log_path.write_text(raw.decode("utf-8", errors="ignore"), encoding="utf-8")
        
        os.chdir(original_cwd)  # Restore original directory

# def _run_lammps(
#     input_file: str, log_file: str, directory_path: str | None = None, CPUS: int = 1
# ) -> None:
#     import os
#     import subprocess

#     lmp_command = "/root/lammps/build/lmp"
#     original_cwd = Path.cwd()
#     if directory_path:
#         os.chdir(directory_path)
#     try:
#         command = [
#             "mpirun",
#             "--allow-run-as-root",
#             "--bind-to",
#             "core",
#             "--map-by",
#             "core",
#             "-np",
#             str(CPUS),
#             lmp_command,
#             "-in",
#             input_file,
#             "-log",
#             log_file,
#         ]
#         subprocess.run(command, shell=False, check=True, capture_output=True, text=True)
#         with Path(log_file).open("rb") as log_f:
#             log_content = log_f.read()
#         text = log_content.decode("utf-8", errors="ignore")
#         with Path(log_file).open("w", encoding="utf-8") as dst:
#             dst.write(text)
#     # except subprocess.CalledProcessError:
#     #     import log_lammps_reader

#     #     log_path = Path(log_file)
#     #     if log_path.exists():
#     #         raw = log_path.read_bytes()
#     #         log_path.write_text(raw.decode("utf-8", errors="ignore"), encoding="utf-8")

#     #     error_log = log_lammps_reader.log_starts_with(log_file, "ERROR")
#     #     raise ValueError(f"LAMMPS simulation failed: {error_log}") from None
#     except subprocess.CalledProcessError:
#         import log_lammps_reader

#         log_path = Path(log_file)
#         print(f"Log file exists: {log_path.exists()}")  # ← add this
#         if log_path.exists():
#             raw = log_path.read_bytes()
#             print(f"Raw bytes length: {len(raw)}")  # ← add this
#             log_path.write_text(raw.decode("utf-8", errors="ignore"), encoding="utf-8")
#             print("Sanitization done")  # ← add this

#         error_log = log_lammps_reader.log_starts_with(log_file, "ERROR")
#         raise ValueError(f"LAMMPS simulation failed: {error_log}") from None
#     finally:
#         os.chdir(original_cwd)


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the parent directory of a file path exists.

    Args:
        file_path: Path to a file
    """
    from pathlib import Path

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def generate_output_capture_code() -> str:
    """
    Generate the Python code string for capturing execution results.

    This utility function creates the code that gets appended to user code
    to capture variables and prepare them for JSON serialization.

    Returns:
        str: Python code string for variable capture
    """
    return """

import json
import types
import sys
from collections.abc import Iterable

result_ = {}

def is_json_serializable(obj, max_depth=10, current_depth=0):
    '''Check if object is JSON serializable with depth limit'''
    if current_depth > max_depth:
        return False

    try:
        # Handle basic types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return True

        # Handle sequences (but not strings)
        if isinstance(obj, (list, tuple)):
            return all(is_json_serializable(item, max_depth, current_depth + 1) for item in obj)

        # Handle dictionaries
        if isinstance(obj, dict):
            return all(
                isinstance(k, str) and is_json_serializable(v, max_depth, current_depth + 1)
                for k, v in obj.items()
            )

        # Quick test with actual JSON serialization for edge cases
        json.dumps(obj)
        return True
    except (TypeError, ValueError, RecursionError, OverflowError):
        return False

def safe_convert_to_serializable(obj):
    '''Convert common non-serializable types to serializable ones'''
    try:
        import numpy as np
        # Handle numpy arrays
        if hasattr(obj, '__module__') and obj.__module__ == 'numpy':
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
    except ImportError:
        pass

    # Handle sets
    if isinstance(obj, set):
        return list(obj)

    # Handle other iterables (but not strings/bytes)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
        try:
            return list(obj)
        except:
            pass

    return obj

# Try to capture from preferred variable names first
preferred_vars = ['output', 'result', 'filtered_data', 'processed_data', 'dataset']
captured = False

for var_name in preferred_vars:
    if var_name in locals():
        var_value = locals()[var_name]
        converted_value = safe_convert_to_serializable(var_value)
        if is_json_serializable(converted_value):
            result_[var_name] = converted_value
            captured = True
            break

# Fallback: capture any user-defined variables
if not captured:
    excluded = {
        '__builtins__', '__name__', '__doc__', '__package__', '__loader__',
        '__spec__', '__annotations__', '__cached__', '__file__',
        'json', 'types', 'sys', 'Iterable', 'result_', 'preferred_vars',
        'captured', 'excluded', 'local_vars', 'is_json_serializable',
        'safe_convert_to_serializable', 'var_name', 'var_value', 'converted_value'
    }

    local_vars = dict(locals())  # Create snapshot

    # Collect all suitable variables
    suitable_vars = {}
    for var_name, var_value in local_vars.items():
        if (not var_name.startswith('_') and
            var_name not in excluded and
            not isinstance(var_value, types.ModuleType) and
            not callable(var_value)):

            converted_value = safe_convert_to_serializable(var_value)
            if is_json_serializable(converted_value):
                suitable_vars[var_name] = converted_value

    # If we have suitable variables, include them all
    if suitable_vars:
        result_.update(suitable_vars)

print('EXECUTION_RESULT:', json.dumps(result_))
"""


def safe_convert_timeout(timeout) -> int:
    """
    Safely convert timeout parameter to integer.

    Args:
        timeout: Timeout value (could be string, int, float, or None)

    Returns:
        int: Valid timeout value
    """
    if timeout is None:
        return 300

    try:
        return int(float(timeout))  # Handle both "60" and "60.5"
    except (ValueError, TypeError):
        return 300  # Default fallback


def parse_execution_output(stdout: str) -> tuple[dict, list[str]]:
    """
    Parse subprocess output to extract execution results and regular output.

    Args:
        stdout: Raw stdout from subprocess

    Returns:
        tuple: (execution_result dict, output_lines list)
    """
    from contextlib import suppress

    stdout_lines = stdout.strip().split("\n") if stdout.strip() else []
    execution_result = {}
    output_lines = []

    for line in stdout_lines:
        if line.startswith("EXECUTION_RESULT:"):
            with suppress(json.JSONDecodeError):
                execution_result = json.loads(line[17:])
        else:
            output_lines.append(line)

    return execution_result, output_lines

@app.function(
    image=lammps_image,
    cpu=1,
    memory=5120,
    volumes={
        "/results": volume_sim,
        "/eval_structures": volume_eval_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def check_structure(target, atom_style, result):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.lammps.data import LammpsData
    volume_sim.reload()
    try:
        ld1 = LammpsData.from_file(target, atom_style=atom_style)
        ld2 = LammpsData.from_file(result, atom_style=atom_style)
        matcher = StructureMatcher()
        are_equal = matcher.fit(ld1.structure, ld2.structure)
        return 1.0 if are_equal else 0.0
    except Exception:
        logger.exception("Error in check_structure")  # full traceback here
        return 0.0
    finally:
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=1,
    memory=5120,
    volumes={
        "/results": volume_sim,
        "/test_files": volume_test_files,
        "/potentials": volume_potential.read_only(),
    },
)
def check_potential(target: str, result: str):
    volume_sim.reload()
    try:
        with open(target, "r") as f:
            target_content = f.read()
        with open(result, "r") as f:
            result_content = f.read()
        return 1.0 if target_content == result_content else 0.0
    except Exception as e:
        logger.warning(f"Error in check_potential: {e}")
        return 0.0
    finally:
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=1,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def get_nth_run_log(
    path: str, n: int = 0, save: str | None = None, index: int | None = None
) -> str:
    import log_lammps_reader

    volume_sim.reload()
    try:
        final_string = ""
        log_data = log_lammps_reader.parse(path, n)
        final_string += f"{log_data.head()}\n"
        if save:
            log_data.write_csv(save)
            final_string += f"Thermo data for run {n} saved to {save}.\n"
        if index is not None:
            if 0 <= index < log_data.height:
                row = log_data.row(index)
                final_string += f"Data at index {index}: {row}\n"
            else:
                final_string += f"Index {index} is out of bounds for data with {log_data.height} rows.\n"
        return final_string
    except Exception as e:
        return f"Failed to parse thermo data for run {n}: {e}"
    finally:
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=1,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def keyword_log_extractor(path: str, keyword: str) -> dict:
    import log_lammps_reader

    result = {}
    volume_sim.reload()
    try:
        fixes = log_lammps_reader.log_starts_with(path, keyword)
        if not fixes:
            raise ValueError(f"Keyword '{keyword}' not found in log.")
        result[keyword] = fixes
        return result
    except Exception as e:
        return {"error": f"Error processing keyword '{keyword}': {e}"}
    finally:
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=1,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def execute_python_script(
    script_path: str,
    args: list | None = None,
    timeout: int = 600,
    working_dir: str | None = None,
) -> str:
    import sys

    volume_sim.reload()
    try:
        if not Path(script_path).exists():
            return json.dumps(
                {"success": False, "error": f"Script file not found: {script_path}"}
            )

        # Prepare command
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(str(arg) for arg in args)

        # Execute
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir or Path.cwd(),
            check=False,
        )

        result = {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "return_code": process.returncode,
            "command": " ".join(cmd),
        }
        volume_sim.commit()
        return json.dumps(result, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "success": False,
                "error": f"Script execution timed out after {timeout} seconds",
            }
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
    finally:
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=1,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def execute_python_code(
    python_code: str,
    input_data: str | None = None,
    save_output_to: str | None = None,
    timeout: int = 300,
) -> str:
    volume_sim.reload()
    import subprocess
    import sys
    import traceback
    from pathlib import Path

    try:
        # Ensure timeout is valid
        timeout = safe_convert_timeout(timeout)

        # Create directory if needed
        if save_output_to:
            ensure_directory_exists(save_output_to)

        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Prepare the code with input data if provided
            full_code = ""
            if input_data:
                full_code += "import json\n"
                full_code += f"input_data = json.loads({input_data!r})\n"

            full_code += python_code
            full_code += generate_output_capture_code()

            f.write(full_code)
            temp_file = f.name

        # Execute the code
        process = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
            check=False,
        )

        # Clean up
        Path(temp_file).unlink()

        # Parse output
        execution_result, output_lines = parse_execution_output(process.stdout)

        # Save output if requested
        saved_path = None
        if save_output_to and execution_result:
            try:
                with Path(save_output_to).open("w") as f:
                    json.dump(execution_result, f, indent=2)
                saved_path = save_output_to
            except Exception as e:
                logger.warning(
                    f"Warning: Failed to save output to {save_output_to}: {e}"
                )

        result = {
            "success": process.returncode == 0,
            "stdout": "\n".join(output_lines),
            "stderr": process.stderr,
            "return_code": process.returncode,
            "execution_result": execution_result,
            "saved_to": saved_path,
        }
        if process.returncode != 0:
            result["error"] = process.stderr or "Script execution failed"

        return json.dumps(result, indent=2)

    except subprocess.TimeoutExpired:
        from pathlib import Path

        if "temp_file" in locals() and Path(temp_file).exists():
            Path(temp_file).unlink()
        return json.dumps(
            {
                "success": False,
                "error": f"Code execution timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_result": {},
            }
        )
    except Exception as e:
        if "temp_file" in locals() and Path(temp_file).exists():
            Path(temp_file).unlink()
        return json.dumps(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_result": {},
            }
        )
    finally:
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=CPUS,
    timeout=7200,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def run_lammps(
    input_file: str,
    log_file: str,
    local_workspace: str | None = None,
    remote_workspace: str | None = None,
) -> None:
    """
    Run a LAMMPS simulation.

    Args:
        input_file (str): Path to the LAMMPS input script file.
        log_file (str): Path where the log file output will be stored.


    Raises:
        ValueError: If the simulation fails.
    """
    def sanitize_lammps_input_inplace(input_file: str) -> None:
        import re
        import fsspec
        logger.info("Sanitizing LAMMPS input: %s", input_file)

        fs = fsspec.filesystem("file")

        volume_sim.reload()

        # Read
        try:
            with fs.open(input_file, "r") as f:
                text = f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading file {input_file}: {e}") from e

        logger.info("Original input content:\n%s", text)

        original_text = text
        if local_workspace and remote_workspace:
            text = text.replace(local_workspace, remote_workspace)

        log_cmd_re = re.compile(r"^\s*log\s+", re.IGNORECASE)
        lines = text.splitlines(keepends=True)

        cleaned = []
        removed = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                cleaned.append(line)
                continue
            if log_cmd_re.match(stripped):
                removed += 1
                continue
            cleaned.append(line)

        new_text = "".join(cleaned)

        if removed == 0 and new_text == original_text:
            logger.info(
                "No log commands or local workspace paths found; no changes made."
            )
            return

        # Write
        try:
            with fs.open(input_file, "w") as f:
                f.write(new_text)
        except Exception as e:
            raise RuntimeError(f"Error writing file {input_file}: {e}") from e

        logger.info("Removed %d log command(s) from %s", removed, input_file)
        volume_sim.commit()

    original_input: str | None = None
    try:
        volume_sim.reload()
        original_input = Path(input_file).read_text(encoding="utf-8")
        sanitize_lammps_input_inplace(input_file)
        volume_sim.reload()
        input_path = Path(input_file)
        directory_path = input_path.parent
        input_file_ = input_path.name
        _run_lammps(input_file_, log_file, str(directory_path), CPUS)
        volume_sim.commit()

    except Exception as e:
        # raise ValueError(f"LAMMPS simulation failed: {e!s}") from e

        log_path = Path(input_file).parent / log_file
        if log_path.exists():
            raw = log_path.read_bytes()
            log_path.write_text(raw.decode("utf-8", errors="ignore"), encoding="utf-8")
        raise ValueError(f"{e!s}") from e
    finally:
        # The local copy remains the source of truth. Restore the uploaded input
        # after any path rewriting so downloading the workspace never leaks a
        # Modal-only /results/corral/jobs/... path into it.
        if original_input is not None:
            Path(input_file).write_text(original_input, encoding="utf-8")
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def convert_structure_to_lammps_data(
    structure: str, output_file: str, atom_style: str = "charge"
) -> None:
    """
    Convert a CIF-format structure into a LAMMPS data file. This function takes a CIF string as input (assumed to be the output from the previous tool),
    converts it to a pymatgen Structure object, and then generates the corresponding LAMMPS data
    file.

    Args:
        structure: The CIF-format structure as a string.
        output_file: The path to the output LAMMPS data file to be written.
        atom_style: The atom style for LAMMPS. Common values include:
            - "atomic": Basic atomic positions and masses.
            - "charge": Atoms with charge data (default).

    Returns:
        None. Writes the LAMMPS data to the specified output file.
    """
    from pymatgen.core import Structure
    from pymatgen.io.lammps.data import LammpsData

    # Convert CIF string to pymatgen Structure object
    volume_sim.reload()
    try:
        # Load structure from CIF file
        structure_obj = Structure.from_file(structure)

        # Check if the structure contains ONLY Si
        elements = {str(el) for el in structure_obj.composition.elements}
        is_silicon_only = elements == {"Si"} or elements == {"Cu"} or elements == {"Al"}

        # Convert to LAMMPS data
        lammps_data = LammpsData.from_structure(structure_obj, atom_style=atom_style)
        lammps_data.write_file(output_file)

        # Only modify the file if it's pure silicon
        if is_silicon_only:
            with Path(output_file).open() as f:
                lines = f.readlines()

            # Insert tilt line after zlo zhi
            for i, line in enumerate(lines):
                if "zlo zhi" in line:
                    lines.insert(i + 1, "0.0 0.0 0.0 xy xz yz\n")
                    break

            # Write final file
            with Path(output_file).open("w") as f:
                f.writelines(lines)

        volume_sim.commit()

    except Exception as e:
        raise ValueError(
            f"An unexpected error occurred while converting to Lammps data format: {e!s}"
        ) from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def run_bash_command(
    command: str, args: list[str] | None = None, shell: bool = False
) -> str:
    """Execute a bash command with optional arguments.

    Args:
        command: The bash command to execute (e.g., 'mkdir', 'ls').
        args: A list of arguments for the bash command.
    """
    # volume_potential.reload()

    volume_sim.reload()
    if args is None:
        args = []
    try:
        # Construct the full command with arguments
        full_command = [command, *args]

        # Run the command
        result = subprocess.run(
            full_command,
            shell=shell,  # Avoid shell injection risks
            check=True,  # Raise CalledProcessError on failure
            capture_output=True,
            text=True,
        )

        # Return the standard output
        volume_sim.commit()
        return result.stdout

    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e!s}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def list_files(path: str, recursive: bool = False) -> list[str]:
    """List files in a directory"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        return fs.ls(path, detail=False, recursive=recursive)
    except Exception as e:
        raise RuntimeError(f"Error listing files: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def read_file(path: str) -> str:
    """Read contents of a file"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, "r") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading files: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
        "/image_files": volume_image_files,
    },
)
def read_file_mode(path: str, mode="rb") -> str:
    """Read contents of a file"""
    import base64

    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, mode) as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error reading files: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def read_large_file(path: str, start: int, length: int, encoding: str) -> str:
    """Read specific portion of a large file"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, "rb") as f:
            f.seek(start)
            chunk = f.read(length)
        return chunk.decode(encoding, errors="replace")
    except Exception as e:
        f"[ERROR] Could not read file chunk: {e}"


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def write_file(path: str, content: str) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        with fs.open(path, "w") as f:
            f.write(content)
        volume_sim.commit()
    except Exception as e:
        raise RuntimeError(f"Error writing files: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def file_info(path: str) -> dict:
    """Get file information"""
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        return fs.info(path)
    except Exception as e:
        raise RuntimeError(f"Error getting file info: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def copy_file(source: str, destination: str) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        fs.copy(source, destination)
        volume_sim.commit()
    except Exception as e:
        raise RuntimeError(f"Error copying from {source} to {destination}: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def move_file(source: str, destination: str) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        # fsspec does not always provide a move method; if not, copy then remove.
        fs.mv(source, destination)
        volume_sim.commit()
    except Exception:
        copy_file(source, destination)
        fs.rm(source)


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def mkdir(path: str, create_parents: bool = False) -> None:
    fs = fsspec.filesystem("file")
    volume_sim.reload()
    try:
        if create_parents and hasattr(fs, "mkdirs"):
            # Many fsspec implementations support mkdirs.
            # If not available, fall back to calling mkdir for each missing part.
            fs.mkdirs(path, exist_ok=True)
        else:
            fs.mkdir(path)
        volume_sim.commit()
    except Exception as e:
        raise RuntimeError(f"Error creating directory {path}: {e}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def cat_files(paths: list[str], separator: str = "\n") -> str:
    contents = []
    for path in paths:
        try:
            contents.append(read_file(path))
        except Exception as e:
            raise RuntimeError(f"Error reading file {path}: {e}") from e
    return separator.join(contents)
