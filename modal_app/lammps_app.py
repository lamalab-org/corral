from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import fsspec
import modal
from envs_tools.lammps import _run_lammps, lammps_image
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

CPUS = 1

# with volume_potential.batch_upload() as batch:
#     batch.put_directory("./potentials/", "/")

# with volume_struct.batch_upload() as batch:
#     batch.put_directory("./structures/", "/")

# with volume_test_files.batch_upload() as batch:
#     batch.put_directory("./test_files/", "/")

# Global registry for terminal sessions
_terminal_sessions: dict[str, dict[str, Any]] = {}
_session_lock = threading.Lock()
MAX_OUTPUT_SIZE = 60 * 1024


def truncate_output(output: str, max_size: int = MAX_OUTPUT_SIZE) -> str:
    """
    Truncate output if it exceeds maximum size.

    Args:
        output (str): The output string to truncate
        max_size (int): Maximum size in bytes. Defaults to 60KB.

    Returns:
        str: Truncated output with informational message if needed
    """
    if len(output) > max_size:
        # Keep 80% from the beginning and 20% from the end
        truncation_msg = f"\n\n... [Output truncated: {len(output) - max_size} bytes omitted] ...\n\n"
        msg_size = len(truncation_msg)
        available_size = max_size - msg_size

        beginning_size = int(available_size * 0.8)
        end_size = available_size - beginning_size

        return output[:beginning_size] + truncation_msg + output[-end_size:]
    return output


def get_or_create_session(session_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """
    Get an existing terminal session or create a new one.

    Args:
        session_id (str | None): Optional session ID to retrieve

    Returns:
        tuple[str, dict[str, Any]]: Tuple of (session_id, session_dict)
    """
    with _session_lock:
        if session_id and session_id in _terminal_sessions:
            return session_id, _terminal_sessions[session_id]

        # Create new session
        new_session_id = str(uuid.uuid4())
        _terminal_sessions[new_session_id] = {
            "id": new_session_id,
            "cwd": str(Path.cwd()),
            "env": os.environ.copy(),
            "history": [],
            "created_at": time.time(),
        }
        logger.info(f"Created new terminal session: {new_session_id}")
        return new_session_id, _terminal_sessions[new_session_id]


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
    import json
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
    cpu=CPUS,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
        "/test_files": volume_test_files,
    },
)
def run_in_terminal(
    command: str,
    timeout: int | None = 300,
    session_id: str | None = None,
) -> str:
    logger.info(f"Executing terminal command: '{command}' (timeout={timeout})")
    volume_sim.reload()
    try:
        # Get or create session
        sess_id, session = get_or_create_session(session_id)
        cwd = session["cwd"]
        env = session["env"]

        # Execute command and wait for completion
        logger.info(f"Executing command in directory: {cwd}")

        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
            text=True,
            check=False,
        )

        # Truncate output if needed
        stdout = truncate_output(process.stdout)
        stderr = truncate_output(process.stderr)

        # Check for directory changes (extract from commands like 'cd')
        if command.strip().startswith("cd "):
            # Simple cd parsing - in reality, this is complex
            parts = command.strip().split(maxsplit=1)
            if len(parts) > 1:
                new_dir = parts[1].strip()
                # Resolve relative to current cwd
                target_path = Path(cwd) / new_dir
                if target_path.exists() and target_path.is_dir():
                    session["cwd"] = str(target_path.resolve())
                    logger.info(
                        f"Updated session working directory to: {session['cwd']}"
                    )

        # Record in session history
        session["history"].append(
            {
                "command": command,
                "exit_code": process.returncode,
                "timestamp": time.time(),
            }
        )

        result = {
            "success": process.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": process.returncode,
            "command": command,
            "cwd": cwd,
            "session_id": sess_id,
        }

        logger.info(f"Command completed with exit code: {process.returncode}")

        return json.dumps(result, indent=2)

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds: {command}")
        return json.dumps(
            {
                "success": False,
                "error": "TimeoutExpired",
                "message": f"Command execution exceeded timeout of {timeout} seconds",
                "command": command,
                "timeout": timeout,
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return json.dumps(
            {
                "success": False,
                "error": type(e).__name__,
                "message": str(e),
                "command": command,
            },
            indent=2,
        )
    finally:
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=CPUS,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
        "/test_files": volume_test_files,
    },
)
def execute_python_script(
    script_path: str,
    args: list | None = None,
    timeout: int = 600,
    working_dir: str | None = None,
) -> str:
    import json
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
    cpu=CPUS,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
        "/test_files": volume_test_files,
    },
)
def list_terminal_sessions() -> str:
    volume_sim.reload()
    try:
        with _session_lock:
            sessions_info = []
            for sess_id, session in _terminal_sessions.items():
                sessions_info.append(
                    {
                        "session_id": sess_id,
                        "cwd": session["cwd"],
                        "command_count": len(session["history"]),
                        "created_at": session["created_at"],
                    }
                )

        result = {
            "success": True,
            "session_count": len(sessions_info),
            "sessions": sessions_info,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return json.dumps(
            {
                "success": False,
                "error": type(e).__name__,
                "message": str(e),
            },
            indent=2,
        )
    finally:
        volume_sim.commit()


@app.function(
    image=lammps_image,
    cpu=CPUS,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
    import json
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
    timeout=3600,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
        "/test_files": volume_test_files,
    },
)
def run_lammps(input_file: str, log_file: str) -> None:
    """
    Run a LAMMPS simulation.

    Args:
        input_file (str): Path to the LAMMPS input script file.
        log_file (str): Path where the log file output will be stored.


    Raises:
        ValueError: If the simulation fails.
    """
    volume_sim.reload()
    input_path = Path(input_file)
    directory_path = input_path.parent
    input_file_ = input_path.name
    try:
        _run_lammps(input_file_, log_file, str(directory_path))
        volume_sim.commit()

    except Exception as e:
        raise ValueError(f"LAMMPS simulation failed: {e!s}") from e


@app.function(
    image=lammps_image,
    cpu=1.0,
    memory=5120,
    volumes={
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
        "/potentials": volume_potential,
        "/results": volume_sim,
        "/structures": volume_struct,
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
