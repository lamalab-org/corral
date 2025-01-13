from corral.base import Tool, ToolArgument
from corral.utils import tool
from ase import Atoms
from dotenv import load_dotenv
import os
import subprocess
from typing import List, Optional

# load_dotenv("../.env")

@tool
def run_lammps(input_file: str, log_file: str) -> str:
    """Runs a LAMMPS simulation with the specified input file and outputs to a log file.

    Args:
        input_file: Path to the LAMMPS input script.
        log_file: Path to the log file where LAMMPS will write its output.
    """
    if not os.path.exists(input_file):
        return f"Error: Input file not found: {input_file}"

    # Construct the LAMMPS command with the -log option
    command = f"lmp -in {input_file} -log {log_file}"
    try:
        print(f"Executing LAMMPS command: {command}")
        # Run the command and capture the output
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        # Return a success message with the log file location
        return f"LAMMPS simulation completed successfully. Log written to {log_file}."
    except subprocess.CalledProcessError as e:
        # Combine stdout and stderr for better error diagnostics
        output = (e.stdout or "") + (e.stderr or "")
        error_message = (
            f"Error: LAMMPS simulation failed with exit code {e.returncode}.\n"
            f"Command: {e.cmd}\n"
            f"Error output: {output.strip() if output else 'No error message captured'}"
        )
        print(error_message)
        return error_message
    except Exception as e:
        unexpected_error_message = f"An unexpected error occurred: {str(e)}"
        print(unexpected_error_message)
        return unexpected_error_message

@tool
def bash_command(command: str, args: Optional[List[str]] = None) -> str:
    """Executes a bash command with optional arguments.

    Args:
        command: The bash command to run (e.g., 'ls', 'cp').
        args: A list of arguments to pass to the command (e.g., file paths).
    """
    if not command:
        raise ValueError("The 'command' argument is required.")

    # Combine command and arguments
    full_command = [command] + (args or [])
    print(f"Executing command: {' '.join(full_command)}")

    try:
        result = subprocess.run(full_command, check=True, text=True, capture_output=True)
        return result.stdout.strip()  # Return command output
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {e.stderr.strip()}")
        raise

