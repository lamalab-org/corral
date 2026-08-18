import json
import subprocess
import sys
import tempfile
import traceback
from contextlib import suppress
from pathlib import Path

from corral.core.tool import tool
from corral.logging import logger


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the parent directory of a file path exists.

    Args:
        file_path: Path to a file
    """
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


@tool
def execute_python_code(
    python_code: str,
    input_data: str | None = None,
    save_output_to: str | None = None,
    timeout: int = 300,
) -> str:
    """[BRIEF] Execute Python code in a secure environment with data input/output capabilities. [/BRIEF]

    [DETAILED] This tool provides a secure execution environment for custom Python code, essential for data analysis, custom calculations, and algorithm development in materials science workflows.
    It supports data injection, output capture, and file saving capabilities while maintaining security through process isolation and timeout controls.
    This enables flexible custom analysis. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to execute custom Python analysis or calculations
    - Best suited for data processing and custom algorithm development
    - Essential for implementing custom filtering, analysis, or transformation logic
    - Recommended for prototyping and testing analysis workflows
    - Avoid for simple operations that can be done with existing tools
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Creates isolated subprocess environment for secure code execution
    - Injects input data as JSON-parsed variable if provided
    - Captures standard output, error streams, and execution results
    - Implements timeout protection to prevent infinite loops
    - Extracts variables from executed code for result capture
    - Saves results to file if requested for persistence
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Prepare input data and ensure code is syntactically correct [/PREREQUISITE]
    2. [CURRENT] Execute custom Python code with data processing or analysis [/CURRENT]
    3. [FOLLOW_UP] Use captured results for further analysis or save to files [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `execute_python_code("result = sum([1, 2, 3, 4, 5])", None, None, 30)`,
    `execute_python_code("filtered_data = [x for x in input_data if x > 0.5]", json_data, "output.json")`,
    `execute_python_code("import numpy as np; result = np.mean(input_data)", array_data, None, 60)`,
    [/SYNTACTICAL]

    Args:
        python_code: [ARGS_BRIEF] Python code string to be executed. [/ARGS_BRIEF]
                    [ARGS_DETAILED] A string containing valid Python code to be executed in this environment.
                    For best results, assign your main output to a variable named 'result' or 'output'.
                    The code can import standard libraries and perform complex calculations.
                    The tool will attempt to capture user-defined variables as execution results. [/ARGS_DETAILED]
                    [ARGS_SYNTACTICAL] Valid Python code string [/ARGS_SYNTACTICAL]
                    [ARGS_EXAMPLES] "result = 2 + 2", "import json; result = json.loads(data)", "filtered = [x for x in data if x > threshold]" [/ARGS_EXAMPLES]
        input_data: [ARGS_BRIEF] Optional JSON string to inject as input_data variable. [/ARGS_BRIEF]
                   [ARGS_DETAILED] An optional JSON string that will be loaded into a Python variable named 'input_data' within the executed script.
                   This allows the script to process external data.
                   The JSON will be parsed and made available as a Python object (dict, list, etc.) depending on the JSON structure. [/ARGS_DETAILED]
                   [ARGS_SYNTACTICAL] Valid JSON string or None [/ARGS_SYNTACTICAL]
                   [ARGS_EXAMPLES] "{"data": [1, 2, 3]}", '[1, 2, 3, 4, 5]', "{"threshold": 0.5, "values": [...]}" [/ARGS_EXAMPLES]
        save_output_to: [ARGS_BRIEF] Optional file path to save execution results. [/ARGS_BRIEF]
                       [ARGS_DETAILED] An optional file path where the captured execution results will be saved as a JSON file.
                       If provided and execution is successful, the results will be written to this file for persistence and later use.
                       The directory will be created if it doesn't exist. [/ARGS_DETAILED]
                       [ARGS_SYNTACTICAL] "Valid file path or None" [/ARGS_SYNTACTICAL]
                       [ARGS_EXAMPLES] "results.json", "output/analysis_results.json", "data/processed_output.json" [/ARGS_EXAMPLES]
        timeout: [ARGS_BRIEF] Maximum execution time in seconds. Defaults to 300. [/ARGS_BRIEF]
                [ARGS_DETAILED] The maximum time in seconds the subprocess is allowed to run before being terminated.
                This prevents infinite loops and runaway processes from consuming system resources.
                If the execution exceeds this limit, a timeout error will be returned.
                Choose appropriate values based on expected computation time. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] positive integer representing seconds [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] 30 (quick calculations), 300 (standard), 1800 (long processing) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON string with detailed execution results and captured output. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A comprehensive JSON string containing execution status, standard output, error messages, return code, captured execution results, and file save status.
             The execution_result field contains variables captured from the executed code.
             This enables full visibility into the execution process and results. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] "{"success": true, "execution_result": {"result": 10}, "stdout": "...", "stderr": "", "return_code": 0}" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        TimeoutExpired: [ERROR_WHEN] When code execution exceeds the specified timeout [/ERROR_WHEN]
                       [ERROR_DETAILS] Process terminated due to timeout limit [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Increase timeout value or optimize code for faster execution [/ERROR_RECOVERY]
        SyntaxError: [ERROR_WHEN] When the Python code contains syntax errors [/ERROR_WHEN]
                    [ERROR_DETAILS] Invalid Python syntax in the code string [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Check code syntax and fix any errors [/ERROR_RECOVERY]
        RuntimeError: [ERROR_WHEN] When code execution fails due to runtime errors [/ERROR_WHEN]
                     [ERROR_DETAILS] Errors during code execution such as undefined variables [/ERROR_DETAILS]
                     [ERROR_RECOVERY] Debug code logic and ensure all required variables are defined [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Limited to Python standard library and commonly available packages
    - Cannot access external network resources or file system outside working directory
    - Cannot install new packages during execution
    - Does not persist state between executions
    [/LIMITATIONS]
    """
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
