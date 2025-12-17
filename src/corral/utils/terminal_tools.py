import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from corral.backend.tool import tool

# Global registry for terminal sessions
_terminal_sessions: dict[str, dict[str, Any]] = {}
_session_lock = threading.Lock()

# Maximum output size to prevent memory issues (60KB as mentioned in reference)
MAX_OUTPUT_SIZE = 60 * 1024

# Maximum session history length to prevent excessive memory usage
MAX_HISTORY_LENGTH = 100  # Adjust as needed


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


@tool
def run_in_terminal(
    command: str,
    timeout: int | None = 300,
    session_id: str | None = None,
) -> str:
    """[BRIEF] Execute shell commands in a persistent terminal session with state preservation. For managing environments and dependencies uv is used. [/BRIEF]

    [DETAILED] This tool executes shell commands in a persistent terminal environment that maintains working directory, environment variables, and command history across multiple invocations.
    Commands are executed synchronously and block until completion, with comprehensive output capture and timeout protection.
    This is essential for running system commands, installing dependencies, and executing workflows that require shell access.
    The persistent session ensures that directory changes, environment modifications, and other session state persist across commands. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to execute shell/terminal commands (bash, zsh, etc.)
    - Best suited for system operations, and dependency installation
    - Essential for running command-line tools and utilities
    - Recommended for git operations, package installations, and system commands
    - Avoid for Python code execution (use execute_python_code instead)
    - Avoid for long-running processes that don't need to block (consider implementing background support if needed)
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Maintains persistent terminal session with preserved working directory
    - Preserves environment variables across command invocations
    - Executes commands using system shell (bash/zsh/cmd based on OS)
    - Captures stdout and stderr streams with automatic truncation
    - Implements timeout protection to prevent hanging processes
    - Tracks command history and execution results within session
    - Automatically truncates output if it exceeds 60KB to prevent memory issues
    - Blocks until command completes or timeout is reached
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Identify a command that needs to be run in the terminal such as installing a package that you want to use [/PREREQUISITE]
    2. [CURRENT] Execute command with appropriate timeout [/CURRENT]
    3. [FOLLOW_UP] Process the output or use returned session_id for subsequent commands [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    [
        `run_in_terminal("ls -la", 30)`,
        `run_in_terminal("npm install", 600)`,
        `run_in_terminal("python -m venv env", 60)`,
        `run_in_terminal("git status", 10)`,
        `run_in_terminal("pip install numpy pandas", 300, "existing-session-id")`,
    ]
    [/SYNTACTICAL]

    Args:
        command (str):
            [BRIEF] The shell command to execute. [/BRIEF]
            [DETAILED] A valid shell command string that will be executed in the terminal.
            The command should be compatible with the system's default shell (bash/zsh on Unix, cmd on Windows).
            Multi-line commands are not supported - use semicolons or && to chain commands.
            For commands that use pagers (like git log), disable paging with flags (e.g., 'git --no-pager log').
            Use absolute paths when possible to avoid ambiguity. [/DETAILED]
            [SYNTACTIC] "valid shell command string" [/SYNTACTIC]
            [EXAMPLES] "ls -la /home/user", "git clone https://github.com/repo.git", "npm install --save package-name" [/EXAMPLES]

        timeout (int):
            [BRIEF] Maximum execution time in seconds. Defaults to 300 (5 minutes). [/BRIEF]
            [DETAILED] The maximum time in seconds the command is allowed to run before being terminated.
            This prevents hanging processes and ensures resource management.
            Set to None to disable timeout (use with caution, only for trusted commands).
            For quick commands, use shorter timeouts (10-60s). For installations or builds, use longer timeouts (300-1800s). [/DETAILED]
            [SYNTACTIC] positive integer representing seconds, or None [/SYNTACTIC]
            [EXAMPLES] 30 (quick commands), 300 (default, moderate operations), 600 (builds/installations), None (no limit) [/EXAMPLES]

        session_id (str):
            [BRIEF] Optional session ID to use an existing terminal session. [/BRIEF]
            [DETAILED] The unique identifier of a terminal session to execute the command in.
            If provided, the command runs in the context of that session (preserving directory and environment).
            If None, a new session is created or the default session is used.
            This allows maintaining state across multiple command invocations.
            Useful for workflows that require sequential commands in the same context. [/DETAILED]
            [SYNTACTIC] UUID string or None [/SYNTACTIC]
            [EXAMPLES] "123e4567-e89b-12d3-a456-426614174000", None [/EXAMPLES]

    Returns:
        str:
            [BRIEF] JSON string with execution results including output, errors, and process information. [/BRIEF]
            [DETAILED] A JSON-formatted string containing the execution status, stdout, stderr, exit code, working directory, and session ID.
            The output is automatically truncated if it exceeds 60KB to prevent memory issues.
            Provides comprehensive information for debugging and workflow integration.
            The command blocks until completion or timeout. [/DETAILED]
            [EXAMPLES] '{"success": true, "stdout": "total 48\\ndrwxr-xr-x  12 user  staff  384 Oct 10 10:00 .", "stderr": "", "exit_code": 0, "cwd": "/home/user", "session_id": "abc123"}' [/EXAMPLES]

    [RAISES] Exceptions:
        TimeoutExpired:
            [ERROR_WHEN] When command execution exceeds the specified timeout [/ERROR_WHEN]
            [ERROR_DETAILS] Command was terminated due to timeout limit [/ERROR_DETAILS]
            [ERROR_RECOVERY] Increase timeout value or optimize command, or set timeout to None for no limit [/ERROR_RECOVERY]

        OSError:
            [ERROR_WHEN] When the command cannot be executed due to system errors [/ERROR_WHEN]
            [ERROR_DETAILS] System-level error preventing command execution [/ERROR_DETAILS]
            [ERROR_RECOVERY] Check command syntax, system resources, and permissions [/ERROR_RECOVERY]

        ValueError:
            [ERROR_WHEN] When invalid parameters are provided [/ERROR_WHEN]
            [ERROR_DETAILS] Command or parameters are malformed [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify command syntax and parameter values [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Does not support multi-line commands (use semicolons or && to chain)
    - Output is truncated after 60KB to prevent memory overflow
    - Interactive commands requiring user input may hang
    - Commands block until completion (no background execution support)
    - Session state is not persisted across tool restarts
    - Working directory changes only persist within the same session
    - Pager commands (like 'less', 'more') should be avoided or disabled
    [/LIMITATIONS]
    """
    logger.info(f"Executing terminal command: '{command}' (timeout={timeout})")

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
                if target_path.is_dir():
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
        # Enforce maximum history length
        if len(session["history"]) > MAX_HISTORY_LENGTH:
            session["history"] = session["history"][-MAX_HISTORY_LENGTH:]

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


@tool
def list_terminal_sessions() -> str:
    """[BRIEF] List all active terminal sessions and their information. [/BRIEF]

    [DETAILED] This tool provides an overview of all active terminal sessions, including session IDs, working directories, command history count, and creation times.
    This is useful for managing multiple terminal contexts and understanding the current state of terminal operations.
    Each session maintains its own working directory and environment, allowing for isolated command execution contexts. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use to see all available terminal sessions
    - Best suited for debugging session-related issues
    - Essential for understanding terminal session state
    - Recommended when managing multiple parallel workflows
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE]  You used the tool `run_in_terminal`, and you need to run it again within the same terminal such that the context is preserved. [/PREREQUISITE]
    2. [CURRENT] Retrieve list of active sessions and their status [/CURRENT]
    3. [FOLLOW_UP] Use session IDs with `run_in_terminal` to execute in specific contexts [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - Queries the internal session registry
    - Collects information about each active session
    - Returns formatted data including session IDs and metadata
    - Provides current working directory for each session
    - Shows command history count to understand session usage
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `list_terminal_sessions()`,
    ]
    [/SYNTACTICAL]

    Args:
        None

    Returns:
        str:
            [BRIEF] JSON string with list of all active terminal sessions. [/BRIEF]
            [DETAILED] A JSON-formatted string containing an array of session objects, each with session ID, current working directory, number of commands executed, and creation timestamp.
            Provides complete overview of terminal session state for management and debugging. [/DETAILED]
            [EXAMPLES] '{"sessions": [{"session_id": "abc123", "cwd": "/home/user/project", "command_count": 5, "created_at": 1696956000.0}]}' [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] For any unexpected errors during session listing [/ERROR_WHEN]
            [ERROR_DETAILS] An unexpected error occurred [/ERROR_DETAILS]
            [ERROR_RECOVERY] Review error message and stack trace for details [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Sessions are not persisted across tool restarts
    - Cannot retrieve sessions from other tool instances
    - Limited to sessions created by this tool
    [/LIMITATIONS]
    """
    logger.info("Listing all terminal sessions")

    try:
        with _session_lock:
            sessions_info = [
                {
                    "session_id": sess_id,
                    "cwd": session["cwd"],
                    "command_count": len(session["history"]),
                    "created_at": session["created_at"],
                }
                for sess_id, session in _terminal_sessions.items()
            ]

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
