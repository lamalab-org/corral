from __future__ import annotations

from modal import Image

_quantum_espresso_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("libfftw3-dev", "quantum-espresso")
    .pip_install("loguru")
    .run_commands(
        """echo 'export PATH="/root/q-e-qe-7.2/bin:$PATH"' >> ~/.bashrc && /bin/bash -c 'source ~/.bashrc'"""
    )
)
with _quantum_espresso_image.imports():
    import os
    import subprocess
    import uuid

    from loguru import logger

import mimetypes


def _run_quantum_espresso(
    pw_command: str,
    options: list[str],
    input: str,
    input_file: str = "input.in",
    output_file: str = "output.out",
) -> dict[str, str]:
    """
    Run a Quantum ESPRESSO command with the given options and input.

    Args:
        pw_command: The Quantum ESPRESSO command to run.
        options: The options to pass to the command.
        input: The input to pass to the command.
        input_file: The name of the input file to write.
        output_file: The name of the output file to write.

    Returns:
        Dictionary containing contents of all text files in the run directory.
    """
    run_id = str(uuid.uuid4())
    run_dir = run_id
    input_file = os.path.join(run_dir, input_file)
    output_file = os.path.join(run_dir, output_file)

    try:
        os.makedirs(run_dir)
        with open(input_file, "w") as f:
            f.write(input)
        logger.debug(f"Input written to file: {input_file}")
    except OSError as e:
        logger.error(f"Failed to create directory or write input file: {e}")
        raise e

    try:
        command = [pw_command]

        for opt in options:
            if not opt.startswith("-"):
                command.append(f"-{opt}")
            else:
                command.append(opt)

        command.extend(["-in", input_file])
        command.extend([">", output_file])
        shell = True

        logger.debug(f"Executing Quantum ESPRESSO command: {' '.join(command)}")
        subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)

        results = {}
        for filename in os.listdir(run_dir):
            filepath = os.path.join(run_dir, filename)
            if filename == "input.in":
                continue

            mime_type, _ = mimetypes.guess_type(filepath)
            if mime_type and mime_type.startswith("text/"):
                try:
                    with open(filepath) as f:
                        results[filename] = f.read()
                except Exception as e:
                    logger.error(f"Failed to read file {filename}: {e}")

        return results

    except subprocess.CalledProcessError as e:
        logger.error(f"Quantum Espresso execution failed: {e.cmd}")
        raise Exception(f"Quantum Espresso execution failed: {e.cmd}\n{e.stderr}")
    finally:
        try:
            import shutil

            shutil.rmtree(run_dir)
            logger.debug(f"Cleaned up directory: {run_dir}")
        except OSError as e:
            logger.warning(f"Failed to clean up directory {run_dir}: {e}")
