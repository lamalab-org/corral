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
    import subprocess

    from loguru import logger


def _run_quantum_espresso(pw_command: str, options: list[str], input: str) -> str:
    output_file = "output.out"
    input_file = "input.in"

    try:
        with open(input_file, "w") as f:
            f.write(input)
        logger.debug(f"Input written to file: {input_file}")
    except OSError as e:
        logger.error(f"Failed to write input file: {e}")
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
        logger.debug(f"Reading output from file: {output_file}")
        with open(output_file) as f:
            content = f.read()
            logger.debug(f"Read {len(content)} characters from output file")
            return content

    except subprocess.CalledProcessError as e:
        logger.error(f"Quantum Espresso execution failed: {e.cmd}")
        raise Exception(f"Quantum Espresso execution failed: {e.cmd}\n{e.stderr}")
