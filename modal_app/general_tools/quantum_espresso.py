from __future__ import annotations

from modal import Image

_quantum_espresso_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "g++",
        "gfortran",
        "liblapack-dev",
        "libfftw3-dev",
        "libopenmpi-dev",
    )
    .pip_install("loguru")
)
with _quantum_espresso_image.imports():
    import os
    import subprocess

    from loguru import logger


def _install_quantum_espresso():
    try:
        logger.debug("Cloning Quantum Espresso repository...")
        subprocess.run(["git", "clone", "https://github.com/QEF/q-e.git"], check=True)

        logger.debug("Changing to q-e directory...")
        os.chdir("q-e")

        logger.debug("Running configure script...")
        subprocess.run(["./configure"], check=True)

        logger.debug("Compiling Quantum Espresso...")
        subprocess.run(["make", "all"], check=True)

        logger.debug("Adding Quantum Espresso to PATH...")
        home = os.path.expanduser("~")
        bashrc_path = os.path.join(home, ".bashrc")
        with open(bashrc_path, "a") as f:
            f.write("\nexport PATH=$PATH:$HOME/q-e/bin\n")

        logger.debug("Updating shell environment...")
        subprocess.run(f"bash -c 'source {bashrc_path}'", shell=True, check=True)

        logger.debug("Quantum Espresso installation completed successfully")
        return "Quantum Espresso installed successfully"

    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed at step: {e.cmd}")
        raise Exception(f"Installation failed at step: {e.cmd}")


quantum_espresso_image = _quantum_espresso_image.run_function(_install_quantum_espresso)


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
