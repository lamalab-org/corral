from __future__ import annotations

from modal import Image

_lammps_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "cmake",
        "wget",
        "build-essential",
        "liblapack-dev",
        "libfftw3-dev",
        "libopenmpi-dev",
        "openmpi-bin",
    )
    .pip_install("loguru", "fsspec", "numpy", "matplotlib")
)
with _lammps_image.imports():
    import os
    import subprocess

    from loguru import logger


def _install_lammps():

    try:
        logger.debug("Cloning LAMMPS repository...")    
        subprocess.run(["git", "clone", "https://github.com/lammps/lammps.git"], check=True)

        logger.debug("Making lammps/build directory")
        os.makedirs("lammps/build", exist_ok=True)

        logger.debug("Changing to lammps/build")
        os.chdir("lammps/build")
        
        logger.debug("Running CMAKE for MANY BODY PACKAGE")
        subprocess.check_call("cmake ../cmake -D PKG_MANYBODY=on -D PKG_ATC=yes", shell=True)
    
        logger.debug("CMAKE build...")
        subprocess.check_call("cmake --build .", shell=True)

        logger.debug("MAKE for installing...")
        subprocess.check_call("make install", shell=True)
    
        logger.debug("Adding LAMMPS to PATH...")
        home = os.path.expanduser("~")
        bashrc_path = os.path.join(home, ".bashrc")
        with open(bashrc_path, "a") as f:
            f.write('\nexport PATH="/root/lammps/build:$PATH"\n')

        logger.debug("Updating shell environment...")
        subprocess.run(f"bash -c 'source {bashrc_path}'", shell=True, check=True)

        logger.debug("LAMMPS installation completed successfully")
        return "LAMMPS installed successfully"
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed at step: {e.cmd}")
        raise Exception(f"Installation failed at step: {e.cmd}")

lammps_image = _lammps_image.run_function(_install_lammps)

def _run_lammps(input_file: str, log_file: str) -> None:
    """
    Runs a LAMMPS simulation using a specified input file and writes the log output to a given log file.

    Args:
        input_file (str): Path to the LAMMPS input script file.
        log_file (str): Path where the log file output will be stored.

    Returns:
        dict: A dictionary containing the log file content and input file content.

    Raises:
        ValueError: If the LAMMPS simulation fails.
    """
    import subprocess
    import os

    lmp_command = "/root/lammps/build/lmp"

    try:
        command = [lmp_command, "-in", input_file, "-log", log_file]
        # command = [lmp_command, "-in", input_file]
        subprocess.run(command, shell=False, check=True, capture_output=True, text=True)

        # # Read input file
        # if os.path.exists(input_file):
        #     with open(input_file, "r") as f:
        #         results["input_file"] = f.read()

        # # Read log file
        # if os.path.exists(log_file):
        #     with open(log_file, "r") as f:
        #         results["log_file"] = f.read()
        # else:
        #     raise FileNotFoundError(f"Log file '{log_file}' was not created.")

        # return results

    except subprocess.CalledProcessError as e:
        raise ValueError(f"LAMMPS simulation failed: {e.stderr or e}") from e