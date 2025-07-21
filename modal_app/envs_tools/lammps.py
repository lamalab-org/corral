from __future__ import annotations

from modal import Image

# cuda_version = "12.8.0"  # should be no greater than host CUDA version
# flavor = "devel"  #  includes full CUDA toolkit
# operating_sys = "ubuntu22.04"
# tag = f"{cuda_version}-{flavor}-{operating_sys}"

_lammps_image = (
    Image.debian_slim(python_version="3.12")
    # modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
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
    .pip_install("loguru", "fsspec", "numpy", "matplotlib", "pymatgen")
    .run_commands(
        "echo 'export LAMMPS_POTENTIALS=\"/potentials/EAM:/potentials/EAM_FS:/potentials/TERSOFF\"' >> /root/.bashrc"
    )
)
with _lammps_image.imports():
    import os
    import subprocess

    from loguru import logger


def _install_lammps():
    try:
        logger.debug("Cloning LAMMPS repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/lammps/lammps.git"], check=True
        )

        logger.debug("Making lammps/build directory")
        from pathlib import Path

        Path("lammps/build").mkdir(parents=True, exist_ok=True)

        logger.debug("Changing to lammps/build")
        os.chdir("lammps/build")

        logger.debug("Running CMAKE for MANY BODY PACKAGE")
        subprocess.check_call(
            "cmake ../cmake -D PKG_MANYBODY=on -D PKG_ATC=yes", shell=True
        )
        logger.debug("Running CMake with presets and GPU support...")
        subprocess.check_call(
            "cmake -C ../cmake/presets/most.cmake "
            "-C ../cmake/presets/nolib.cmake "
            "../cmake",
            shell=True,
        )
        # logger.debug("Running CMake with GPU and KOKKOS support...")
        # subprocess.check_call(
        #     "cmake -C ../cmake/presets/most.cmake "
        #     "-C ../cmake/presets/nolib.cmake "
        #     "-D PKG_GPU=on "
        #     "-D GPU_API=cuda "
        #     "../cmake",
        #     shell=True
        # )

        logger.debug("CMAKE build...")
        subprocess.check_call("cmake --build .", shell=True)

        logger.debug("MAKE for installing...")
        subprocess.check_call("make install", shell=True)

        logger.debug("LAMMPS installed successfully.")
        return "LAMMPS installed successfully"

    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed at step: {e.cmd}")
        raise Exception(f"Installation failed at step: {e.cmd}") from e


lammps_image = _lammps_image.run_function(_install_lammps)


def _run_lammps(
    input_file: str, log_file: str, directory_path: str | None = None, num_cpus: int = 1
) -> None:
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

    import os
    import subprocess
    from pathlib import Path

    lmp_command = "/root/lammps/build/lmp"
    original_cwd = Path.cwd()
    if directory_path:
        os.chdir(directory_path)
    try:
        # command = ["mpirun", "--allow-run-as-root", "-np", "8", lmp_command, "-in", input_file, "-log", log_file]
        # if use_cpus:
        logger.info("Running LAMMPS with multiple CPUs")
        # command = [lmp_command, "-sf", "gpu", "-pk", "gpu", "1", "-in", input_file, "-log", log_file]
        command = [
            "mpirun",
            "--allow-run-as-root",
            "-np",
            str(num_cpus),
            lmp_command,
            "-in",
            input_file,
            "-log",
            log_file,
        ]
        # else:
        #     logger.info("Running LAMMPS with single CPU")
        #     command = [lmp_command, "-in", input_file, "-log", log_file]
        # command = [lmp_command, "-in", input_file]
        subprocess.run(command, shell=False, check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        raise ValueError(f"LAMMPS simulation failed: {e.stderr or e}") from e

    finally:
        os.chdir(original_cwd)  # Restore original directory
