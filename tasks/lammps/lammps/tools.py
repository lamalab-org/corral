from corral.base import Tool, ToolArgument
from corral.utils import tool
from typing import Dict
from dotenv import load_dotenv
import os
import modal
from loguru import logger
import subprocess

@tool
def run_lammps(input_file: str, log_file: str) -> str:
    """Run a LAMMPS simulation. 

    Args:
        input_file: Path to the LAMMPS input script file.
        log_file: Path where the log file output will be stored.
    """
    try:
        run_lammps_sim = modal.Function.lookup("simagent", "run_lammps")
        run_lammps_sim.remote(input_file, log_file)
        return "simulation ran successfully!!"

    except ValueError as e:
        # Raise a ValueError with more context about the failure
        raise ValueError(f"The LAMMPS simulation failed with a ValueError: {str(e)}")
    except Exception as e:
        # Handle unexpected errors
        raise Exception(f"An unexpected error occurred while running the LAMMPS simulation: {str(e)}")

@tool
def plot_stress_strain(data_file: str, output_image: str) -> str:
    """
    Generate a stress-strain curve plot from a text file and save it as an image.
    Plots stress in x, y, and z directions.

    Args:
        data_file: Path to the text file containing strain and stress values.
        output_image: Path where the image will be saved (e.g., "plot.png").
    """
    try:
        plot_stress_strain_sim = modal.Function.lookup("simagent", "plot_stress_strain")
        plot_stress_strain_sim.remote(data_file, output_image)
        return "plotted_suceessfully!!!"

    except ValueError as e:
        # Raise a ValueError with more context about the failure
        raise ValueError(f"Plotting failed with a ValueError: {str(e)}")
    except Exception as e:
        # Handle unexpected errors
        raise Exception(f"An unexpected error occurred while plotting: {str(e)}")

def create_tools() -> Dict[str, Tool]:
    """Create all available tools"""
    return {
        "run_lammps": run_lammps,
    }
    