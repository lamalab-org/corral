import json
from pathlib import Path

from loguru import logger


def combine_json_files(
    base_filename="task", max_index=8, output_filename="task_all_subtasks.json"
):
    """
    Combines multiple JSON files (e.g., task_0_subtasks.json, task_1_subtasks.json, ...)
    into a single JSON object.

    Args:
        base_filename (str): The base name of the files (e.g., 'task').
        max_index (int): The highest index number of the files to combine.
        output_filename (str): The name of the combined output file.
    """
    combined_data = {}

    for i in range(max_index + 1):
        filename = f"{base_filename}_{i}_subtasks.json"
        if not Path(filename).exists():
            logger.info(f"Warning: File not found: {filename}. Skipping.")
            continue

        try:
            with Path(filename).open("r") as f:
                # Load the data from the current file
                data = json.load(f)

                # Check if the data is a dictionary and contains the expected task structure
                if isinstance(data, dict):
                    # Append the contents of the current file's dictionary
                    # to the combined_data dictionary
                    combined_data.update(data)
                else:
                    # If the file's content is not a dictionary, it might need special handling.
                    # For this generic script, we'll assume it's a dictionary structure
                    # like the one you provided.
                    logger.info(
                        f"Warning: Content of {filename} is not a dictionary. Skipping update."
                    )

        except json.JSONDecodeError:
            logger.info(f"Error: Failed to decode JSON from {filename}. Skipping.")
        except Exception as e:
            logger.info(
                f"An unexpected error occurred while processing {filename}: {e}. Skipping."
            )

    # Write the combined dictionary to the output file
    try:
        with Path(output_filename).open("w") as outfile:
            # Use an indent for a nicely formatted output JSON file
            json.dump(combined_data, outfile, indent=4)
        logger.info(f"\nSuccessfully combined all tasks into {output_filename}")
    except Exception as e:
        logger.info(f"Error: Failed to write output file {output_filename}: {e}")


if __name__ == "__main__":
    # Assuming files are task_0_subtasks.json up to task_8_subtasks.json
    combine_json_files(max_index=8)
