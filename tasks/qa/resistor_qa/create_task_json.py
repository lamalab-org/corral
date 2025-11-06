import json
import uuid
from pathlib import Path
from loguru import logger


def convert_mcq_to_task_format(questions):
    """Convert MCQ questions to task format and save each as separate file."""
    for i, question_data in enumerate(questions):
        # Ensure question_data is a dictionary
        if isinstance(question_data, list):
            # Assuming a list format like [question_text, options_dict]
            if (
                len(question_data) == 2
                and isinstance(question_data[0], str)
                and isinstance(question_data[1], dict)
            ):
                question_data = {
                    "question": question_data[0],
                    "options": question_data[1],
                }
            else:
                print(
                    f"Warning: Skipping malformed question data (list format) at index {i}: {question_data}"
                )
                continue
        elif not isinstance(question_data, dict):
            print(
                f"Warning: Skipping malformed question data (not dict or list) at index {i}: {question_data}"
            )
            continue

        question_text = question_data["question"]
        options = question_data["options"]

        # Format the input text with question and options
        options_text = ""
        option_keys = list(options.keys())

        for j, option_text in enumerate(option_keys, 1):
            options_text += f"{j}. {option_text}\n"

        input_text = f"{question_text}\n\nOptions:\n{options_text.strip()}"

        # Create target_scores dictionary mapping options to scores
        target_scores = dict(options.items())

        # Generate UUID for this specific question
        question_uuid = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"Resistor Environment QA {i + 1}")
        )

        # Create the task structure as a list with single task
        task_data = [
            {
                "name": f"Resistor Environment QA {i + 1}",
                "uuid": question_uuid,
                "description": f"Multiple choice question about electricity, resistors and current QA: {question_text[:100]}...",
                "examples": [{"input": input_text, "target_scores": target_scores}],
                "keywords": [
                    "electricity",
                    "resistors",
                    "resistor_networks",
                    "circuits",
                    "current",
                    "undergrad_physics",
                ],
                "metrics": ["multiple_choice_grade"],
                "preferred_score": "multiple_choice_grade",
            }
        ]

        # Save each question as separate JSON file
        filename = f"resistor_question_{i + 1:03d}.json"
        with Path(filename).open("w") as f:
            json.dump(task_data, f, indent=2)


def main():
    # Read the MCQ JSON file
    try:
        with Path("questions_resistor.json").open() as f:
            mcq_data = json.load(f)
    except FileNotFoundError:
        return

    # Handle both list and dict formats
    if isinstance(mcq_data, list):
        questions = mcq_data
    else:
        # If it's a dict, get the first key's value
        questions = next(iter(mcq_data.values()))

    # Convert to task format and save separate files
    convert_mcq_to_task_format(questions)


if __name__ == "__main__":
    main()
