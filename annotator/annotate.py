import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Rubrics definition (keeping same structure as provided)
RUBRICS = {
    "task_rubrics": {
        "correctness": {
            "question": "Do the agent logs lead to the correct answer?",
            "description": "If the answer returned by the agent is correct, then this is correct, or the box is checked. Otherwise, it is incorrect. This only checks the final answer, not the intermediate steps.",
            "examples": [
                "Answer returned by the agent: 'Final Answer: [B-](CCC1=CC=CC=C1)(F)(F)F.[K+]'\nCorrectness: Correct",
                "Answer returned by the agent: 'Final Answer: CCO=F'\nCorrectness: Incorrect",
            ],
            "automatic_check": True,
            "step_wise": False,
        },
        "insanity": {
            "repeated_message": {
                "question": "Did the agent avoid repeating the same exact message two or more iterations?",
                "description": "If the agent returned the same exact message two or more consecutive iterations. Note the negative character of the question. This is the checkbox that would stick unchecked if insanity is observed. Otherwise, we check the box if the agent keeps a normal and rational behavior.",
                "examples": [
                    "`iteration 3`: 'Now I need to provide the final answer with the correct format.' `iteration 4`: 'Now I need to provide the final answer with the correct format.' `iteration 5`: 'Now I need to provide the final answer with the correct format.': incorrect since the agent repeated the same message three times."
                ],
                "automatic_check": True,
                "step_wise": False,
            },
            "repeated_tokens": {
                "question": "Did the agent avoid repeating the same tokens until the max output tokens limit?",
                "description": "If for one message, the agent reproduced the same repeated tokens until the max output tokens limit. Note the negative character of the question. This is the checkbox that would stick unchecked if insanity is observed. Otherwise, we check the box if the agent keeps a normal and rational behavior.",
                "examples": [
                    "'The spectra suggest that the molecule could be the one with SMILES: COCCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC#CCOCC...': incorrect since the agent repeated the same tokens until the max output tokens limit."
                ],
                "automatic_check": False,
                "step_wise": False,
            },
        },
        "task_understanding": {
            "task_decomposition": {
                "question": "Is the task decomposed correctly?",
                "description": "If the agent decomposed the task correctly before proposing the respective steps, then this is correct, or the box is checked. Otherwise, it is incorrect.",
                "examples": [
                    '\'Based on the input data, let me try a structure: ethyl 4-((5-chloropyridin-2-yl)amino)benzoate\n\nLet me verify this with the formula tool first.\nAction: get_formula_from_smiles\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"\': incorrect since the agent did not decompose the task correctly. The task decomposition is to find the functional groups in the spectra, not to verify a structure.'
                ],
                "automatic_check": False,
                "step_wise": False,
            },
            "input_understanding": {
                "question": "Is there a mistake that could have been avoided by using knowledge provided in the context?",
                "description": "If the agent made a mistake that could have been avoided by using knowledge provided in the context, then this is incorrect, or the box is checked. Otherwise, it is correct.",
                "examples": [
                    "For example, in spectra, if some of the spectra are not used, resulting in the agent not finding the correct functional groups, then this is incorrect. If the agent uses all the spectra and finds the correct functional groups, then this is correct."
                ],
                "automatic_check": False,
                "step_wise": False,
            },
        },
        "tool_usage": {
            "tool_sampling": {
                "question": "Did the agent incur into sampling a tool with different inputs?",
                "description": "If the agent used the tool with different inputs (being the inputs always correct) two or more consecutive times, then this is correct, or the box is checked. Otherwise, it is incorrect.",
                "examples": [
                    "`iteration 2`: {'tool_name': 'simulate_spectra', 'arguments': {'smiles': 'CC1=C(C(=O)O)C=CC=C1C(=O)C'}\n`iteration 3`: {'tool_name': 'simulate_spectra', 'arguments': {'smiles': 'CC1=CC=CC=C1C(=O)OC(=O)C'}}\n`iteration 4`: {'tool_name': 'simulate_spectra', 'arguments': {'smiles': 'CC1=CC=C(C(=O)C)C(C(=O)O)=C1'}}: incorrect",
                    "`iteration 2`: {'tool_name': 'get_formula_from_smiles', 'arguments': {'smiles': 'C#CCC#CCCCCC'}\n`iteration 3`: {'tool_name': 'get_formula_from_smiles', 'arguments': {'smiles': 'C#CC#CCCCCCC'}}: incorrect",
                ],
                "automatic_check": True,
                "step_wise": False,
            },
            "tool_usage_reasoning": {
                "question": "Did the agent provide a reasoning for using a specific tool?",
                "description": "If the agent used the tool explaining the reasoning behind why that tool is used, then this is correct, or the box is checked. Otherwise, it is incorrect. Note that for this we do not care about if the reasoning is correct or not, just that the agent provided a reasoning.",
                "examples": [
                    '\'Action: get_formula_from_smiles\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"}\': incorrect since the agent did not explain why it is using the tool get_formula_from_smiles.',
                    '\'Let me verify the structure further by simulating the spectra to compare with the given data.\nAction: simulate_spectra\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"}\': correct since the agent explained why it is using the tool simulate_spectra.',
                ],
                "automatic_check": False,
                "step_wise": True,
            },
            "rational": {
                "logical_tool_usage": {
                    "question": "Is the step meaningful at this point in the trajectory and conductive toward the end goal? Hint: A step would be non-meaningful if the information obtained in there can be used in no way for the final solution.",
                    "description": "If the tool used by the agent is logical for that step, taking into account the end goal and the information known at that step, then this is correct or the box is checked. Otherwise, it is incorrect.",
                    "examples": [
                        '\'The final answer is the SMILES CC1=C(C(=O)O)C=CC=C1C(=O)C. I will know retrieve the carbon shifts and then I will provide the final answer.\nAction: get_carbon_shifts\nAction Input: {"smiles": "CC1=C(C(=O)O)C=CC=C1C(=O)C"}\': incorrect since the agent is retrieving some prior knowledge that is not needed to provide the final answer.',
                        '\'I will start by retrieving the carbon shifts of the molecule to better understand the spectra provided.\nAction: get_carbon_shifts\nAction Input: {"smiles": "CC1=C(C(=O)O)C=CC=C1C(=O)C"}\': correct since the agent is retrieving some prior knowledge that is needed to fully understand the input.',
                    ],
                    "automatic_check": False,
                    "step_wise": True,
                },
                "efficient_execution": {
                    "question": "Is the action performed in the most efficient way? For example, was an optimized tool used when it was available?",
                    "description": "If the agent executed the action in the most efficient way, then this is correct, or the box is checked. Otherwise, it is incorrect.",
                    "examples": [
                        '\'I need to join the different datasets: Action: read_file\nAction Input: {"file_path": "dataset1.csv"}... Action: write_file\nAction Input: {"file_path": "joined_dataset.csv", "data": "..."}\': incorrect since the agent is reading and writing files, which is not the most efficient way to join datasets.',
                        '\'I will write a python script to join the datasets: Action: write_file\nAction Input: {"file_path": "join_datasets.py", "data": "import pandas as pd...\': correct since the agent is using a python script to join the datasets, which is the most efficient way to do it.',
                    ],
                    "automatic_check": False,
                    "step_wise": True,
                },
            },
            "tool_calling_error": {
                "question": "Did the agent call the tool correctly or did it incur some error?",
                "description": "If the agent called the tool correctly, i.e., without incurring in argument format errors or an incorrect tool name, then this is correct or the box is checked. Otherwise, it is incorrect. Note that this only checks if the tool was called correctly, not if the tool returned the expected output.",
                "examples": [
                    '\'Action: get_formula_from_smiles\nAction Input: {"smiles": "[C#CCC#CCCCCC]"}\': incorrect since the agent passed an incorrect argument format to the tool get_formula_from_smiles list instead of string.',
                    '\'Action: wrong_tool_name\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"}\': incorrect since the agent used a wrong tool name.',
                    '\'Action: get_formula_from_smiles\nAction Input: {"smiles": "O=C(OCC)c1ccc(Nc2ccc(Cl)cn2)cc1"}\': correct since the agent passed the correct argument format to the tool get_formula_from_smiles string.',
                ],
                "automatic_check": False,
                "step_wise": True,
            },
        },
    }
}

USER_TAGS = ["NA", "MRG", "KMJ", "NMA", "CG", "SJ"]


def load_json_files(directory):
    """Load all JSON files from the specified directory."""
    json_files = []
    if Path(directory).exists():
        json_files.extend(
            [
                filename
                for filename in os.listdir(directory)
                if filename.endswith(".json")
                and not filename.endswith("_ANNOTATED.json")
            ]
        )
    return sorted(json_files)


def load_log_file(directory, filename):
    """Load a specific log file."""
    filepath = Path(directory) / filename
    try:
        with filepath.open() as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading file {filename}: {e!s}")
        return None


def identify_message_type(message):
    """Identify the type of message based on its structure."""
    if message.get("role") == "system":
        return "system"
    elif message.get("role") == "user" and "name" not in message:
        return "task"
    elif message.get("role") == "assistant":
        return "agent_action"
    elif message.get("role") == "tool" or (
        message.get("role") == "user" and "name" in message
    ):
        return "tool_call"
    else:
        return "unknown"


def extract_agent_actions(messages):
    """Extract agent action messages from the log."""
    agent_actions = []
    for i, message in enumerate(messages):
        if identify_message_type(message) == "agent_action":
            agent_actions.append((i, message))
    return agent_actions


def flatten_rubrics(rubrics, parent_key=""):
    """Flatten nested rubrics structure."""
    items = []
    for key, value in rubrics.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict) and "question" in value:
            items.append((new_key, value))
        elif isinstance(value, dict):
            items.extend(flatten_rubrics(value, new_key))
    return items


def display_rubric_item(key, rubric, prefix=""):
    """Display a single rubric item with checkbox and comment field."""
    question = rubric["question"]
    description = rubric["description"]

    # Create two columns for checkbox and comment
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"**{key}**")
        st.markdown(f"*{question}*")

        # Add collapsible description
        with st.expander("💡 Description"):
            st.write(description)

        # Show examples in expander
        if rubric.get("examples"):
            with st.expander("📖 Examples"):
                for example in rubric["examples"]:
                    st.code(example, language="text")

        checkbox_result = st.checkbox("Yes", key=f"{prefix}{key}_checkbox")

    with col2:
        st.markdown("**Comment**")
        st.write("")  # Add some spacing
        comment_result = st.text_area(
            "Add comment (optional)",
            key=f"{prefix}{key}_comment",
            height=100,
            placeholder="Add your notes here...",
        )

    return checkbox_result, comment_result


def format_message_for_display(message, index):
    """Format a message for display with type identification."""
    msg_type = identify_message_type(message)

    # Create a header with message type and index
    header = f"Message {index} ({msg_type.upper()})"

    # Get the main content
    content = message.get("content", "")

    # For tool calling agent messages, also show tool_calls if present
    if "tool_calls" in message:
        tool_calls_str = json.dumps(message["tool_calls"], indent=2)
        content = f"{content}\n\nTool Calls:\n{tool_calls_str}"

    # For tool responses, show the full response structure
    if msg_type == "tool_call":
        # Show additional fields for tool calls
        additional_fields = {
            k: v for k, v in message.items() if k not in ["content", "role"]
        }
        if additional_fields:
            additional_str = json.dumps(additional_fields, indent=2)
            content = f"{content}\n\nAdditional Fields:\n{additional_str}"

    return header, content


def save_annotated_file(output_directory, original_filename, annotated_data, user_tag):
    """Save annotated logs to specified output directory."""
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = original_filename.replace(".json", "")
    new_filename = f"{base_name}_{user_tag}_ANNOTATED.json"
    filepath = output_path / new_filename

    try:
        with filepath.open("w") as f:
            json.dump(annotated_data, f, indent=2)
        return new_filename
    except Exception as e:
        st.error(f"Error saving file: {e!s}")
        return None


def initialize_session_state():
    """Initialize session state variables."""
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "annotation_phase" not in st.session_state:
        st.session_state.annotation_phase = "stepwise"  # "stepwise" or "taskwise"
    if "step_annotations" not in st.session_state:
        st.session_state.step_annotations = {}
    if "step_comments" not in st.session_state:
        st.session_state.step_comments = {}
    if "task_annotations" not in st.session_state:
        st.session_state.task_annotations = {}
    if "task_comments" not in st.session_state:
        st.session_state.task_comments = {}
    if "selected_file_index" not in st.session_state:
        st.session_state.selected_file_index = 0
    if "annotation_started" not in st.session_state:
        st.session_state.annotation_started = False


def reset_annotation_state():
    """Reset annotation state for new file."""
    st.session_state.current_step = 0
    st.session_state.annotation_phase = "stepwise"
    st.session_state.step_annotations = {}
    st.session_state.step_comments = {}
    st.session_state.task_annotations = {}
    st.session_state.task_comments = {}


def main():
    st.set_page_config(page_title="Agent Log Annotation", layout="wide")

    initialize_session_state()

    st.title("Agent Log Annotation Tool")

    # Directory selection
    st.subheader("📁 Setup Directories")
    col1, col2 = st.columns(2)

    with col1:
        input_directory = st.text_input(
            "Input directory (JSON files):",
            value=st.session_state.get("input_directory", ""),
            key="input_directory_input",
        )

    with col2:
        output_directory = st.text_input(
            "Output directory (annotated files):",
            value=st.session_state.get("output_directory", ""),
            key="output_directory_input",
            help="Leave blank to save in same directory as input",
        )

    # Validation and start button
    if input_directory and Path(input_directory).exists():
        st.session_state.input_directory = input_directory

        # Set output directory (default to input directory if blank)
        if output_directory.strip():
            st.session_state.output_directory = output_directory
        else:
            st.session_state.output_directory = input_directory

        # Load available files
        json_files = load_json_files(input_directory)

        if not json_files:
            st.warning("No JSON files found in the specified directory.")
            return

        # Show file count and start button
        st.info(f"Found {len(json_files)} JSON files to annotate")

        if not st.session_state.annotation_started:
            if st.button("🚀 Start Annotation", type="primary"):
                st.session_state.annotation_started = True
                st.rerun()
            return

        # Main annotation interface (only show after start button is pressed)
        st.divider()
        st.subheader("📝 Annotation Interface")

        # File selection with progress indicator
        col1, col2 = st.columns([3, 1])

        with col1:
            selected_file_index = st.selectbox(
                "Current file:",
                range(len(json_files)),
                format_func=lambda x: json_files[x],
                index=st.session_state.selected_file_index,
                key="file_selector",
            )

        with col2:
            st.metric("Files Progress", f"{selected_file_index + 1}/{len(json_files)}")

        # Reset state if file changed
        if selected_file_index != st.session_state.selected_file_index:
            st.session_state.selected_file_index = selected_file_index
            reset_annotation_state()

        selected_file = json_files[selected_file_index]

        # User tag selection
        user_tag = st.selectbox("Select user tag:", USER_TAGS)

        # Load the selected file
        log_data = load_log_file(input_directory, selected_file)

        if log_data:
            # Display metadata
            st.subheader("📋 Log Metadata")
            metadata = {k: v for k, v in log_data.items() if k != "messages"}
            st.json(metadata)

            # Show complete messages with expandable sections (as reference)
            with st.expander("📖 Complete Message History (Reference)", expanded=False):
                messages = log_data.get("messages", [])
                for i, message in enumerate(messages):
                    header, content = format_message_for_display(message, i)

                    # Color-code different message types
                    msg_type = identify_message_type(message)
                    if msg_type == "system":
                        st.info(f"**{header}**")
                    elif msg_type == "task":
                        st.warning(f"**{header}**")
                    elif msg_type == "agent_action":
                        st.success(f"**{header}**")
                    elif msg_type == "tool_call":
                        st.error(f"**{header}**")
                    else:
                        st.write(f"**{header}**")

                    if content:
                        st.code(content, language="text")

                    if i < len(messages) - 1:
                        st.divider()

            # Extract agent actions
            agent_actions = extract_agent_actions(log_data.get("messages", []))

            if not agent_actions:
                st.warning("No agent actions found in this log.")
                return

            # Get rubrics
            step_rubrics = flatten_rubrics(RUBRICS["task_rubrics"])
            step_rubrics = [
                (k, v) for k, v in step_rubrics if v.get("step_wise", False)
            ]

            task_rubrics = flatten_rubrics(RUBRICS["task_rubrics"])
            task_rubrics = [
                (k, v) for k, v in task_rubrics if not v.get("step_wise", False)
            ]

            st.divider()

            # Step-wise annotation phase
            if st.session_state.annotation_phase == "stepwise":
                st.header("🔄 Step-wise Annotation")

                # Progress indicator
                progress_col1, progress_col2, progress_col3 = st.columns([1, 2, 1])
                with progress_col2:
                    st.progress(
                        (st.session_state.current_step + 1) / len(agent_actions)
                    )
                    st.write(
                        f"Step {st.session_state.current_step + 1} of {len(agent_actions)}"
                    )

                # Display current step
                if st.session_state.current_step < len(agent_actions):
                    msg_idx, action = agent_actions[st.session_state.current_step]

                    st.subheader(
                        f"Action {st.session_state.current_step + 1} (Message {msg_idx})"
                    )

                    # Show action content
                    st.code(action.get("content", ""))

                    # Show tool_calls if present
                    if "tool_calls" in action:
                        st.subheader("Tool Calls")
                        st.json(action["tool_calls"])

                    st.divider()

                    # Initialize step annotations if not exists
                    if msg_idx not in st.session_state.step_annotations:
                        st.session_state.step_annotations[msg_idx] = {}
                        st.session_state.step_comments[msg_idx] = {}

                    # Display step rubrics
                    for key, rubric in step_rubrics:
                        checkbox_result, comment_result = display_rubric_item(
                            key, rubric, f"step_{msg_idx}_"
                        )
                        st.session_state.step_annotations[msg_idx][key] = (
                            checkbox_result
                        )
                        st.session_state.step_comments[msg_idx][key] = comment_result
                        st.divider()

                # Navigation buttons
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    if st.button(
                        "← Previous", disabled=st.session_state.current_step == 0
                    ):
                        st.session_state.current_step -= 1
                        st.rerun()

                with col2:
                    if st.session_state.current_step < len(agent_actions) - 1:
                        if st.button("Next →"):
                            st.session_state.current_step += 1
                            st.rerun()
                    else:
                        if st.button("Proceed to Task-Level Rubrics →", type="primary"):
                            st.session_state.annotation_phase = "taskwise"
                            st.rerun()

                with col3:
                    st.write(
                        f"Step {st.session_state.current_step + 1}/{len(agent_actions)}"
                    )

            # Task-wise annotation phase
            elif st.session_state.annotation_phase == "taskwise":
                st.header("📝 Task-Level Rubrics")

                # Display task rubrics
                for key, rubric in task_rubrics:
                    checkbox_result, comment_result = display_rubric_item(
                        key, rubric, "task_"
                    )
                    st.session_state.task_annotations[key] = checkbox_result
                    st.session_state.task_comments[key] = comment_result
                    st.divider()

                # Action buttons
                col1, col2 = st.columns([1, 1])

                with col1:
                    if st.button("← Back to Step-wise", type="secondary"):
                        st.session_state.annotation_phase = "stepwise"
                        st.rerun()

                with col2:
                    if st.button("💾 Save Annotations", type="primary"):
                        # Create annotated data
                        annotated_data = copy.deepcopy(log_data)

                        # Add task-level annotations
                        annotated_data["task_annotations"] = (
                            st.session_state.task_annotations
                        )
                        annotated_data["task_comments"] = st.session_state.task_comments

                        # Add step-wise annotations to agent action messages
                        for (
                            msg_idx,
                            annotations,
                        ) in st.session_state.step_annotations.items():
                            for ann_key, ann_value in annotations.items():
                                annotated_data["messages"][msg_idx][ann_key] = ann_value

                        # Add step-wise comments to agent action messages
                        for msg_idx, comments in st.session_state.step_comments.items():
                            for comment_key, comment_value in comments.items():
                                annotated_data["messages"][msg_idx][
                                    f"{comment_key}_comment"
                                ] = comment_value

                        # Add annotation metadata
                        annotated_data["annotation_metadata"] = {
                            "user_tag": user_tag,
                            "annotation_timestamp": datetime.now(
                                tz=timezone.utc
                            ).isoformat(),
                            "original_file": selected_file,
                        }

                        # Save the file
                        saved_filename = save_annotated_file(
                            st.session_state.output_directory,
                            selected_file,
                            annotated_data,
                            user_tag,
                        )

                        if saved_filename:
                            st.success(f"✅ Annotations saved as: {saved_filename}")

                            # Show next file button if available
                            if selected_file_index < len(json_files) - 1:
                                st.divider()
                                if st.button(
                                    "➡️ Save and Go to Next File",
                                    type="primary",
                                    key="next_file_btn",
                                ):
                                    # Move to next file and reset everything
                                    st.session_state.selected_file_index += 1
                                    reset_annotation_state()
                                    # Force rerun to refresh the interface
                                    st.rerun()
                            else:
                                st.info("🎉 This was the last file in the directory!")

    elif input_directory:
        st.error("❌ Input directory not found. Please check the path.")


if __name__ == "__main__":
    main()
