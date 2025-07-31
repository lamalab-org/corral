import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from rubrics import RUBRICS_v1 as RUBRICS

USER_TAGS = ["NA", "MRG", "KMJ", "NMA", "CG", "SJ"]


def check_repeated_agent_messages(messages):
    """
    Returns a set of message indices where the agent repeated the same message
    two or more consecutive times.
    """
    repeated_indices = set()
    last_content = None
    last_idx = None
    repeat_count = 1
    for i, message in enumerate(messages):
        if identify_message_type(message) == "agent_action":
            content = message.get("content", "")
            if content == last_content:
                repeat_count += 1
                if repeat_count >= 2:
                    repeated_indices.add(i)
                    if last_idx is not None:
                        repeated_indices.add(last_idx)
            else:
                repeat_count = 1
            last_content = content
            last_idx = i
        else:
            last_content = None
            last_idx = None
            repeat_count = 1
    return repeated_indices


def premark_correctness_from_trials(agent_logs_dir, trials_correctness):
    """
    For each task_id in trials_correctness, find agent log files in agent_logs_dir matching {task_id}_*.json,
    sort by timestamp, and if the corresponding trial's success is True, pre-mark 'correctness.final_answer' as 'Yes'.
    """
    agent_logs_dir = Path(agent_logs_dir)
    # Always initialize premarked_annotations so it exists in session state
    if "premarked_annotations" not in st.session_state:
        st.session_state.premarked_annotations = {}
    for task_id, trial_list in trials_correctness.items():
        # Find all agent log files for this task_id
        files = sorted(agent_logs_dir.glob(f"{task_id}_*.json"))
        # Discard files containing _ANNOTATED or _INPROGRESS in their names
        files = [
            f
            for f in files
            if "_ANNOTATED" not in f.name and "_INPROGRESS" not in f.name
        ]

        # Sort files by timestamp in filename
        def extract_ts(f, task_id=task_id):
            m = re.search(rf"{re.escape(task_id)}_(\d{{8}}_\d{{6}})\\.json$", str(f))
            return m.group(1) if m else ""

        files = sorted(files, key=extract_ts)
        # For each trial, if success, pre-mark
        for i, trial in enumerate(trial_list):
            trial_list_item = list(trial)
            trial_id, success = trial_list_item
            if i < len(files):
                log_path = files[i]
                log_filename = log_path.name
                if success is True or success == "True":
                    # Find the rubric key for correctness.final_answer
                    task_rubrics = flatten_rubrics(RUBRICS["task_rubrics"])
                    for key, _rubric in task_rubrics:
                        if key == "correctness.final_answer":
                            if "premarked_annotations" not in st.session_state:
                                st.session_state.premarked_annotations = {}
                            st.session_state.premarked_annotations[log_filename] = {
                                key: "Yes"
                            }
                            break


def get_already_annotated_files(output_directory):
    """Get list of already annotated files from output directory."""
    annotated_files = set()
    if Path(output_directory).exists():
        for filename in os.listdir(output_directory):
            if filename.endswith("_ANNOTATED.json"):
                # Extract original filename by removing user tag and _ANNOTATED.json
                parts = filename.split("_")
                if len(parts) >= 3:  # filename_TAG_ANNOTATED.json
                    original_name = "_".join(parts[:-2]) + ".json"
                    annotated_files.add(original_name)
    return annotated_files


def load_json_files(input_directory, output_directory):
    """Load all JSON files from the input directory, excluding already annotated and in-progress ones."""
    json_files = []
    if Path(input_directory).exists():
        # Get all JSON files from input directory
        all_files = [
            filename
            for filename in os.listdir(input_directory)
            if filename.endswith(".json")
            and not filename.endswith("_ANNOTATED.json")
            and not filename.endswith("_INPROGRESS.json")
        ]

        # Get already annotated files from output directory
        annotated_files = get_already_annotated_files(output_directory)

        # Filter out already annotated files
        json_files = [f for f in all_files if f not in annotated_files]

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
    return [
        (i, message)
        for i, message in enumerate(messages)
        if identify_message_type(message) == "agent_action"
    ]


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
    """Display a single rubric item with radio buttons and comment field."""
    question = rubric["question"]
    description = rubric["description"]

    # Display rubric question and radio buttons in a single row format
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

    # Radio buttons for Yes/NA selection
    checkbox_result = st.radio(
        "Select response:",
        options=["Yes", "NA"],
        key=f"{prefix}{key}_radio",
        horizontal=True,
        index=None,  # No default selection
    )

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


def save_intermediate_progress(output_directory, original_filename, log_data, user_tag):
    """Save intermediate annotation progress."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = original_filename.replace(".json", "")
    progress_filename = f"{base_name}_{user_tag}_INPROGRESS.json"
    filepath = output_path / progress_filename

    try:
        # Create annotated data with current progress
        annotated_data = copy.deepcopy(log_data)

        # Add current annotations
        annotated_data["task_annotations"] = st.session_state.get(
            "task_annotations", {}
        )
        annotated_data["task_comments"] = st.session_state.get("task_comments", {})

        # Add step-wise annotations and comments to agent action messages
        messages_length = len(annotated_data.get("messages", []))
        for msg_idx, annotations in st.session_state.get(
            "step_annotations", {}
        ).items():
            if 0 <= int(msg_idx) < messages_length:
                for ann_key, ann_value in annotations.items():
                    annotated_data["messages"][msg_idx][ann_key] = ann_value

        for msg_idx, comments in st.session_state.get("step_comments", {}).items():
            if 0 <= int(msg_idx) < messages_length:
                for comment_key, comment_value in comments.items():
                    annotated_data["messages"][msg_idx][f"{comment_key}_comment"] = (
                        comment_value
                    )

        # Add progress metadata
        annotated_data["progress_metadata"] = {
            "user_tag": user_tag,
            "current_step": st.session_state.get("current_step", 0),
            "annotation_phase": st.session_state.get("annotation_phase", "stepwise"),
            "last_saved": datetime.now(tz=timezone.utc).isoformat(),
            "original_file": original_filename,
        }

        with filepath.open("w") as f:
            json.dump(annotated_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving progress: {e!s}")
        return False


def load_intermediate_progress(output_directory, original_filename, user_tag):
    """Load intermediate annotation progress if exists."""
    output_path = Path(output_directory)
    base_name = original_filename.replace(".json", "")
    progress_filename = f"{base_name}_{user_tag}_INPROGRESS.json"
    filepath = output_path / progress_filename

    if not filepath.exists():
        return False

    try:
        with filepath.open() as f:
            data = json.load(f)

        # Validate rubrics to ensure loaded keys are still valid
        valid_rubric_keys = {k for k, _ in flatten_rubrics(RUBRICS["task_rubrics"])}

        # Load progress metadata
        if "progress_metadata" in data:
            metadata = data["progress_metadata"]
            st.session_state.current_step = min(
                int(metadata.get("current_step", 0)),
                len(extract_agent_actions(data.get("messages", []))) - 1
                if data.get("messages")
                else 0,
            )
            st.session_state.annotation_phase = metadata.get(
                "annotation_phase", "stepwise"
            )

        # Load task annotations, filtering out invalid keys
        if "task_annotations" in data:
            st.session_state.task_annotations = {
                k: v
                for k, v in data["task_annotations"].items()
                if k in valid_rubric_keys
            }
        if "task_comments" in data:
            st.session_state.task_comments = {
                k: v for k, v in data["task_comments"].items() if k in valid_rubric_keys
            }

        # Load step annotations from messages
        messages = data.get("messages", [])
        for i, message in enumerate(messages):
            # Find annotation keys (not comments)
            ann_keys = [
                k
                for k in message
                if not k.endswith("_comment")
                and k not in ["role", "content", "tool_calls"]
                and k in valid_rubric_keys
            ]
            comment_keys = [k for k in message if k.endswith("_comment")]

            if ann_keys or comment_keys:
                if i not in st.session_state.step_annotations:
                    st.session_state.step_annotations[i] = {}
                if i not in st.session_state.step_comments:
                    st.session_state.step_comments[i] = {}

                for key in ann_keys:
                    st.session_state.step_annotations[i][key] = message[key]

                for key in comment_keys:
                    original_key = key.replace("_comment", "")
                    if original_key in valid_rubric_keys:
                        st.session_state.step_comments[i][original_key] = message[key]

        return True
    except Exception as e:
        st.error(f"Error loading progress from {filepath}: {e!s}")
        return False


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
    if "json_files" not in st.session_state:
        st.session_state.json_files = []
    if "results_file_path" not in st.session_state:
        st.session_state.results_file_path = ""
    if "trials_correctness" not in st.session_state:
        st.session_state.trials_correctness = {}


# --- Results file loader and extractor ---
def extract_trials_correctness(results_file_path):
    """
    Extracts a dictionary mapping task_ids to a list of sets,
    each set containing (trial_id, success) from the results file.
    """
    import json
    from pathlib import Path

    if not results_file_path or not Path(results_file_path).exists():
        return {}
    with Path(results_file_path).open() as f:
        results = json.load(f)
    task_results = results.get("task_results", {})
    trials_correctness = {}
    for task_id, task_info in task_results.items():
        trials = task_info.get("trials", [])
        trial_list = []
        for trial in trials:
            trial_id = trial.get("trial_id")
            success = trial.get("success")
            trial_list.append((trial_id, success))  # Use tuple to preserve order
        trials_correctness[task_id] = trial_list
    return trials_correctness


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

    # Enable code wrapping for all code blocks using CSS workaround
    st.markdown(
        """
        <style>
        .streamlit-expanderContent pre, .streamlit-expanderContent code, pre, code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    # Prompt for results file after input/output directories
    results_file_path = st.text_input(
        "Results file (metrics/results.json):",
        value=st.session_state.get("results_file_path", ""),
        key="results_file_input",
        help="Path to the results JSON file with metrics and task_results. Optional, but required for trial correctness extraction.",
    )
    if results_file_path:
        st.session_state.results_file_path = results_file_path
        st.session_state.trials_correctness = extract_trials_correctness(
            results_file_path
        )
        if st.session_state.trials_correctness:
            st.success(
                f"Loaded trial correctness for {len(st.session_state.trials_correctness)} tasks from results file."
            )
            # Try to premark agent logs if input_directory is available
            agent_logs_dir = Path(input_directory)
            premark_correctness_from_trials(
                agent_logs_dir, st.session_state.trials_correctness
            )
        else:
            st.info("No trial correctness data found or file missing.")

    # Validation and start button
    if input_directory and Path(input_directory).exists():
        st.session_state.input_directory = input_directory

        # Set output directory (default to input directory if blank)
        if output_directory.strip():
            st.session_state.output_directory = output_directory
        else:
            st.session_state.output_directory = input_directory

        # Load available files (excluding already annotated ones)
        json_files = load_json_files(input_directory, st.session_state.output_directory)
        st.session_state.json_files = json_files  # Cache file list in session state

        if not json_files:
            st.warning(
                "No JSON files found in the specified directory or all files have been annotated."
            )
            # Show information about annotated files
            annotated_files = get_already_annotated_files(
                st.session_state.output_directory
            )
            if annotated_files:
                st.info(
                    f"Found {len(annotated_files)} already annotated files in output directory"
                )
            return

        # Show file count and start button
        st.info(f"Found {len(json_files)} JSON files to annotate")

        if not st.session_state.annotation_started:
            if st.button("🚀 Start Annotation", type="primary"):
                st.session_state.annotation_started = True
                st.session_state.selected_file_index = 0  # Reset to first file
                st.rerun()
            return

        # Main annotation interface (only show after start button is pressed)
        st.divider()
        st.subheader("📝 Annotation Interface")

        # Ensure selected_file_index is within bounds
        if st.session_state.selected_file_index >= len(json_files):
            st.session_state.selected_file_index = 0
            st.session_state.json_files = load_json_files(
                input_directory, st.session_state.output_directory
            )
            json_files = st.session_state.json_files
            if not json_files:
                st.info("🎉 All files have been annotated!")
                st.session_state.annotation_started = False
                st.rerun()
                return

        # Check if we need to refresh the file list
        if st.button("🔄 Refresh File List"):
            st.session_state.json_files = load_json_files(
                input_directory, st.session_state.output_directory
            )
            json_files = st.session_state.json_files
            st.session_state.selected_file_index = min(
                st.session_state.selected_file_index, len(json_files) - 1
            )
            reset_annotation_state()
            st.rerun()

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

        # Reset state if file changed manually
        if selected_file_index != st.session_state.selected_file_index:
            st.session_state.selected_file_index = selected_file_index
            reset_annotation_state()

        selected_file = json_files[selected_file_index]

        # User tag selection
        user_tag = st.selectbox("Select user tag:", USER_TAGS)

        # Load the selected file
        log_data = load_log_file(input_directory, selected_file)

        if log_data:
            # Load intermediate progress
            if load_intermediate_progress(
                st.session_state.output_directory, selected_file, user_tag
            ):
                st.info("📄 Loaded previous annotation progress")
                # Validate current_step against agent_actions
                messages = log_data.get("messages", [])
                agent_actions = extract_agent_actions(messages)
                if st.session_state.current_step >= len(agent_actions):
                    st.session_state.current_step = max(0, len(agent_actions) - 1)
                    save_intermediate_progress(
                        st.session_state.output_directory,
                        selected_file,
                        log_data,
                        user_tag,
                    )
                    st.rerun()

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
                        st.markdown(
                            f"""```text\n{content}\n```""", unsafe_allow_html=True
                        )

                    if i < len(messages) - 1:
                        st.divider()

            # Extract agent actions
            # Only consider messages with role == 'assistant' for repeated message checks
            assistant_messages = [
                m for m in log_data.get("messages", []) if m.get("role") == "assistant"
            ]
            agent_actions = extract_agent_actions(assistant_messages)

            # --- Auto premark insanity.repeated_message rubric if agent repeated same message ---
            repeated_indices = check_repeated_agent_messages(assistant_messages)
            for idx in repeated_indices:
                if idx not in st.session_state.step_annotations:
                    st.session_state.step_annotations[idx] = {}
                st.session_state.step_annotations[idx]["insanity.repeated_message"] = (
                    "Yes"
                )

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

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Annotation Phases")
                st.write(
                    "You can annotate agent actions in two phases: step-wise and task-wise."
                )
                st.write(
                    "1. **Step-wise Annotation**: Annotate each agent action step by step."
                )
                st.write(
                    "2. **Task-wise Annotation**: Provide overall annotations for the task."
                )
                st.write(
                    "You can check Yes if the rubriks are met, if not it is counted as No. Select NA if not applicable (please maybe provide a comment here)."
                )

                with st.expander("TASK & SYSTEM PROMPT", expanded=True):
                    messages = log_data.get("messages", [])
                    for i, message in enumerate(messages):
                        header, content = format_message_for_display(message, i)

                        # Color-code different message types
                        msg_type = identify_message_type(message)
                        if msg_type == "system":
                            st.info(f"**{header}**")
                            if content:
                                st.code(content, language="text")
                            if i < len(messages) - 1:
                                st.divider()
                        elif msg_type == "task":
                            st.warning(f"**{header}**")
                            if content:
                                st.code(content, language="text")
                            if i < len(messages) - 1:
                                st.divider()

            with col2:
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
                            if (
                                checkbox_result is not None
                            ):  # Only update if a selection is made
                                st.session_state.step_annotations[msg_idx][key] = (
                                    checkbox_result
                                )
                                st.session_state.step_comments[msg_idx][key] = (
                                    comment_result
                                )
                                # Save progress after each rubric input
                                save_intermediate_progress(
                                    st.session_state.output_directory,
                                    selected_file,
                                    log_data,
                                    user_tag,
                                )
                            st.divider()

                    # Navigation buttons
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        if st.button(
                            "← Previous", disabled=st.session_state.current_step == 0
                        ):
                            st.session_state.current_step = max(
                                0, st.session_state.current_step - 1
                            )
                            save_intermediate_progress(
                                st.session_state.output_directory,
                                selected_file,
                                log_data,
                                user_tag,
                            )
                            st.rerun()

                    with col2:
                        if st.session_state.current_step < len(agent_actions) - 1:
                            if st.button("Next →"):
                                st.session_state.current_step += 1
                                save_intermediate_progress(
                                    st.session_state.output_directory,
                                    selected_file,
                                    log_data,
                                    user_tag,
                                )
                                st.rerun()
                        else:
                            if st.button(
                                "Proceed to Task-Level Rubrics →", type="primary"
                            ):
                                st.session_state.annotation_phase = "taskwise"
                                save_intermediate_progress(
                                    st.session_state.output_directory,
                                    selected_file,
                                    log_data,
                                    user_tag,
                                )
                                st.rerun()

                    with col3:
                        st.write(
                            f"Step {st.session_state.current_step + 1}/{len(agent_actions)}"
                        )

                # Task-wise annotation phase
                elif st.session_state.annotation_phase == "taskwise":
                    st.header("📝 Task-Level Rubrics")

                    # Display task rubrics
                    premarked = {}
                    # Use only the filename for premarked lookup
                    log_filename = Path(selected_file).name
                    if (
                        "premarked_annotations" in st.session_state
                        and log_filename in st.session_state.premarked_annotations
                    ):
                        premarked = st.session_state.premarked_annotations[log_filename]
                    # Precheck insanity.repeated_message if any step annotation has it set to Yes
                    repeated_prechecked = False
                    for step_ann in st.session_state.step_annotations.values():
                        if step_ann.get("insanity.repeated_message") == "Yes":
                            repeated_prechecked = True
                            break
                    for key, rubric in task_rubrics:
                        default = premarked.get(key)
                        radio_key = f"task_{key}_radio"
                        # Always set the default for correctness.final_answer if premarked and not set by user
                        if key == "correctness.final_answer":
                            if default is not None:
                                st.session_state[radio_key] = default
                            elif radio_key not in st.session_state:
                                st.session_state[radio_key] = None
                        # Precheck insanity.repeated_message if any step annotation has it set to Yes
                        elif key == "insanity.repeated_message" and repeated_prechecked:
                            st.session_state[radio_key] = "Yes"
                        # For other keys, only set if not already set
                        elif default is not None and radio_key not in st.session_state:
                            st.session_state[radio_key] = default

                        checkbox_result, comment_result = display_rubric_item(
                            key, rubric, "task_"
                        )
                        # If not set by user, use premarked value for correctness.final_answer
                        if key == "correctness.final_answer":
                            if checkbox_result is None and default is not None:
                                st.session_state.task_annotations[key] = default
                            else:
                                st.session_state.task_annotations[key] = checkbox_result
                        elif key == "insanity.repeated_message" and repeated_prechecked:
                            st.session_state.task_annotations[key] = "Yes"
                        else:
                            st.session_state.task_annotations[key] = checkbox_result
                        st.session_state.task_comments[key] = comment_result
                        # Save progress after each task rubric input
                        save_intermediate_progress(
                            st.session_state.output_directory,
                            selected_file,
                            log_data,
                            user_tag,
                        )
                        st.divider()

                    # Action buttons
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        if st.button("← Back to Step-wise", type="secondary"):
                            st.session_state.annotation_phase = "stepwise"
                            save_intermediate_progress(
                                st.session_state.output_directory,
                                selected_file,
                                log_data,
                                user_tag,
                            )
                            st.rerun()

                    with col2:
                        save_clicked = st.button("💾 Save Annotations", type="primary")

                    if save_clicked:
                        # Create annotated data
                        annotated_data = copy.deepcopy(log_data)

                        # Add task-level annotations
                        annotated_data["task_annotations"] = (
                            st.session_state.task_annotations
                        )
                        annotated_data["task_comments"] = st.session_state.task_comments

                        # Add step-wise annotations to agent action messages
                        messages_length = len(annotated_data.get("messages", []))
                        for (
                            msg_idx,
                            annotations,
                        ) in st.session_state.step_annotations.items():
                            if 0 <= msg_idx < messages_length:
                                for ann_key, ann_value in annotations.items():
                                    annotated_data["messages"][msg_idx][ann_key] = (
                                        ann_value
                                    )
                            else:
                                st.warning(f"Skipping invalid message index: {msg_idx}")

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
                            # Delete the in-progress file since annotation is complete
                            base_name = selected_file.replace(".json", "")
                            progress_filename = (
                                f"{base_name}_{user_tag}_INPROGRESS.json"
                            )
                            progress_filepath = (
                                Path(st.session_state.output_directory)
                                / progress_filename
                            )
                            if progress_filepath.exists():
                                try:
                                    progress_filepath.unlink()
                                except Exception as e:
                                    st.warning(f"Could not delete progress file: {e!s}")

                            # Refresh file list to exclude the newly annotated file
                            st.session_state.json_files = load_json_files(
                                input_directory, st.session_state.output_directory
                            )
                            json_files = st.session_state.json_files

                            # Move to next file if available
                            if st.session_state.selected_file_index < len(json_files):
                                st.session_state.selected_file_index += 1
                            else:
                                st.session_state.selected_file_index = 0
                                if not json_files:
                                    st.info("🎉 All files have been annotated!")
                                    st.session_state.annotation_started = False

                            reset_annotation_state()
                            st.rerun()

    elif input_directory:
        st.error("❌ Input directory not found. Please check the path.")


if __name__ == "__main__":
    main()
