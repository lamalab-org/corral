import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from rubrics import RUBRICS_v1 as RUBRICS

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


def save_annotated_file(directory, original_filename, annotated_data, user_tag):
    """Idea is to save the annotated logs(in json file) with additional tags"""
    base_name = original_filename.replace(".json", "")
    new_filename = f"{base_name}_{user_tag}_ANNOTATED.json"
    filepath = Path(directory) / new_filename

    try:
        with filepath.open("w") as f:
            json.dump(annotated_data, f, indent=2)
        return new_filename
    except Exception as e:
        st.error(f"Error saving file: {e!s}")
        return None


def main():
    st.set_page_config(page_title="Agent Log Annotation", layout="wide")

    st.title("Agent Log Annotation Tool")

    # Directory selection
    if "directory" not in st.session_state:
        st.session_state.directory = ""

    directory = st.text_input(
        "Enter directory path containing JSON files:", value=st.session_state.directory
    )

    if directory and Path(directory).exists():
        st.session_state.directory = directory

        # Load available files
        json_files = load_json_files(directory)

        if not json_files:
            st.warning("No JSON files found in the specified directory.")
            return

        # File selection
        selected_file = st.selectbox("Select log file to annotate:", json_files)

        if selected_file:
            # User tag selection
            user_tag = st.selectbox("Select user tag:", USER_TAGS)

            # Load the selected file
            log_data = load_log_file(directory, selected_file)

            if log_data:
                # Display metadata
                st.subheader("Log Metadata")
                metadata = {k: v for k, v in log_data.items() if k != "messages"}
                st.json(metadata)

                # Show complete messages with expandable sections
                st.subheader("Complete Message History")
                messages = log_data.get("messages", [])

                with st.expander(
                    f"All Messages ({len(messages)} total)", expanded=False
                ):
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

                        # Show content in code block for better formatting
                        if content:
                            st.code(content, language="text")

                        if i < len(messages) - 1:
                            st.divider()

                # Extract agent actions for annotation
                agent_actions = extract_agent_actions(messages)

                # Might not be useful now becasuse we added all the agent log
                # st.subheader(
                #     f"Agent Actions Summary ({len(agent_actions)} actions found)",
                # )

                # # Show brief summary of agent actions
                # with st.expander("Agent Actions Overview", expanded=False):
                #     for i, (msg_idx, action) in enumerate(agent_actions):
                #         st.write(f"**Action {i+1} (Message {msg_idx})**")
                #         content = action.get("content", "")
                #         preview = (
                #             content[:200] + "..." if len(content) > 200 else content
                #         )
                #         st.write(preview)
                #         if i < len(agent_actions) - 1:
                #             st.write("---")

                st.divider()

                # Task-level rubrics
                st.header("Task-Wise Rubrics")
                task_annotations = {}
                task_comments = {}

                # Get task-level rubrics (step_wise = False)
                task_rubrics = flatten_rubrics(RUBRICS["task_rubrics"])
                task_rubrics = [
                    (k, v) for k, v in task_rubrics if not v.get("step_wise", False)
                ]

                for key, rubric in task_rubrics:
                    checkbox_result, comment_result = display_rubric_item(
                        key, rubric, "task_"
                    )
                    task_annotations[key] = checkbox_result
                    task_comments[key] = comment_result
                    st.divider()

                # Step-wise rubrics for each agent action
                st.header("Step-wise Rubrics")
                step_annotations = {}
                step_comments = {}

                # Get step-wise rubrics (step_wise = True)
                step_rubrics = flatten_rubrics(RUBRICS["task_rubrics"])
                step_rubrics = [
                    (k, v) for k, v in step_rubrics if v.get("step_wise", False)
                ]

                for i, (msg_idx, action) in enumerate(agent_actions):
                    st.subheader(f"Action {i+1} (Message {msg_idx})")

                    # Display the action content in an expandable section
                    with st.expander(f"View Action {i+1} Content", expanded=True):
                        st.code(action.get("content", ""))

                        # Show tool_calls if present (for ToolCallingAgent)
                        if "tool_calls" in action:
                            st.subheader("Tool Calls")
                            st.json(action["tool_calls"])

                    # Step-wise rubrics for this action
                    step_annotations[msg_idx] = {}
                    step_comments[msg_idx] = {}

                    for key, rubric in step_rubrics:
                        checkbox_result, comment_result = display_rubric_item(
                            key, rubric, f"step_{msg_idx}_"
                        )
                        step_annotations[msg_idx][key] = checkbox_result
                        step_comments[msg_idx][key] = comment_result

                    st.divider()

                # Save button
                if st.button("Save Annotations", type="primary"):
                    # Create annotated data
                    annotated_data = copy.deepcopy(log_data)

                    # Add task-level annotations
                    annotated_data["task_annotations"] = task_annotations
                    annotated_data["task_comments"] = task_comments

                    # Add step-wise annotations to agent action messages
                    for msg_idx, annotations in step_annotations.items():
                        # Find the message in the annotated data and add annotations
                        for ann_key, ann_value in annotations.items():
                            annotated_data["messages"][msg_idx][ann_key] = ann_value

                    # Add step-wise comments to agent action messages
                    for msg_idx, comments in step_comments.items():
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
                        directory, selected_file, annotated_data, user_tag
                    )

                    if saved_filename:
                        st.success(
                            f"Annotations saved successfully as: {saved_filename}"
                        )

    elif directory:
        st.error("Directory not found. Please check the path.")


if __name__ == "__main__":
    main()
