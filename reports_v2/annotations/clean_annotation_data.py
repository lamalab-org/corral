"""
Clean and process raw annotation data from corral_dump.json, validate planning statements using LLM,
and save cleaned data to cleaned_corral_annotations.parquet.
"""

import asyncio
import json
from collections import defaultdict
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()  # Load environment variables from .env file if present

lama_aesthetics.get_style("main")

file_path = Path(__file__).parent / "data" / "corral_dump.json"
output_file = Path(__file__).parent / "data" / "cleaned_corral_annotations.parquet"
json_example_file = Path(__file__).parent / "data" / "example_entry.json"

ENV_TAGS = {
    "afm": ["code execution", "validation"],
    "ml": ["retrieval", "code execution", "validation"],
    "catalyst": ["retrieval", "code execution"],
    "md": [
        "retrieval",
        "experiment execution",
    ],
    # "spectra": ["reasoning", "experiment execution", "validation"],
    "resistor": ["reasoning", "validation"],
    "retrosynthesis": ["reasoning", "validation"],
}

KEYS2NAMES = {
    "CG001": "Chandan",
    "MRG01": "Martino",
    "IM0005": "Indrajeet",
    "SJ9999": "Sajid",
    "AAA000003": "Sadra",
    "KMJ01": "Kevin",
    "NA0002": "Nawaf",
}

LABELS = {
    "Positive": [
        "validation_attempt",
        "backtrack_trigger",
        "planning_statement",
        "reasoning_statement",
        "correct_submission",
    ],
    "Neutral": ["neutral", "iteration_limit"],
    "Negative": [
        "missing_validation",
        "unnecessary_tool_use",
        "non_sense",
        "loop_instance",
        "hallucination",
        "wrong_planning",
        "wrong_reasoning",
        "syntax_error",
        "early_final_answer",
        "give_up",
        "inefficient_tool_call",
        "misunderstood_tool",
    ],
}

# All possible markers for counting
ALL_MARKERS = LABELS["Positive"] + LABELS["Neutral"] + LABELS["Negative"]


# Pydantic schema for planning statement validation
class PlanningStatementValidation(BaseModel):
    should_keep: bool
    reasoning: str | None = None


def check_annotatable(node: dict, message: str) -> bool:
    """Check if a file node is annotatable based on its type and content."""
    node_type = node.get("type", "")
    message = node.get("message", "")

    # Non-annotatable conditions
    if node_type in ["system", "tool"]:
        return False

    if node_type == "assistant":
        return True

    return not (node_type == "user" and message.startswith("Observation"))


async def validate_planning_statement(message: str, client: AsyncOpenAI) -> bool:
    """Use LLM to determine if a planning statement should be kept.

    Args:
        message: The message content to validate
        client: AsyncOpenAI client instance

    Returns:
        True if the planning statement should be kept, False otherwise
    """
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are evaluating whether a planning statement from an AI agent is meaningful and should be kept. A planning statement should be kept if it shows genuine multi-step decomposition (even if it is high-level). It should be removed if it's too generic, redundant, doesn't add value, simply states what the next action is going to be, or how to recover from previous action.",
                },
                {
                    "role": "user",
                    "content": f"""Examples of planning statements to keep:\n\n"I'll complete the CO2 adsorption on Cuprous Oxide slab workflow step by step. Let me start by retrieving the necessary structures.\n\n<thought>\nI need to:\n1. Get the Cuprous Oxide (Cu2O) bulk structure from Materials Project (mp-361)\n2. Get the CO2 molecule structure (mp-644607)\n3. Generate slabs from the bulk structure with miller_index [1,1,1]\n4. Select the slab at slab_index 0\n5. Identify adsorption sites on the slab\n6. Select the site with site_type "ontop" and site_index 0\n7. Add CO2 to the slab at height 2.0\n8. Save the final structure as a CIF file\n\nLet me start by getting the Cu2O bulk structure.\n</thought>\n\n<action>get_structure_from_mp_text</action>\n<action_input>{{"mp_id": "mp-361"}}</action_input>"\n\n"I'll help you calculate the surface energy of Aluminum for the [0,0,1] direction. Let me start by understanding the task and gathering the necessary information.\n\n<thought>\nThe task requires calculating the surface energy of Aluminum using:\n1. Materials Project ID: mp-134\n2. Direction: [0, 0, 1]\n3. EAM potential\n4. 5x5x5 supercell replication\n5. CG minimization with specific tolerances\n\nSurface energy formula: γ = (E_surface - E_bulk) / (2 * A)\nwhere:\n- E_surface: energy of the slab with surface\n- E_bulk: energy of the bulk structure\n- A: surface area\n- Factor of 2 accounts for two surfaces created\n\nI need to:\n1. Get the structure from Materials Project\n2. Find the appropriate EAM potential for Aluminum\n3. Create bulk configuration and minimize\n4. Create surface configuration (remove PBC in z-direction) and minimize\n5. Calculate surface energy\n\nLet me start by listing available potentials.\n</thought>\n\n<action>list_files</action>\n<action_input>{{"path": "/potentials"}}</action_input>"\n\nExamples of planning statements to remove:\n\n"<thought>\nGood, there's an EAM directory. Let me check what EAM potentials are available.\n</thought>\n\n<action>list_files</action>\n<action_input>{{"path": "/potentials/EAM"}}</action_input>"\n\n"<thought>\nGreat! The potential supports Si and O with pair_style tersoff. Now let me check the molten structure file to understand its format.\n</thought>\n\n<action>file_info</action>\n<action_input>{{\"path\": \"/structures/melt/SiO2/melt.dat\"}}</action_input>"\n\n\nShould this planning statement be kept?\n\nStatement: {message}""",  # noqa: RUF001
                },
            ],
            response_format=PlanningStatementValidation,
        )

        result = response.choices[0].message.parsed
        if result is not None:
            logger.info("Validated planning statement with LLM.")
            logger.info(" ")
            logger.info(f"Message: {message}")
            logger.info(f"Should keep: {result.should_keep}")
            logger.info(" ")
            logger.debug(
                f"Planning validation: {result.should_keep} - {result.reasoning}"
            )
            return result.should_keep
        else:
            logger.warning("Received None from LLM. Keeping by default.")
            return True
    except Exception as e:
        logger.warning(
            f"Failed to validate planning statement: {e}. Keeping by default."
        )
        return True


async def process_entries_async(
    all_cleaned_entries, client, stem2alldata, stem2filepath
):
    """Process all entries with async LLM validation."""
    # Merge annotations with fileNodes and remove unnecessary fields
    for entry in all_cleaned_entries:
        file_id = entry.get("fileId", None)
        annotator_key = entry.get("mongodbKey")
        if annotator_key and annotator_key in KEYS2NAMES:
            entry["annotator"] = KEYS2NAMES[annotator_key]

        # Add all data from files2label for this file_id
        file_id = entry.get("fileId", "N/A").replace(".json", "")
        if file_id in stem2alldata:
            # Add all fields from the original data
            for key, value in stem2alldata[file_id].items():
                entry[key] = value

        # Add tags based on environment
        environment = entry.get("environment")
        if environment and environment in ENV_TAGS:
            entry["tags"] = ENV_TAGS[environment]
        else:
            entry["tags"] = []

        entry.pop("mongodbKey", None)
        entry.pop("createdAt", None)

        # Load messages from the original log file
        messages_dict = {}
        if file_id in stem2filepath:
            log_file_path = Path(stem2filepath[file_id])
            try:
                if log_file_path.exists():
                    with log_file_path.open("r") as f:  # noqa: ASYNC230
                        log_data = json.load(f)
                        if "messages" in log_data:
                            for idx, msg in enumerate(log_data["messages"]):
                                messages_dict[str(idx)] = msg.get("content", "")
            except Exception as e:
                logger.warning(f"Failed to load messages from {log_file_path}: {e}")

        # Merge annotations into fileNodes
        annotations = entry.get("annotations", {})
        file_nodes = entry.get("fileNodes", [])

        # Initialize marker counts
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        marker_counts = dict.fromkeys(ALL_MARKERS, 0)

        # Collect planning statements to validate
        planning_validations = []
        planning_node_indices = []

        # Handle mismatched lengths between file_nodes and messages
        if len(file_nodes) != len(messages_dict):
            log_messages = log_data.get("messages", [])
            updated_file_nodes = []
            # Map original indices to new indices for annotation remapping
            old_to_new_index = {}

            for old_idx, node in enumerate(file_nodes):
                log_idx = len(updated_file_nodes)
                if node.get("type") != log_messages[log_idx].get("role"):
                    while node.get("type") != log_messages[log_idx].get("role"):
                        updated_file_nodes.append(
                            {
                                "id": str(log_idx),
                                "type": "tool",
                                "annotatable": False,
                            }
                        )
                        log_idx += 1
                    updated_file_nodes.append(
                        {
                            "id": str(log_idx),
                            "type": node.get("type"),
                            "annotatable": False,
                        }
                    )
                    old_to_new_index[str(old_idx)] = str(log_idx)
                else:
                    node["id"] = str(log_idx)
                    updated_file_nodes.append(
                        {
                            "id": (str(log_idx)),
                            "type": node.get("type"),
                            "annotatable": False,
                        }
                    )
                    old_to_new_index[str(old_idx)] = str(log_idx)

            # Add any remaining log messages as new file nodes
            if len(updated_file_nodes) != len(log_messages):
                diff = len(log_messages) - len(updated_file_nodes)
                for _i in range(diff):
                    log_idx = len(updated_file_nodes)
                    updated_file_nodes.append(
                        {
                            "id": str(log_idx),
                            "type": "tool",
                            "annotatable": False,
                        }
                    )

            # Remap annotations to new indices
            remapped_annotations = {}
            for old_idx, annotation_data in annotations.items():
                new_idx = old_to_new_index.get(old_idx)
                if new_idx is not None:
                    remapped_annotations[new_idx] = annotation_data
            annotations = remapped_annotations

            # Replace file_nodes with updated_file_nodes in the entry
            entry["fileNodes"] = updated_file_nodes
            file_nodes = entry.get("fileNodes", [])

        # Assign messages and annotations to file nodes
        if len(file_nodes) == len(log_data.get("messages", [])):
            for idx, node in enumerate(file_nodes):
                node["message"] = messages_dict.get(str(idx), "")
                if idx < 2:
                    node["annotatable"] = False
                else:
                    node["annotatable"] = check_annotatable(
                        node, messages_dict[node.get("id", "")]
                    )

                if node["annotatable"]:
                    markers = annotations.get(str(idx), {}).get("markers", [])
                    node["markers"] = [m for m in markers if m != "todo_list"]
                    node["notes"] = annotations.get(str(idx), {}).get("notes", "")
                    if not node["markers"]:
                        node["markers"] = ["neutral"]

        else:
            raise ValueError(
                f"File nodes length {len(file_nodes)} does not match messages length {len(log_data.get('messages', []))} for file_id {file_id}"
            )

        # Clean and validate planning statements
        for idx, node in enumerate(file_nodes):
            node_id = node.get("id")

            if node_id and node_id in annotations:
                # Add markers and notes to the node
                markers = node.get("markers", [])

                # Check planning statements with LLM (collect for batch processing)
                if "planning_statement" in markers:
                    message = node.get("message", "")
                    if message:
                        planning_validations.append(
                            validate_planning_statement(message, client)
                        )
                        planning_node_indices.append(idx)

        # Process all planning statements concurrently
        if planning_validations:
            validation_results = await asyncio.gather(
                *planning_validations, return_exceptions=True
            )

            # Apply validation results
            for idx, result in zip(
                planning_node_indices, validation_results, strict=True
            ):
                node = file_nodes[idx]
                markers = node.get("markers", [])

                # Handle exceptions
                if isinstance(result, Exception):
                    logger.warning(
                        f"Validation failed for node {node.get('id')}: {result}. Keeping planning_statement."
                    )
                    keep_marker = True
                else:
                    keep_marker = result

                if not keep_marker and "planning_statement" in markers:
                    markers = [m for m in markers if m != "planning_statement"]
                    logger.debug(
                        f"Removed planning_statement from node {node.get('id')}"
                    )

                node["markers"] = markers if markers else ["neutral"]

        # Now count markers after validation
        for node in file_nodes:
            node_id = node.get("id")
            if node_id and node_id in annotations:
                markers = node.get("markers", ["neutral"])
                node["notes"] = annotations[node_id].get("notes", "")

                # Count markers by type
                for marker in markers:
                    if marker in LABELS["Positive"]:
                        positive_count += 1
                    elif marker in LABELS["Negative"]:
                        negative_count += 1
                    elif marker in LABELS["Neutral"]:
                        neutral_count += 1

                    # Count individual markers
                    if marker in marker_counts:
                        marker_counts[marker] += 1

        # Add marker count summaries to entry
        entry["positive_marker_count"] = positive_count
        entry["negative_marker_count"] = negative_count
        entry["neutral_marker_count"] = neutral_count

        # Add individual marker counts
        for marker, count in marker_counts.items():
            entry[f"{marker}_count"] = count

        # Remove annotations dict since data is now in fileNodes
        entry.pop("annotations", None)


def main():
    # Initialize AsyncOpenAI client for planning statement validation
    client = AsyncOpenAI()

    files2label_path = Path(__file__).parent / "data" / "all_files2_label.json"
    with files2label_path.open("r") as f:
        files2label = json.load(f)

    # Create mapping from file stem to all data
    files2label_stem = [Path(file).stem for file in files2label]
    stem2alldata = {Path(file).stem: data for file, data in files2label.items()}
    stem2filepath = {Path(file).stem: file for file in files2label}

    with file_path.open("r") as f:
        data = json.load(f)

    # Group entries by annotator
    annotator_data = defaultdict(list)

    for entry in data:
        annotator_key = entry.get("mongodbKey")
        annotator_name = KEYS2NAMES.get(annotator_key, "Unknown")

        # Add to the annotator's list
        annotator_data[annotator_name].append(entry)

    # Store all cleaned entries and counts for plotting
    all_cleaned_entries = []
    annotator_counts = {}

    # Process each annotator's data
    for annotator_name, entries in annotator_data.items():
        # Filter entries to keep only those with file_id in files2label_stem
        original_count = len(entries)
        filtered_entries = [
            entry
            for entry in entries
            if entry.get("fileId", "N/A").replace(".json", "") in files2label_stem
        ]
        dropped_count = original_count - len(filtered_entries)

        # Also filter out entries with empty annotations
        entries_before_annotation_filter = len(filtered_entries)
        filtered_entries = [
            entry for entry in filtered_entries if entry.get("annotations", {})
        ]
        empty_annotations_dropped = entries_before_annotation_filter - len(
            filtered_entries
        )

        # Group by fileId to find duplicates for this annotator
        file_groups = defaultdict(list)

        for entry in filtered_entries:
            file_id = entry.get("fileId", "N/A").replace(".json", "")
            file_groups[file_id].append(entry)

        # Keep only the entry with submittedAt for each fileId
        cleaned_entries = []
        for file_id, group_entries in file_groups.items():
            if len(group_entries) == 1:
                cleaned_entries.append(group_entries[0])
            else:
                # Find entries with submittedAt
                submitted_entries = [
                    entry for entry in group_entries if "submittedAt" in entry
                ]

                if len(submitted_entries) == 0:
                    # No submittedAt found, keep the last one as fallback
                    logger.warning(
                        f"{annotator_name}, fileId {file_id}: No submittedAt found in {len(group_entries)} duplicates. Keeping last entry."
                    )
                    cleaned_entries.append(group_entries[-1])
                elif len(submitted_entries) == 1:
                    # Exactly one with submittedAt - this is what we want
                    cleaned_entries.append(submitted_entries[0])
                else:
                    # Multiple entries with submittedAt - use updatedAt to choose the most recent
                    logger.warning(
                        f"{annotator_name}, fileId {file_id}: Found {len(submitted_entries)} entries with submittedAt. Selecting the most recent by updatedAt."
                    )

                    # Parse updatedAt timestamps and find the most recent entry
                    entries_with_timestamps = []
                    for entry in submitted_entries:
                        updated_at_obj = entry.get("updatedAt", {})
                        submitted_at_obj = entry.get("submittedAt", {})

                        if isinstance(updated_at_obj, dict):
                            timestamp_str = updated_at_obj.get("$date", "")
                        else:
                            timestamp_str = str(updated_at_obj)

                        # If updatedAt is empty, try submittedAt
                        if not timestamp_str and isinstance(submitted_at_obj, dict):
                            timestamp_str = submitted_at_obj.get("$date", "")
                        elif not timestamp_str:
                            timestamp_str = str(submitted_at_obj)

                        entries_with_timestamps.append(
                            (entry, timestamp_str, updated_at_obj, submitted_at_obj)
                        )

                    # Sort by timestamp (most recent first) and take the first one
                    entries_with_timestamps.sort(key=lambda x: x[1], reverse=True)
                    most_recent_entry = entries_with_timestamps[0][0]

                    logger.warning("  All entries with their timestamps:")
                    for idx, (
                        _entry,
                        _timestamp,
                        raw_updated,
                        raw_submitted,
                    ) in enumerate(entries_with_timestamps):
                        marker = " <- SELECTED" if idx == 0 else ""
                        logger.warning(
                            f"    [{idx}] updatedAt: {raw_updated}, submittedAt: {raw_submitted}{marker}"
                        )
                    cleaned_entries.append(most_recent_entry)

        # Add to all cleaned entries
        all_cleaned_entries.extend(cleaned_entries)
        annotator_counts[annotator_name] = len(cleaned_entries)

        logger.info(
            f"{annotator_name}: {original_count} total entries -> {len(filtered_entries)} after filtering -> {len(cleaned_entries)} unique entries"
        )

        # Show entries dropped and duplicates removed
        if dropped_count > 0:
            logger.info(
                f"  Dropped {dropped_count} entry/entries (file_id not in files2label)"
            )
        if empty_annotations_dropped > 0:
            logger.info(
                f"  Dropped {empty_annotations_dropped} entry/entries (empty annotations)"
            )
        duplicates_removed = len(filtered_entries) - len(cleaned_entries)
        if duplicates_removed > 0:
            logger.info(f"  Removed {duplicates_removed} duplicate(s)")

    # Process all entries with async LLM validation
    asyncio.run(
        process_entries_async(all_cleaned_entries, client, stem2alldata, stem2filepath)
    )

    # Save one example entry as JSON for inspection
    if all_cleaned_entries:
        with json_example_file.open("w") as f:
            json.dump(all_cleaned_entries[0], f, indent=4)
        logger.info(f"Saved example entry to: {json_example_file}")

    # Convert to DataFrame and save as Parquet
    # Convert complex fields to JSON strings for Parquet compatibility
    entries_for_parquet = []
    for entry in all_cleaned_entries:
        entry_copy = entry.copy()
        # Convert complex fields to JSON strings
        if "_id" in entry_copy:
            entry_copy["_id"] = json.dumps(entry_copy["_id"])
        if "updatedAt" in entry_copy:
            entry_copy["updatedAt"] = json.dumps(entry_copy["updatedAt"])
        if "fileNodes" in entry_copy:
            entry_copy["fileNodes"] = json.dumps(entry_copy["fileNodes"])
        if "trial_info" in entry_copy:
            entry_copy["trial_info"] = json.dumps(entry_copy["trial_info"])
        entries_for_parquet.append(entry_copy)

    annotations_df = pd.DataFrame(entries_for_parquet)
    annotations_df.to_parquet(output_file, index=False)
    logger.info(
        f"\nSaved {len(all_cleaned_entries)} total cleaned entries to: {output_file}"
    )

    # Add QA scores to the dataframe
    logger.info("\nAdding QA scores to annotations...")
    qa_base_path = Path(__file__).parent.parent / "qa"

    # Map environment names to QA directory names
    env_map = {
        "md": "md_qa",
        "ml": "ml_qa",
        "catalyst": "catalyst_qa",
        "resistor": "resistor_qa",
        "afm": "afm_qa",
        "retrosynthesis": "retro_qa",
        "spectra": "spectra_qa",
    }

    # Map model names to directory names
    model_map = {
        "gpt-4o": "gpt",
        "claude_sonnet_45": "claude",
    }

    def get_qa_score(environment, model):
        """Get the QA score for a given environment and model."""
        if environment not in env_map or model not in model_map:
            return None

        summary_path = (
            qa_base_path
            / env_map[environment]
            / model_map[model]
            / "reports"
            / "topic_reports"
            / "summary.json"
        )

        if not summary_path.exists():
            logger.warning(f"QA summary not found: {summary_path}")
            return None

        try:
            with summary_path.open("r") as f:
                summary_data = json.load(f)
            return summary_data.get("overall_score")
        except Exception as e:
            logger.warning(f"Error reading QA summary {summary_path}: {e}")
            return None

    annotations_df["qa_score"] = annotations_df.apply(
        lambda row: get_qa_score(row["environment"], row["model"]), axis=1
    )

    # Save the updated dataframe with QA scores
    annotations_df.to_parquet(output_file, index=False)
    logger.info(
        f"Added QA scores: {annotations_df['qa_score'].notna().sum()} rows with scores"
    )

    if annotations_df["qa_score"].notna().any():
        logger.info("\nQA Score Distribution:")
        score_dist = annotations_df.groupby(["environment", "model"])["qa_score"].agg(
            ["first", "count"]
        )
        logger.info(f"\n{score_dist}")

    # Create bar plot
    fig, ax = plt.subplots(
        figsize=(lama_aesthetics.TWO_COL_WIDTH, lama_aesthetics.TWO_COL_HEIGHT)
    )

    # Sort annotators by count (descending)
    sorted_items = sorted(annotator_counts.items(), key=lambda x: x[1], reverse=True)
    annotators = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    bars = ax.bar(annotators, counts, color="steelblue", edgecolor="black")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Annotator", fontsize=12)
    ax.set_ylabel("Number of Annotations", fontsize=12)
    range_frame(ax, np.array([0, len(annotators) - 1]), np.array([0, max(counts)]))

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_dir = Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    plot_file = Path(__file__).parent / "plots" / "annotation_counts.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved bar plot to: {plot_file}")
    plt.close()


if __name__ == "__main__":
    main()
