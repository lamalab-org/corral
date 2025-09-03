"""
Embed the tool descriptions of the tools with the different verbosity levels:
- Brief
- Workflow
- Comprehensive
The script reads the tools files of the specified environments and extracts the tool descriptions for each verbosity level.
This works by parsing the Python files in the tools directory and looking for functions decorated with @tool.
Then embed them using OpenAI's embedding model (text-embedding-3-large).
The embeddings are saved as numpy files following the next path:
- tool embeddings: f"embeddings/{task_name}_embeddings_{level}.npy"
- tool names: f"embeddings/{task_name}_tool_names_{level}.npy"
"""

import ast
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv("../../.env", override=True)
client = OpenAI()


def extract_tool_functions(file_path: str) -> dict[str, str]:
    """
    Extract functions with @tool decorator and their docstrings from a Python file.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        Dictionary with function names as keys and docstrings as values
    """
    tool_functions = {}

    try:
        with Path(file_path).open(encoding="utf-8") as file:
            content = file.read()

        # Parse the Python file into an AST
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if the function has decorators
                has_tool_decorator = False

                for decorator in node.decorator_list:
                    # Handle simple decorator like @tool
                    if (
                        isinstance(decorator, ast.Name)
                        and decorator.id == "tool"
                        or isinstance(decorator, ast.Attribute)
                        and decorator.attr == "tool"
                    ):
                        has_tool_decorator = True
                        break
                    # Handle decorator with arguments like @tool(hidden_args=["h_smiles"])
                    # Check if the function being called is 'tool'
                    if (
                        isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "tool"
                        or isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == "tool"
                    ):
                        has_tool_decorator = True
                        break

                if has_tool_decorator:
                    # Extract the docstring
                    docstring = ast.get_docstring(node)
                    if docstring:
                        tool_functions[node.name] = docstring
                    else:
                        tool_functions[node.name] = ""

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

    return tool_functions


def find_tools_files(task_path: Path) -> list:
    """
    Find tools.py files in task directory, checking both src and task subdirectories.

    Args:
        task_path: Path to the task directory

    Returns:
        List of paths to tools.py files found
    """
    tools_files = []

    # Check for tools.py in the main task directory
    main_tools = task_path / "tools.py"
    if main_tools.exists():
        tools_files.append(main_tools)

    # Check for tools.py in src subdirectory
    src_tools = task_path / "src" / "tools.py"
    if src_tools.exists():
        tools_files.append(src_tools)

    # Check for tools.py in task subdirectory (named same as parent)
    task_tools = task_path / task_path.name / "tools.py"
    if task_tools.exists():
        tools_files.append(task_tools)

    return tools_files


def extract_task_tools(
    tasks_directory: str, task_list: list | None = None
) -> dict[str, dict[str, str]]:
    """
    Extract tool functions from specified tasks or all tasks in directory.

    Args:
        tasks_directory: Path to the directory containing task folders
        task_list: List of specific task names to process. If None, process all tasks.

    Returns:
        Dictionary with task names as keys and nested dictionaries of tool functions
    """
    tasks_dir = Path(tasks_directory)

    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_directory}")

    result = {}

    # Get list of tasks to process
    tasks_to_process = []
    if task_list is None:
        # Process all subdirectories
        for item in tasks_dir.iterdir():
            if item.is_dir():
                tasks_to_process.append(item)
                # Check for src subdirectory with a task folder inside
                src_dir = item / "src"
                if src_dir.exists() and src_dir.is_dir():
                    tasks_to_process.extend(
                        [
                            subitem
                            for subitem in src_dir.iterdir()
                            if subitem.is_dir() and subitem.name == item.name
                        ]
                    )
    else:
        # Process only specified tasks
        for task_name in task_list:
            main_task = tasks_dir / task_name
            if main_task.exists() and main_task.is_dir():
                tasks_to_process.append(main_task)
                src_dir = main_task / "src"
                if src_dir.exists() and src_dir.is_dir():
                    src_task = src_dir / task_name
                    if src_task.exists() and src_task.is_dir():
                        tasks_to_process.append(src_task)

    for task_path in tasks_to_process:
        task_name = task_path.name
        logger.info(f"Processing task: {task_name}")

        # Find all tools.py files in this task
        tools_files = find_tools_files(task_path)

        if not tools_files:
            logger.warning(f"No tools.py file found in task: {task_name}")
            result[task_name] = {}
            continue

        # Extract tools from all found tools.py files
        task_tools = {}
        for tools_file in tools_files:
            logger.info(f"  Processing tools file: {tools_file}")
            file_tools = extract_tool_functions(str(tools_file))
            task_tools.update(file_tools)
            logger.info(f"    Found {len(file_tools)} tool functions")

        result[task_name] = task_tools

    return result


def embed_text(text: str) -> list[float]:
    """
    Embed text using OpenAI's text-embedding-3-small model.
    """
    response = client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding


def format_tool_for_embedding(docstring: str, mode: str = "full") -> str:
    # Format tool information for embedding based on mode.
    # Modes:
    #   'brief': extract text between [BRIEF] and [/BRIEF]
    #   'workflow': extract text between [BRIEF] and [/WORKFLOW_INTEGRATION]
    #   'full': use entire docstring
    if mode == "brief":
        import re

        match = re.search(r"\[BRIEF\](.*?)\[/BRIEF\]", docstring, re.DOTALL)
        return match.group(1).strip() if match else ""
    elif mode == "workflow":
        import re

        match = re.search(
            r"\[BRIEF\](.*?)\[/WORKFLOW_INTEGRATION\]", docstring, re.DOTALL
        )
        return match.group(1).strip() if match else ""
    else:
        return docstring


def embed_and_save_task_tools(
    task_tools: dict[str, dict[str, str]], output_dir: str = "embeddings"
):
    """
    Embed all tools for each task and save as numpy files for three levels: brief, workflow, full.
    Args:
        task_tools: Dictionary mapping task names to their tools
        output_dir: Directory to save the embedding files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    levels = ["brief", "workflow", "full"]
    for task_name, tools in task_tools.items():
        logger.info(f"\n=== Processing Task: {task_name} ===")
        if not tools:
            logger.warning(f"  No tools found for task: {task_name}")
            continue
        for level in levels:
            logger.info(f"  -- Embedding level: {level}")
            embeddings = []
            tool_names = []
            for tool_name, docstring in tools.items():
                logger.info(f"    Processing tool: {tool_name}")
                formatted_text = format_tool_for_embedding(docstring, mode=level)
                try:
                    embedding = embed_text(formatted_text)
                    embeddings.append(embedding)
                    tool_names.append(tool_name)
                    logger.info(f"      ✓ Successfully embedded {tool_name}")
                except Exception as e:
                    logger.error(f"      ✗ Failed to embed {tool_name}: {e}")
                    continue
            if embeddings:
                embeddings_array = np.array(embeddings)
                embeddings_file = (
                    Path(output_dir) / f"{task_name}_embeddings_{level}.npy"
                )
                np.save(embeddings_file, embeddings_array)
                metadata_file = Path(output_dir) / f"{task_name}_tool_names_{level}.npy"
                np.save(metadata_file, np.array(tool_names))
                logger.info(
                    f"    ✓ Saved {len(embeddings)} embeddings to {embeddings_file}"
                )
                logger.info(f"    ✓ Saved tool names to {metadata_file}")
                logger.info(f"    ✓ Embedding shape: {embeddings_array.shape}")
            else:
                logger.warning(
                    f"    ✗ No embeddings generated for task: {task_name} ({level})"
                )


def load_task_embeddings(task_name: str, embeddings_dir: str = "embeddings"):
    """
    Load embeddings and tool names for a specific task.

    Args:
        task_name: Name of the task
        embeddings_dir: Directory containing the embedding files

    Returns:
        tuple: (embeddings_array, tool_names_array)
    """
    """
    Load embeddings and tool names for a specific task and embedding level.
    Args:
        task_name: Name of the task
        embeddings_dir: Directory containing the embedding files
        level: Embedding level ('brief', 'workflow', 'full')
    Returns:
        tuple: (embeddings_array, tool_names_array)
    """
    # Default to 'full' level if not specified
    level = "full"
    embeddings_file = Path(embeddings_dir) / f"{task_name}_embeddings_{level}.npy"
    metadata_file = Path(embeddings_dir) / f"{task_name}_tool_names_{level}.npy"

    if not Path(embeddings_file).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    embeddings = np.load(embeddings_file)
    tool_names = np.load(metadata_file) if Path(metadata_file).exists() else None

    return embeddings, tool_names


def main():
    """
    Main function to demonstrate usage of the task tools extractor with embedding.
    """
    # Example usage
    tasks_directory = "../../tasks"  # Change this to your tasks directory path
    specific_tasks = ["spectra_elucidation", "ml", "corral_md", "catalyst"]

    # Extract task tools (assuming this function exists)
    task_tools = extract_task_tools(tasks_directory, specific_tasks)

    # Print results (original functionality)
    for task_name, tools in task_tools.items():
        logger.info(f"\n=== Task: {task_name} ===")
        if not tools:
            logger.warning("  No tool functions found")
        else:
            for func_name, docstring in tools.items():
                logger.info(f"  Function: {func_name}")
                if docstring:
                    logger.info(
                        f"  Docstring preview: {docstring[:100]}..."
                    )  # Print first 100 characters of docstring
                else:
                    raise ValueError(f"No docstring found for {func_name}")

    # Embed and save tools
    logger.info("\n" + "=" * 50)
    logger.info("EMBEDDING AND SAVING TOOLS")
    logger.info("=" * 50)
    embed_and_save_task_tools(task_tools)

    return task_tools


def example_usage():
    """
    Example of how to load and use the saved embeddings.
    """
    try:
        # Load embeddings for a specific task
        task_name = "spectra_elucidation"
        embeddings, tool_names = load_task_embeddings(task_name)

        logger.info(f"\nLoaded embeddings for task: {task_name}")
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Tool names: {tool_names}")

        # You can now use these embeddings for similarity search, clustering, etc.

    except FileNotFoundError as e:
        logger.error(f"Could not load embeddings: {e}")


if __name__ == "__main__":
    # Run the main function
    extracted_tools = main()

    # Example of loading the saved embeddings
    logger.info("\n" + "=" * 50)
    logger.info("EXAMPLE: LOADING SAVED EMBEDDINGS")
    logger.info("=" * 50)
    example_usage()
