"""
Code2Latex - Automatic LaTeX documentation generation from Corral task environments.

This module provides utilities for generating formatted LaTeX files (with colorboxes)
containing task descriptions, tools, and scoring information.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class CacheMetadata:
    """Metadata for cache file matching."""

    env_name: str
    level: str | int


class Code2Latex:
    """
    Utility class for generating LaTeX documentation from task environments.

    This class provides methods to generate formatted LaTeX files with colorboxes
    containing task descriptions, available tools, and scoring information.
    It includes caching support to accumulate subtasks over multiple runs.
    """

    DEFAULT_CACHE_DIR: ClassVar[str] = ".code2latex_cache"

    # LaTeX color schemes
    MAIN_TASK_COLORS: ClassVar[dict[str, str]] = {
        "background": "blue!5",
        "frame": "blue!75!black",
    }
    SUBTASK_COLORS: ClassVar[dict[str, str]] = {
        "background": "gray!5",
        "frame": "gray!60!black",
    }

    @classmethod
    def colorbox(
        cls,
        name: str,
        description: str,
        tools: list[str],
        scoring_fn: Callable,
        metadata: CacheMetadata,
        output_dir: str,
        task_id: str | None = None,
        subtask_index: int | None = None,
        cache_dir: str | None = None,
    ) -> str:
        """
        Generate LaTeX colorbox documentation for a task.

        Workflow based on cache state:
        1. If NO cache exists → this is the main task:
           - Save task to cache as main_task
           - Do NOT generate .tex file yet (wait for subtasks)
        2. If cache EXISTS → this is a subtask:
           - Add task to cache as subtask (nested inside main task)
           - Generate complete .tex file with main task + all subtasks

        Args:
            name: Task name
            description: Task description
            tools: List of tool names
            scoring_fn: Scoring function
            metadata: Metadata for cache matching
            output_dir: Directory for .tex output
            task_id: Optional task identifier (defaults to name)
            subtask_index: Optional index for ordering subtasks
            cache_dir: Optional custom cache directory

        Returns:
            Path to generated .tex file (empty string for main task until subtasks added)
        """
        cache_path = cls._get_cache_path(
            metadata, Path(cache_dir) if cache_dir else Path(cls.DEFAULT_CACHE_DIR)
        )

        # Build task dict for caching
        task_dict = {
            "task_id": task_id or name,
            "name": name,
            "description": description,
            "tools": tools,
            "scoring_function": scoring_fn.__name__
            if hasattr(scoring_fn, "__name__")
            else str(scoring_fn),
            "subtask_index": subtask_index,
        }

        # Check if cache exists to determine if this is main task or subtask
        existing_cache = cls._load_cache(cache_path)

        if existing_cache is None:
            # No cache → this is the main task
            cache_data = cls._create_cache_entry(metadata, task_dict)
            cls._save_cache(cache_path, cache_data)
            logger.info(f"Saved main task to cache: {cache_path}")
            # Return empty string - .tex will be generated when subtasks are added
            return ""
        else:
            # Cache exists → this is a subtask, add it inside the main task
            cache_data = cls._merge_task_data(existing_cache, task_dict)
            logger.info(f"Added subtask to cache: {task_dict.get('task_id')}")

            # Save updated cache (in case more subtasks follow)
            cls._save_cache(cache_path, cache_data)

            # Generate output file
            output_path = (
                Path(output_dir) / f"{metadata.env_name}_level_{metadata.level}.tex"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            generated_path = cls._render_latex(cache_data, output_path)
            logger.info(f"Generated LaTeX file: {generated_path}")

            return generated_path

    @classmethod
    def _get_cache_path(cls, metadata: CacheMetadata, cache_dir: Path) -> Path:
        """Generate cache file path from metadata."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{metadata.env_name}_level_{metadata.level}.json"

    @classmethod
    def _load_cache(cls, cache_path: Path) -> dict[str, Any] | None:
        """Load existing cache entry if exists."""
        if not cache_path.exists():
            return None

        try:
            with cache_path.open(encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Loaded cache from: {cache_path}")
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Failed to load cache from {cache_path}: {e}. Treating as empty cache."
            )
            return None

    @classmethod
    def _save_cache(cls, cache_path: Path, data: dict[str, Any]) -> None:
        """Save cache entry to JSON file."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved cache to: {cache_path}")

    @classmethod
    def _create_cache_entry(
        cls, metadata: CacheMetadata, task_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a new cache entry from metadata and task data as main task."""
        return {
            "metadata": asdict(metadata),
            "main_task": task_dict,
            "subtasks": [],
        }

    @classmethod
    def _remove_cache(cls, cache_path: Path) -> None:
        """Remove cache file if it exists."""
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Deleted cache file: {cache_path}")

    @classmethod
    def _merge_task_data(
        cls, existing: dict[str, Any], new_task_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merge new task data into existing cache entry as a subtask.

        When cache exists, all new tasks are added as subtasks inside the main task.
        If a subtask with the same task_id exists, it is updated.
        """
        result = existing.copy()

        # Always add as subtask when merging (cache exists = subtask)
        subtasks = result.get("subtasks", [])
        existing_idx = None
        for i, subtask in enumerate(subtasks):
            if subtask.get("task_id") == new_task_dict.get("task_id"):
                existing_idx = i
                break

        if existing_idx is not None:
            # Update existing subtask
            subtasks[existing_idx] = new_task_dict
            logger.debug(f"Updated existing subtask: {new_task_dict.get('task_id')}")
        else:
            # Add new subtask
            subtasks.append(new_task_dict)
            logger.debug(f"Added new subtask: {new_task_dict.get('task_id')}")

        result["subtasks"] = subtasks
        return result

    @classmethod
    def _render_latex(cls, cache_data: dict[str, Any], output_path: Path) -> str:
        """Render the LaTeX file from cache data."""
        latex_content = cls._generate_latex_content(cache_data)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(latex_content)

        return str(output_path)

    @classmethod
    def _generate_latex_content(cls, cache_data: dict[str, Any]) -> str:
        """Generate complete LaTeX content from cache data."""
        lines = []

        # Add LaTeX preamble comment
        lines.append("% Auto-generated by Code2Latex")
        lines.append("% Requires: tcolorbox, listings packages")
        lines.append("")

        main_task = cache_data.get("main_task")
        subtasks = cache_data.get("subtasks", [])

        if main_task:
            # Generate main task colorbox
            lines.append(cls._generate_task_colorbox(main_task, is_main=True))

            # Add subtasks nested inside main task (before closing the main box)
            # Subtasks are rendered in the order they were added (execution order)
            if subtasks:
                for i, subtask in enumerate(subtasks, 1):
                    lines.append("")
                    lines.append(f"% Subtask {i}")
                    lines.append(
                        cls._generate_task_colorbox(
                            subtask, is_main=False, subtask_num=i
                        )
                    )

            # Close main task box
            lines.append("")
            lines.append(r"\end{tcolorbox}")
        elif subtasks:
            # No main task, just render subtasks in the order they were added
            for i, subtask in enumerate(subtasks, 1):
                if i > 1:
                    lines.append("")
                lines.append(f"% Subtask {i}")
                lines.append(
                    cls._generate_task_colorbox(
                        subtask, is_main=False, subtask_num=i, standalone=True
                    )
                )

        return "\n".join(lines)

    @classmethod
    def _generate_task_colorbox(
        cls,
        task: dict[str, Any],
        is_main: bool = True,
        subtask_num: int | None = None,
        standalone: bool = False,
    ) -> str:
        """Generate a single task colorbox."""
        lines = []

        # Select colors based on task type
        colors = cls.MAIN_TASK_COLORS if is_main else cls.SUBTASK_COLORS

        # Build title
        name = task.get("name", "Untitled Task")
        if not is_main and subtask_num is not None:
            title = f"Subtask {subtask_num}: {name}"
        else:
            title = name

        # Escape special LaTeX characters in title
        title = cls._escape_latex(title)

        # Start colorbox
        lines.append(rf"\begin{{tcolorbox}}[title={title}, breakable, enhanced,")
        lines.append(
            rf"                  colback={colors['background']}, colframe={colors['frame']}]"
        )
        lines.append("")

        # Add description
        description = task.get("description", "")
        if description:
            lines.append(cls._escape_latex(description))
            lines.append("")

        # Only show Tools and Scoring Function for main task
        if is_main:
            # Add tools
            tools = task.get("tools", [])
            if tools:
                lines.append(r"\medskip")
                tools_formatted = ", ".join(
                    rf"\texttt{{{cls._escape_latex(t)}}}" for t in tools
                )
                lines.append(rf"\textbf{{Tools:}} {tools_formatted}")
                lines.append("")

            # Add scoring function
            scoring_fn = task.get("scoring_function", "")
            if scoring_fn:
                lines.append(r"\medskip")
                lines.append(
                    rf"\textbf{{Scoring Function:}} \texttt{{{cls._escape_latex(scoring_fn)}}}"
                )
                lines.append("")

        # Close box for subtasks or standalone
        if not is_main or standalone:
            lines.append(r"\end{tcolorbox}")

        return "\n".join(lines)

    @classmethod
    def _escape_latex(cls, text: str) -> str:
        """Escape special LaTeX characters in text."""
        if not text:
            return text

        # Characters that need escaping in LaTeX
        replacements = [
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ]

        result = text
        for char, replacement in replacements:
            # Don't double-escape backslashes
            if char == "\\":
                # Only escape lone backslashes, not already escaped ones
                result = result.replace(char, replacement)
            else:
                result = result.replace(char, replacement)

        return result

    @classmethod
    def longtable(
        cls,
        tools: list[dict[str, Any]],
        metadata: CacheMetadata,
        output_dir: str,
        cache_dir: str | None = None,
    ) -> str:
        """
        Generate LaTeX longtable documentation for tools.

        Similar to colorbox, this method uses caching to accumulate tools over multiple runs.

        Args:
            tools: List of tool dictionaries with keys:
                - name: Tool name
                - description: Tool description
                - arguments: List of argument dicts with name, type, description, required, default, choices
            metadata: Metadata for cache matching
            output_dir: Directory for .tex output
            cache_dir: Optional custom cache directory

        Returns:
            Path to generated .tex file
        """
        cache_path = cls._get_longtable_cache_path(
            metadata, Path(cache_dir) if cache_dir else Path(cls.DEFAULT_CACHE_DIR)
        )

        # Load existing cache or create new
        existing_cache = cls._load_cache(cache_path)

        if existing_cache:
            # Merge new tools with existing
            cache_data = cls._merge_tools_data(existing_cache, tools)
            logger.info(f"Merged tools data into existing cache: {cache_path}")
        else:
            # Create new cache entry
            cache_data = cls._create_tools_cache_entry(metadata, tools)
            logger.info(f"Created new tools cache entry: {cache_path}")

        # Save updated cache
        cls._save_cache(cache_path, cache_data)

        # Generate output file
        output_path = (
            Path(output_dir) / f"{metadata.env_name}_level_{metadata.level}_tools.tex"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generated_path = cls._render_longtable_latex(cache_data, output_path)
        logger.info(f"Generated tools LaTeX file: {generated_path}")

        return generated_path

    @classmethod
    def _get_longtable_cache_path(
        cls, metadata: CacheMetadata, cache_dir: Path
    ) -> Path:
        """Generate cache file path for longtable tools from metadata."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{metadata.env_name}_level_{metadata.level}_tools.json"

    @classmethod
    def _create_tools_cache_entry(
        cls, metadata: CacheMetadata, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create a new cache entry for tools."""
        return {
            "metadata": asdict(metadata),
            "tools": {tool["name"]: tool for tool in tools},
        }

    @classmethod
    def _merge_tools_data(
        cls, existing: dict[str, Any], new_tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Merge new tools data into existing cache entry.

        Tools are merged by name - if a tool with the same name exists, it is updated.
        """
        result = existing.copy()
        tools_dict = result.get("tools", {})

        for tool in new_tools:
            tool_name = tool.get("name")
            if tool_name:
                tools_dict[tool_name] = tool
                logger.debug(f"Added/updated tool: {tool_name}")

        result["tools"] = tools_dict
        return result

    @classmethod
    def _render_longtable_latex(
        cls, cache_data: dict[str, Any], output_path: Path
    ) -> str:
        """Render the LaTeX longtable file from cache data."""
        latex_content = cls._generate_longtable_content(cache_data)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(latex_content)

        return str(output_path)

    @classmethod
    def _generate_longtable_content(cls, cache_data: dict[str, Any]) -> str:
        """Generate complete LaTeX longtable content from cache data."""
        lines = []

        # Add LaTeX preamble comment
        lines.append("% Auto-generated by Code2Latex")
        lines.append("% Requires: longtable, booktabs packages")
        lines.append("")

        tools = cache_data.get("tools", {})
        metadata = cache_data.get("metadata", {})

        if not tools:
            lines.append("% No tools found")
            return "\n".join(lines)

        # Sort tools by name for consistent ordering
        sorted_tools = sorted(tools.values(), key=lambda x: x.get("name", ""))

        # Begin longtable - single column for description-style layout
        lines.append(r"\begin{longtable}{p{\textwidth}}")
        lines.append(
            r"\caption{Tools for "
            + cls._escape_latex(metadata.get("env_name", "Unknown"))
            + r" (Level "
            + cls._escape_latex(str(metadata.get("level", "")))
            + r")} \\"
        )
        lines.append(r"\endfirsthead")
        lines.append("")
        lines.append(
            r"\multicolumn{1}{c}{\tablename\ \thetable{} -- continued from previous page} \\"
        )
        lines.append(r"\endhead")
        lines.append("")
        lines.append(r"\multicolumn{1}{r}{Continued on next page} \\")
        lines.append(r"\endfoot")
        lines.append("")
        lines.append(r"\endlastfoot")
        lines.append("")

        # Add each tool
        for i, tool in enumerate(sorted_tools):
            if i > 0:
                lines.append(r"\midrule")

            name = cls._escape_latex(tool.get("name", ""))
            description = cls._escape_latex(tool.get("description", ""))
            returns_info = cls._escape_latex(tool.get("returns", ""))

            # Tool name as header
            lines.append(rf"\textbf{{\texttt{{{name}}}}} \\")

            # Begin description environment
            lines.append(r"\begin{description}")

            # Tool description first
            lines.append(rf"    \item[\textbf{{Description:}}] {description}")

            # Format arguments
            arguments = tool.get("arguments", [])
            lines.append(r"    \item[\textbf{Arguments:}]")
            if arguments:
                lines.append(r"    \begin{description}")
                for arg in arguments:
                    arg_name = cls._escape_latex(arg.get("name", ""))
                    arg_type = cls._escape_latex(arg.get("type", ""))
                    arg_desc = cls._escape_latex(arg.get("description", ""))
                    lines.append(
                        rf"        \item \texttt{{{arg_name}}} ({arg_type}): {arg_desc}"
                    )
                lines.append(r"    \end{description}")
            else:
                lines.append(r"    None")

            # Returns section if available
            if returns_info:
                lines.append(rf"    \item[\textbf{{Returns:}}] {returns_info}")

            lines.append(r"\end{description} \\")

        lines.append(r"\end{longtable}")

        return "\n".join(lines)

    @classmethod
    def _format_tool_arguments(cls, arguments: list[dict[str, Any]]) -> str:
        """Format tool arguments for LaTeX table cell."""
        if not arguments:
            return "None"

        arg_lines = []
        for arg in arguments:
            name = cls._escape_latex(arg.get("name", ""))
            arg_type = cls._escape_latex(arg.get("type", ""))
            description = cls._escape_latex(arg.get("description", ""))
            required = arg.get("required", False)
            default = arg.get("default")
            choices = arg.get("choices")

            # Build argument string
            req_marker = "*" if required else ""
            arg_str = rf"\texttt{{{name}{req_marker}}} ({arg_type})"

            if description:
                arg_str += f": {description}"

            if choices:
                choices_str = ", ".join(cls._escape_latex(str(c)) for c in choices)
                arg_str += rf" [choices: {choices_str}]"
            elif default is not None:
                arg_str += rf" [default: {cls._escape_latex(str(default))}]"

            arg_lines.append(arg_str)

        # Join with newlines within the cell
        return r" \newline ".join(arg_lines)

    @classmethod
    def clear_cache(
        cls,
        env_name: str | None = None,
        level: str | int | None = None,
        cache_dir: str | None = None,
    ) -> int:
        """
        Clear cache files.

        Args:
            env_name: Optional environment name to filter by
            level: Optional level to filter by
            cache_dir: Optional custom cache directory

        Returns:
            Number of cache files deleted
        """
        cache_path = Path(cache_dir) if cache_dir else Path(cls.DEFAULT_CACHE_DIR)

        if not cache_path.exists():
            return 0

        deleted_count = 0

        if env_name and level is not None:
            # Delete specific cache files (both main and tools)
            targets = [
                cache_path / f"{env_name}_level_{level}.json",
                cache_path / f"{env_name}_level_{level}_tools.json",
            ]
            for target in targets:
                if target.exists():
                    target.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted cache file: {target}")
        elif env_name:
            # Delete all cache files for an environment (both main and tools)
            for cache_file in cache_path.glob(f"{env_name}_level_*.json"):
                cache_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted cache file: {cache_file}")
            logger.info(
                f"Deleted {deleted_count} cache files for environment: {env_name}"
            )
        else:
            # Delete all cache files
            for cache_file in cache_path.glob("*.json"):
                cache_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted cache file: {cache_file}")
            logger.info(f"Deleted {deleted_count} cache files")

        return deleted_count
