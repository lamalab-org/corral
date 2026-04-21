import re
from enum import Enum
from typing import ClassVar

from loguru import logger

from corral.backend.env import Environment
from corral.backend.tool_utils import format_json_schema_type


class MalformedDocstringError(ValueError):
    """Raised when a docstring has unclosed tags"""


class ToolVerbosity(Enum):
    """Defines different levels of tool description verbosity for ablation studies"""

    MINIMAL = "minimal"  # Just tool name and basic description
    BRIEF = "brief"  # [BRIEF] sections only
    DETAILED = "detailed"  # [BRIEF] + [DETAILED] sections
    PROCEDURAL = "procedural"  # [BRIEF] + [DETAILED] + [PROCEDURAL] sections
    CONTEXTUAL = (
        "contextual"  # [BRIEF] + [DETAILED] + [PROCEDURAL] + [CONTEXTUAL] sections
    )
    WORKFLOW = "workflow"  # [BRIEF] + [DETAILED] + [PROCEDURAL] + [CONTEXTUAL] + [WORKFLOW_INTEGRATION] sections
    SYNTACTICAL = "syntactical"  # [BRIEF] + [DETAILED] + [PROCEDURAL] + [CONTEXTUAL] + [WORKFLOW_INTEGRATION] + [SYNTACTICAL] sections
    COMPREHENSIVE = (
        "comprehensive"  # All sections including [RAISES], [LIMITATIONS], etc.
    )
    FULL = "full"  # Original complete docstring without any filtering


class VerbosityConfig:
    """Configuration for tool verbosity levels"""

    # Define which sections to include for each verbosity level
    VERBOSITY_SECTIONS: ClassVar[dict[ToolVerbosity, set[str]]] = {
        ToolVerbosity.MINIMAL: {
            "BASIC"
        },  # Fixed: MINIMAL should show basic description
        ToolVerbosity.BRIEF: {"BASIC", "BRIEF"},  # Include basic as fallback
        ToolVerbosity.DETAILED: {"BASIC", "BRIEF", "DETAILED"},
        ToolVerbosity.PROCEDURAL: {"BASIC", "BRIEF", "DETAILED", "PROCEDURAL"},
        ToolVerbosity.CONTEXTUAL: {
            "BASIC",
            "BRIEF",
            "DETAILED",
            "PROCEDURAL",
            "CONTEXTUAL",
        },
        ToolVerbosity.WORKFLOW: {
            "BASIC",
            "BRIEF",
            "DETAILED",
            "PROCEDURAL",
            "CONTEXTUAL",
            "WORKFLOW_INTEGRATION",
        },
        ToolVerbosity.SYNTACTICAL: {
            "BASIC",
            "BRIEF",
            "DETAILED",
            "PROCEDURAL",
            "CONTEXTUAL",
            "WORKFLOW_INTEGRATION",
            "SYNTACTICAL",
        },
        ToolVerbosity.COMPREHENSIVE: {
            "BASIC",
            "BRIEF",
            "DETAILED",
            "PROCEDURAL",
            "CONTEXTUAL",
            "WORKFLOW_INTEGRATION",
            "SYNTACTICAL",
            "RAISES",
            "LIMITATIONS",
            "EXAMPLES",
        },
        ToolVerbosity.FULL: {"*"},  # Special case for all sections
    }

    @classmethod
    def get_sections_for_verbosity(cls, verbosity: ToolVerbosity) -> set[str]:
        """Get the set of sections to include for a given verbosity level"""
        return cls.VERBOSITY_SECTIONS.get(verbosity, set())

    """Enhanced docstring processor with proper parsing and verbosity filtering"""

    # All supported tagged sections
    SUPPORTED_KEYWORDS: ClassVar[list[str]] = [
        "BRIEF",
        "DETAILED",
        "PROCEDURAL",
        "WORKFLOW_INTEGRATION",
        "CONTEXTUAL",
        "SYNTACTICAL",
        "RAISES",
        "LIMITATIONS",
        "EXAMPLES",
        "ARGS_BRIEF",
        "ARGS_DETAILED",
        "ARGS_SYNTACTICAL",
        "ARGS_EXAMPLES",
        "RETURNS_BRIEF",
        "RETURNS_DETAILED",
        "RETURNS_EXAMPLES",
    ]

    # Pre-compile regex patterns for performance
    _keyword_pattern: ClassVar[str] = "|".join(
        re.escape(keyword) for keyword in SUPPORTED_KEYWORDS
    )
    _keyword_regex: ClassVar[re.Pattern] = re.compile(
        rf"\[({_keyword_pattern})\](.*?)\[/\1\]", re.DOTALL | re.IGNORECASE
    )
    _args_regex: ClassVar[re.Pattern] = re.compile(
        r"Args:(.*?)(?=Returns:|$)", re.DOTALL | re.IGNORECASE
    )
    _returns_regex: ClassVar[re.Pattern] = re.compile(
        r"Returns:(.*?)$", re.DOTALL | re.IGNORECASE
    )

    # Cache for parsed docstrings to improve performance
    _parsed_cache: ClassVar[dict[str, dict[str, str]]] = {}
    _filtered_cache: ClassVar[
        dict[tuple, str]
    ] = {}  # (docstring_hash, verbosity) -> filtered_result

    @staticmethod
    def _get_docstring_hash(docstring: str) -> str:
        """Get a hash for the docstring for caching purposes"""
        return str(hash(docstring))

    @classmethod
    def _validate_tag_closure(cls, docstring: str) -> None:
        """
        Validate that all opening tags have corresponding closing tags.

        Args:
            docstring: The docstring to validate

        Raises:
            MalformedDocstringError: If unclosed tags are found
        """
        # Find all opening tags
        opening_pattern = rf"\[({cls._keyword_pattern})\]"
        opening_tags = re.findall(opening_pattern, docstring, re.IGNORECASE)

        # For each unique tag, count openings and closings
        unclosed = []
        checked_tags = set()

        for tag in opening_tags:
            tag_upper = tag.upper()
            if tag_upper in checked_tags:
                continue
            checked_tags.add(tag_upper)

            # Count openings and closings for this specific tag (case-insensitive)
            opening_count = len(
                re.findall(rf"\[{re.escape(tag)}\]", docstring, re.IGNORECASE)
            )
            closing_count = len(
                re.findall(rf"\[/{re.escape(tag)}\]", docstring, re.IGNORECASE)
            )

            if opening_count > closing_count:
                unclosed.append(
                    f"{tag_upper} (opened {opening_count} times, closed {closing_count} times)"
                )

        if unclosed:
            message = f"Docstring has unclosed tags: {', '.join(unclosed)}"
            raise MalformedDocstringError(message)

    @classmethod
    def _clean_nested_tags(cls, content: str) -> str:
        """Remove any nested [TAG]...[/TAG] patterns and keep just the content"""
        nested_tag_pattern = r"\[([A-Z_]+)\](.*?)\[/\1\]"

        # Keep replacing until no more nested tags found
        while re.search(nested_tag_pattern, content, re.DOTALL):
            content = re.sub(nested_tag_pattern, r"\2", content, flags=re.DOTALL)

        return content.strip()

    @classmethod
    def extract_all_sections(cls, docstring: str) -> dict[str, str]:
        """Extract all tagged sections and standard sections from docstring"""
        if not docstring:
            return {}

        # Check cache first
        doc_hash = cls._get_docstring_hash(docstring)
        if doc_hash in cls._parsed_cache:
            return cls._parsed_cache[doc_hash].copy()

        sections = {}

        try:
            # Validate tag closure before processing
            cls._validate_tag_closure(docstring)

            # Find all tagged sections using pre-compiled regex
            matches = cls._keyword_regex.findall(docstring)
            for section_name, content in matches:
                if content.strip():  # Only add non-empty sections
                    sections[section_name.upper()] = cls._clean_nested_tags(
                        content.strip()
                    )

            # Remove tagged sections to get basic content
            doc_without_tags = cls._keyword_regex.sub("", docstring)

            # Extract standard sections (Args, Returns)
            args_match = cls._args_regex.search(doc_without_tags)
            if args_match:
                sections["ARGS"] = args_match.group(1).strip()
                # Remove Args section from basic content
                doc_without_tags = cls._args_regex.sub("", doc_without_tags)

            returns_match = cls._returns_regex.search(doc_without_tags)
            if returns_match:
                sections["RETURNS"] = returns_match.group(1).strip()
                # Remove Returns section from basic content
                doc_without_tags = cls._returns_regex.sub("", doc_without_tags)

            # Extract basic description (everything that's left)
            basic_content = doc_without_tags.strip()
            if basic_content:
                sections["BASIC"] = basic_content

            # Cache the result
            cls._parsed_cache[doc_hash] = sections.copy()

        except MalformedDocstringError:
            # Re-raise malformed docstring errors without catching them
            raise
        except Exception as e:
            logger.error(f"Error parsing docstring: {e}")
            # Fallback to basic parsing
            sections["BASIC"] = (
                docstring.split("\n")[0].strip()
                if docstring
                else "No description available"
            )

        return sections

    @classmethod
    def get_basic_description(cls, docstring: str) -> str:
        """Extract basic description from any docstring format"""
        if not docstring:
            return "No description available"

        sections = cls.extract_all_sections(docstring)

        # Priority order for basic description: BRIEF > BASIC > first line
        if sections.get("BRIEF"):
            return sections["BRIEF"]
        elif sections.get("BASIC"):
            # Take first paragraph from basic content
            basic_lines = sections["BASIC"].split("\n")
            first_paragraph: list[str] = []
            for line in basic_lines:
                stripped_line = line.strip()
                if (
                    not stripped_line and first_paragraph
                ):  # Stop at first empty line after content
                    break
                if stripped_line:
                    first_paragraph.append(stripped_line)
            return " ".join(first_paragraph) if first_paragraph else sections["BASIC"]
        else:
            # Fallback to first line
            return (
                docstring.split("\n")[0].strip()
                if docstring
                else "No description available"
            )

    @classmethod
    def filter_tool_description(cls, docstring: str, verbosity: ToolVerbosity) -> str:
        """Filter tool description based on verbosity level with caching"""
        if not docstring:
            return "No description available"

        # Check cache first
        cache_key = (cls._get_docstring_hash(docstring), verbosity)
        if cache_key in cls._filtered_cache:
            return cls._filtered_cache[cache_key]

        try:
            # Handle FULL verbosity - return original docstring
            if verbosity == ToolVerbosity.FULL:
                result = docstring
                cls._filtered_cache[cache_key] = result
                return result

            # Handle MINIMAL verbosity - return basic description only
            if verbosity == ToolVerbosity.MINIMAL:
                result = cls.get_basic_description(docstring)
                cls._filtered_cache[cache_key] = result
                return result

            # For other verbosity levels, extract sections and filter
            sections = cls.extract_all_sections(docstring)
            if not sections:
                result = cls.get_basic_description(docstring)
                cls._filtered_cache[cache_key] = result
                return result

            included_sections = VerbosityConfig.get_sections_for_verbosity(verbosity)
            content_parts = []

            # Add sections in priority order, avoiding duplicates
            section_order = [
                "BRIEF",  # Highest priority for brief descriptions
                "BASIC",  # Fallback if no BRIEF
                "DETAILED",
                "PROCEDURAL",
                "CONTEXTUAL",
                "WORKFLOW_INTEGRATION",
                "SYNTACTICAL",
                "EXAMPLES",
                "RAISES",
                "LIMITATIONS",
            ]

            added_sections = set()

            for section_name in section_order:
                if (
                    section_name in included_sections
                    and section_name in sections
                    and section_name not in added_sections
                ):
                    # Skip BASIC if we already have BRIEF (BRIEF takes precedence)
                    if section_name == "BASIC" and "BRIEF" in added_sections:
                        continue

                    # Add section with appropriate formatting
                    content = sections[section_name]
                    if section_name == "PROCEDURAL":
                        content_parts.append(f"When to use: {content}")
                    elif section_name == "WORKFLOW_INTEGRATION":
                        content_parts.append(f"Workflow: {content}")
                    elif section_name == "RAISES":
                        content_parts.append(f"Exceptions: {content}")
                    elif section_name == "LIMITATIONS":
                        content_parts.append(f"Limitations: {content}")
                    else:
                        content_parts.append(content)

                    added_sections.add(section_name)

            # Fallback if no content was added
            if not content_parts:
                result = cls.get_basic_description(docstring)
            else:
                result = "\n\n".join(content_parts)

            # Cache and return result
            cls._filtered_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error filtering tool description: {e}")
            # Fallback to basic description
            result = cls.get_basic_description(docstring)
            cls._filtered_cache[cache_key] = result
            return result

    @classmethod
    def filter_argument_description(
        cls, arg_desc: str, verbosity: ToolVerbosity
    ) -> str:
        """Filter argument description based on verbosity with improved logic"""
        if not arg_desc:
            return arg_desc

        try:
            if verbosity == ToolVerbosity.MINIMAL:
                # Just the basic part before any tags or extra info
                return arg_desc.split("[")[0].split("(choices:")[0].strip()

            else:
                # For DETAILED and above, include tagged sections based on verbosity
                sections = cls.extract_all_sections(arg_desc)
                sections = {
                    k.replace("ARGS_", "") if k.startswith("ARGS_") else k: v
                    for k, v in sections.items()
                }
                included_sections = VerbosityConfig.get_sections_for_verbosity(
                    verbosity
                )

                parts = []

                # Add basic content first (if no BRIEF section exists)
                if "BASIC" in sections and "BRIEF" not in sections:
                    parts.append(sections["BASIC"])

                # Add sections based on verbosity
                parts.extend(
                    [
                        sections[section_name]
                        for section_name in [
                            "BRIEF",
                            "DETAILED",
                            "SYNTACTICAL",
                            "EXAMPLES",
                        ]
                        if section_name in included_sections
                        and section_name in sections
                    ]
                )

                # Keep choices info if present
                if "(choices:" in arg_desc:
                    choices_start = arg_desc.find("(choices:")
                    choices_end = arg_desc.find(")", choices_start)
                    if choices_end != -1:
                        choices_part = arg_desc[choices_start : choices_end + 1]
                        parts.append(choices_part)

                return "\n".join(parts) if parts else arg_desc

        except Exception as e:
            logger.error(f"Error filtering argument description: {e}")
            # Fallback to original description
            return arg_desc

    @classmethod
    def clear_cache(cls):
        """Clear the parsing and filtering caches"""
        cls._parsed_cache.clear()
        cls._filtered_cache.clear()

    @classmethod
    def get_cache_stats(cls) -> dict[str, int]:
        """Get cache statistics for monitoring"""
        return {
            "parsed_cache_size": len(cls._parsed_cache),
            "filtered_cache_size": len(cls._filtered_cache),
        }


def get_tools_guide_with_verbosity(
    env: Environment, verbosity: ToolVerbosity | None
) -> str:
    """Generate tools guide with specified verbosity level"""
    if not env.tools:
        return "No tools available."
    if verbosity is None:
        verbosity = ToolVerbosity.FULL
    tools_descriptions = []

    for tool in env.tools.values():
        # Filter tool description
        filtered_description = VerbosityConfig.filter_tool_description(
            tool.description, verbosity
        )

        # Format arguments based on verbosity
        if verbosity == ToolVerbosity.MINIMAL:
            args_desc = ", ".join(arg.name for arg in tool.arguments)
            tools_descriptions.append(
                f"**{tool.name}**: {filtered_description}\nArguments: {args_desc}"
            )
        else:
            args_desc = []
            properties = tool.params_json_schema.get("properties", {})
            for arg in tool.arguments:
                filtered_arg_desc = VerbosityConfig.filter_argument_description(
                    arg.description, verbosity
                )

                prop = properties.get(arg.name, {})
                type_str = format_json_schema_type(prop) if prop else arg.type

                required = (
                    "required" if arg.required else f"optional, default: {arg.default}"
                )
                args_desc.append(
                    f"- {arg.name} ({type_str}, {required}): {filtered_arg_desc}"
                )

            tool_guide = f"""Tool: {tool.name}
Description: {filtered_description}
Arguments:
{chr(10).join(args_desc)}"""
            tools_descriptions.append(tool_guide)

    tools_guide = "\n\n".join(tools_descriptions)

    # Add usage instructions based on verbosity
    if verbosity == ToolVerbosity.MINIMAL:
        return f"Available Tools:\n{tools_guide}"
    else:
        return f"""Available Tools:
{tools_guide}

How to use tools:
1. Each tool call must specify the tool name and required arguments
2. Tools may return errors if arguments are invalid
3. You can make multiple tool calls as needed
4. All tool calls are recorded and affect your final score

Example tool call format:
{{
    "tool_name": "tool_name",
    "arguments": {{
        "arg1": value1,
        "arg2": value2
    }}
}}
"""
