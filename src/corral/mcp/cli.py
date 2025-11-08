"""
CLI for running MCP servers for different Corral domains.

This module provides a command-line interface to start MCP servers for
different scientific domains.
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from corral.mcp.mcp_server import MCPServer


def main() -> None:
    """Main entry point for the MCP server CLI."""
    parser = argparse.ArgumentParser(
        description="Run an MCP server for Corral tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run retrosynthesis MCP server
  python -m corral.mcp.cli retrosynthesis.tools

  # Run with custom work directory
  python -m corral.mcp.cli retrosynthesis.tools --work-dir /path/to/workspace

  # Run with custom server name
  python -m corral.mcp.cli ml.tools --name corral-ml-custom

  # Run for multiple domains (not implemented yet - use multiple processes)
        """,
    )

    parser.add_argument(
        "domain_module",
        type=str,
        help="Domain module containing tools (e.g., 'retrosynthesis.tools', 'ml.tools')",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Server name (default: corral-{domain})",
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for tools (passed to tools with hidden work_dir argument)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all logging except errors",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()  # Remove default handler
    if args.quiet:
        logger.add(sys.stderr, level="ERROR")
    elif args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Validate work_dir if provided
    work_dir = None
    if args.work_dir:
        work_dir_path = Path(args.work_dir)
        if not work_dir_path.exists():
            logger.warning(f"Work directory does not exist: {args.work_dir}")
        work_dir = str(work_dir_path.absolute())

    try:
        # Create and run the server
        server = MCPServer(
            domain_module=args.domain_module,
            server_name=args.name,
            work_dir=work_dir,
        )

        logger.info(f"Starting MCP server: {server.server.name}")
        logger.info(f"Domain module: {args.domain_module}")
        if work_dir:
            logger.info(f"Work directory: {work_dir}")

        # Run the server
        server.run()

    except ImportError as e:
        logger.error(f"Failed to import domain module '{args.domain_module}': {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
