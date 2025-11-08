# Corral MCP Integration

This directory contains the Model Context Protocol (MCP) integration for Corral tools.

## Overview

The MCP integration allows Corral's scientific tools to be exposed as MCP servers, enabling them to be used by any MCP-compatible client such as:

- Claude Desktop
- VS Code with MCP extension
- Custom MCP clients
- Other AI assistants supporting MCP

## Architecture

```
┌─────────────────────────────────────┐
│  MCP Clients                        │
│  (Claude Desktop, VS Code, etc)     │
└──────────────┬──────────────────────┘
               │ JSON-RPC (stdio)
┌──────────────▼──────────────────────┐
│  MCPServer (corral.mcp.mcp_server)  │
│  - Handles MCP protocol             │
│  - Auto-discovers tools             │
│  - Executes tool calls              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  ToolRegistry                       │
│  - Loads domain tools               │
│  - Manages tool instances           │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Domain Tools (@tool decorated)     │
│  tasks/{domain}/tools.py            │
│  (unchanged - existing code)        │
└─────────────────────────────────────┘
```

## Key Components

### 1. ToolRegistry (`tool_registry.py`)

Registry for auto-discovering and managing tools:

- Loads tools from domain modules
- Auto-discovers `Tool` instances from module attributes (created by `@tool` decorator)
- No need for `create_tools()` function

### 2. Schema Converter (`schema_converter.py`)

Converts Corral tool schemas to MCP JSON Schema:

- Maps Python types to JSON Schema types
- Handles complex types (lists, dicts, unions)
- Preserves descriptions, defaults, and constraints

### 3. CLI (`cli.py`)

Command-line interface for running MCP servers:

```bash
python -m corral.mcp.cli retrosynthesis.tools
python -m corral.mcp.cli ml.tools --work-dir /path/to/workspace
```

## Usage

### Option 1: Using the CLI

Run an MCP server for a specific domain:

```bash
# Retrosynthesis tools
python -m corral.mcp.cli retrosynthesis.tools

# With custom work directory
python -m corral.mcp.cli retrosynthesis.tools --work-dir /path/to/workspace

# Verbose logging
python -m corral.mcp.cli retrosynthesis.tools -v
```

### Option 2: Programmatic Use

```python
from corral.mcp import MCPServer

# Create server
server = MCPServer(
    domain_module="retrosynthesis.tools",
    server_name="corral-retrosynthesis",
    work_dir="/path/to/workspace",
)

# Run server
server.run()
```

## Integration with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "corral-retrosynthesis": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/mat-agent-bench",
        "python",
        "-m",
        "corral.mcp.cli",
        "retrosynthesis.tools",
        "--work-dir",
        "/path/to/workspace"
      ]
    },
    "corral-ml": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/mat-agent-bench",
        "python",
        "-m",
        "corral.mcp.cli",
        "ml.tools"
      ]
    }
  }
}
```

Then restart Claude Desktop and the tools will be available!

## Running Multiple Servers

To expose tools from multiple domains simultaneously, run separate server processes:

```bash
# Terminal 1 - Retrosynthesis
python -m corral.mcp.cli retrosynthesis.tools

# Terminal 2 - ML
python -m corral.mcp.cli ml.tools

# Terminal 3 - QA
python -m corral.mcp.cli qa.tools
```

Or configure multiple servers in Claude Desktop config:

```json
{
  "mcpServers": {
    "corral-retrosynthesis": {
      "command": "uv",
      "args": ["run", "python", "-m", "corral.mcp.cli", "retrosynthesis.tools"]
    },
    "corral-ml": {
      "command": "uv",
      "args": ["run", "python", "-m", "corral.mcp.cli", "ml.tools"]
    }
  }
}
```

## Features

### Automatic Tool Discovery

The registry automatically discovers tools from domain modules:

- Auto-discovers all `Tool` instances created by the `@tool` decorator
- No manual registration required
- No need for `create_tools()` function

### Hidden Arguments Support

Tools with `hidden_args` (like `work_dir`) are supported:

- Pass `work_dir` to the server
- Server automatically adds hidden args to tool calls
- Transparent to MCP clients

### Error Handling

Comprehensive error handling:

- Tool not found errors
- Invalid argument errors
- Execution errors
- All errors returned as MCP error responses

### Type Safety

Full type conversion support:

- Python type hints → JSON Schema
- Complex types (lists, dicts, unions)
- Optional arguments
- Default values

## Development

### Testing

Test the schema converter:

```python
from corral.mcp.schema_converter import tool_to_json_schema
from retrosynthesis.tools import search_template_catalog_by_criteria

schema = tool_to_json_schema(search_template_catalog_by_criteria)
print(schema)
```

Test the registry:

```python
from corral.mcp import ToolRegistry

registry = ToolRegistry("retrosynthesis.tools")
print(f"Loaded {len(registry)} tools")
print(registry.list_tool_names())
```

Test a server:

```python
from corral.mcp import MCPServer

server = MCPServer("retrosynthesis.tools")
# Server is ready but not running yet
```

### Logging

Use `-v` flag for verbose logging:

```bash
python -m corral.mcp.cli retrosynthesis.tools -v
```

Or `-q` for quiet mode (errors only):

```bash
python -m corral.mcp.cli retrosynthesis.tools -q
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:

1. You're in the correct directory
2. The domain module is in your Python path
3. All dependencies are installed (`uv sync`)

### Tool Not Found

If tools aren't discovered:

1. Verify tools are decorated with `@tool` decorator
2. Check that tools are defined at module level (not inside classes or functions)
3. Use `-v` flag to see discovery logs

### Execution Errors

If tool execution fails:

1. Check tool arguments are correctly typed
2. Verify `work_dir` is set if needed
3. Review error messages in logs
