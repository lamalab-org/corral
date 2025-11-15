# Corral Tools Documentation

## Overview

Corral provides a powerful tool system that allows you to create reusable, type-safe tools for AI agents. Tools are the building blocks that agents use to perform actions and gather information.

## Creating Tools

### Basic Tool with `@tool` Decorator

The simplest way to create a tool is using the `@tool` decorator:

```python
from corral.backend.tool import tool


@tool
def calculate_sum(a: int, b: int) -> str:
    """Calculate the sum of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum as a string
    """
    return str(a + b)
```

### Requirements for Tool Functions

Every tool function must have:

1. **Type hints** for all parameters and return value
2. **Complete docstring** with:
   - Brief description
   - `Args:` section documenting each parameter
   - `Returns:` section describing the return value
3. **String return type** (or convertible to string)

### Optional Parameters with Defaults

```python
@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the database with optional result limit.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)

    Returns:
        Search results as formatted string
    """
    # Implementation
    return f"Found {limit} results for: {query}"
```

### Parameters with Choices

You can restrict parameter values to specific choices:

```python
@tool
def analyze_molecule(smiles: str, property_type: str) -> str:
    """Analyze molecular properties.

    Args:
        smiles: SMILES string of the molecule
        property_type: Type of property to analyze (choices: ["mass", "polarity", "reactivity"])

    Returns:
        Analysis results
    """
    # Implementation
    return f"Analyzing {property_type} of {smiles}"
```

The docstring format `(choices: ["option1", "option2"])` automatically creates validation.

### Hidden Arguments

Sometimes you want to fix certain parameters without exposing them to the agent:

```python
@tool(hidden_args=["api_key"])
def query_api(endpoint: str, api_key: str = "secret-key-123") -> str:
    """Query an external API.

    Args:
        endpoint: API endpoint to query

    Returns:
        API response
    """
    # The agent only sees 'endpoint', but both parameters are used
    return f"Querying {endpoint} with key {api_key}"
```

## Tool Class Methods

### `to_mcp(verbosity=None)`

Convert a Corral tool to MCP (Model Context Protocol) format:

```python
from corral.backend.tool import tool
from corral.router.verbosity import ToolVerbosity


@tool
def my_tool(param: str) -> str:
    """Tool description.

    Args:
        param: Parameter description

    Returns:
        Result
    """
    return param


# Convert to MCP format
mcp_def = my_tool.for_mcp()
print(mcp_def)
# {
#     "name": "my_tool",
#     "description": "Tool description.",
#     "inputSchema": {
#         "type": "object",
#         "properties": {
#             "param": {
#                 "type": "string",
#                 "description": "Parameter description"
#             }
#         },
#         "required": ["param"]
#     }
# }

# With specific verbosity level
brief_def = my_tool.for_mcp(verbosity=ToolVerbosity.BRIEF)
workflow_def = my_tool.for_mcp(verbosity=ToolVerbosity.WORKFLOW)
comprehensive_def = my_tool.for_mcp(verbosity=ToolVerbosity.COMPREHENSIVE)
```

**Verbosity Levels:**
- `BRIEF`: Minimal descriptions, just the essentials
- `WORKFLOW`: Moderate detail, suitable for workflow documentation
- `COMPREHENSIVE`: Full detail (default), all information included

### `from_mcp(mcp_tool_definition)`

This method is intentionally not implemented. Corral tools are the source of truth for tool definitions:

```python
from corral.backend.tool import Tool

# This will raise NotImplementedError
try:
    tool = Tool.from_mcp({"name": "test", "description": "test", "inputSchema": {}})
except NotImplementedError as e:
    print(e)  # "Converting from MCP to Corral Tool is not currently supported..."
```

### `validate_arguments(arguments)`

Validate that provided arguments meet tool requirements:

```python
@tool
def process_data(value: int, mode: str) -> str:
    """Process data with specified mode.

    Args:
        value: Integer value to process
        mode: Processing mode (choices: ["fast", "accurate"])

    Returns:
        Processing result
    """
    return f"Processed {value} in {mode} mode"


# Validate arguments
valid, error = process_data.validate_arguments({"value": 42, "mode": "fast"})
print(valid)  # True
print(error)  # None

valid, error = process_data.validate_arguments({"value": "invalid", "mode": "fast"})
print(valid)  # False
print(error)  # "Invalid type for argument value. Expected int"

valid, error = process_data.validate_arguments({"value": 42, "mode": "slow"})
print(valid)  # False
print(error)  # "Invalid value for mode. Must be one of: ['fast', 'accurate']"
```

### `execute(**kwargs)`

Execute the tool with provided arguments:

```python
result = my_tool.execute(param="test")
print(result)  # Returns string result
```

### `get_usage_guide()`

Get a formatted usage guide for the tool:

```python
guide = my_tool.get_usage_guide()
print(guide)
# Tool: my_tool
# Description: Tool description.
# Arguments:
# - param (str, required): Parameter description
```

## Advanced Tool Patterns

### Inheriting from Tool Class

For more complex tools, you can inherit from the `Tool` class directly:

```python
from corral.backend.tool import Tool
from corral.backend.schema import ToolArgument


class DatabaseTool(Tool):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

        super().__init__(
            name="database_query",
            description="Query the database",
            arguments=[
                ToolArgument(
                    name="query",
                    type="str",
                    description="SQL query to execute",
                    required=True,
                )
            ],
        )

    def execute(self, **kwargs) -> str:
        query = kwargs["query"]
        # Execute query using self.connection_string
        return f"Results for: {query}"


# Create instance
db_tool = DatabaseTool("postgresql://localhost/mydb")
```

### Modal Tools (Cloud Execution)

Run computationally expensive operations in the cloud using Modal:

```python
from corral.utils.modal import modal_tool, MODAL_TOOL_REGISTRY
from modal import Image, App

app = App("my-science-app")


@modal_tool(
    app=app,
    image=Image.debian_slim().pip_install("rdkit", "numpy"),
    memory=2048,
    timeout=300,
)
def complex_simulation(parameters: str) -> str:
    """Run complex molecular simulation in the cloud.

    Args:
        parameters: Simulation parameters as JSON string

    Returns:
        Simulation results
    """
    # This code runs in Modal's cloud environment
    import rdkit

    # Perform expensive computation
    return "Simulation complete"


# Access the tool
tool_instance = MODAL_TOOL_REGISTRY["complex_simulation"]
```

## MCP Integration

### Creating an MCP Server

Create a Model Context Protocol server to expose your tools to MCP clients:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPTool, TextContent
import importlib
import inspect
from corral.backend.tool import Tool

# Load tools from a module
module = importlib.import_module("my_domain.tools")
tools = {name: obj for name, obj in inspect.getmembers(module) if isinstance(obj, Tool)}

# Create MCP server
server = Server("my-corral-tools")


@server.list_tools()
async def list_tools() -> list[MCPTool]:
    return [MCPTool(**tool.for_mcp()) for tool in tools.values()]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    tool = tools.get(name)
    if not tool:
        raise ValueError(f"Tool {name} not found")

    # Merge hidden args
    if tool.hidden_args:
        arguments = {**tool.hidden_args, **arguments}

    # Validate
    valid, error = tool.validate_arguments(arguments)
    if not valid:
        raise ValueError(error)

    # Execute
    result = tool.execute(**arguments)
    return [TextContent(type="text", text=str(result))]


# Run server
async def main():
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

For complete MCP integration examples and patterns, see [MCP Migration Guide](MCP_MIGRATION.md).

## Tool Discovery

Tools can be automatically discovered from modules:

```python
import importlib
import inspect
from corral.backend.tool import Tool


def discover_tools(module_name: str) -> dict[str, Tool]:
    """Discover all tools in a module."""
    module = importlib.import_module(module_name)
    return {
        name: obj for name, obj in inspect.getmembers(module) if isinstance(obj, Tool)
    }


# Example usage
tools = discover_tools("retrosynthesis.tools")
for name, tool in tools.items():
    print(f"{name}: {tool.description}")
```

## Best Practices

### 1. Clear Documentation

Always provide clear, comprehensive docstrings:

```python
@tool
def good_tool(molecule: str, temperature: float = 298.15) -> str:
    """Calculate molecular properties at given temperature.

    Computes various thermodynamic properties including enthalpy,
    entropy, and Gibbs free energy using statistical mechanics.

    Args:
        molecule: SMILES string representation of the molecule
        temperature: Temperature in Kelvin (default: 298.15)

    Returns:
        JSON string containing computed properties
    """
    # Implementation
    pass
```

### 2. Type Safety

Always use proper type hints:

```python
from typing import List, Dict, Optional


@tool
def process_molecules(smiles_list: str, options: str = "{}") -> str:
    """Process multiple molecules.

    Args:
        smiles_list: JSON array of SMILES strings
        options: JSON object with processing options (default: {})

    Returns:
        Results as JSON string
    """
    import json

    smiles = json.loads(smiles_list)
    opts = json.loads(options)
    # Process...
    return json.dumps(results)
```

### 3. Error Handling

Provide clear error messages:

```python
@tool
def validate_molecule(smiles: str) -> str:
    """Validate molecular structure.

    Args:
        smiles: SMILES string to validate

    Returns:
        Validation result message
    """
    try:
        # Validation logic
        if not smiles:
            return "Error: SMILES string cannot be empty"
        # More validation...
        return "Valid molecule"
    except Exception as e:
        return f"Validation error: {str(e)}"
```

### 4. Appropriate Choices

Use choices for parameters with limited valid values:

```python
@tool
def set_calculation_method(method: str) -> str:
    """Set quantum chemistry calculation method.

    Args:
        method: Calculation method (choices: ["HF", "DFT", "MP2", "CCSD"])

    Returns:
        Confirmation message
    """
    return f"Method set to {method}"
```

### 5. Hidden Arguments for Configuration

Keep sensitive or fixed configuration out of agent control:

```python
@tool(hidden_args=["api_key", "base_url"])
def query_database(
    query: str, api_key: str = "secret-key", base_url: str = "https://api.example.com"
) -> str:
    """Query remote database.

    Args:
        query: Search query

    Returns:
        Query results
    """
    # Agent only controls 'query', not api_key or base_url
    return f"Querying {base_url} with query: {query}"
```

## Tool Schema

The `ToolArgument` class defines parameter specifications:

```python
from corral.backend.schema import ToolArgument

arg = ToolArgument(
    name="temperature",
    type="float",
    description="Temperature in Kelvin",
    required=False,
    default=298.15,
    choices=None,  # Or list of valid values
)
```

### Supported Types

- `str` - String values
- `int` - Integer values
- `float` - Floating point values
- `bool` - Boolean values
- `list[str]`, `list[int]`, etc. - Lists of typed values
- `dict` - Dictionary/object values

## Testing Tools

Always test your tools before using them with agents:

```python
def test_my_tool():
    # Test with valid arguments
    valid, error = my_tool.validate_arguments({"param": "test"})
    assert valid
    assert error is None

    # Test execution
    result = my_tool.execute(param="test")
    assert isinstance(result, str)

    # Test invalid arguments
    valid, error = my_tool.validate_arguments({"wrong_param": "test"})
    assert not valid
    assert error is not None

    # Test MCP conversion
    mcp_def = my_tool.for_mcp()
    assert "name" in mcp_def
    assert "description" in mcp_def
    assert "inputSchema" in mcp_def
```
