"""Pytest configuration for MCP tests.

This module sets up mocks for external dependencies that aren't needed
for most tests, particularly the mcp.server module.
"""

import sys
from unittest.mock import Mock

# Mock MCP dependencies BEFORE any imports
# This must happen at module import time, not in a fixture
sys.modules["mcp.server"] = Mock()
sys.modules["mcp.server.stdio"] = Mock()
sys.modules["mcp.types"] = Mock()

# Provide mock classes that tests might need
sys.modules["mcp.server"].Server = Mock
sys.modules["mcp.types"].TextContent = Mock
sys.modules["mcp.types"].Tool = Mock
