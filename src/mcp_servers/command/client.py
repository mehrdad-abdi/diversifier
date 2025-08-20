#!/usr/bin/env python3
"""Command MCP Client for general command execution."""

from pathlib import Path
from typing import Dict, Any, Optional

from ..base_client import BaseMCPClient


class CommandMCPClient(BaseMCPClient):
    """MCP Client for command execution operations."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Command MCP Client."""
        super().__init__(project_root, "command-server")

    def _get_server_script_path(self) -> Path:
        """Get the path to the command server script."""
        return Path(__file__).parent / "server.py"

    async def execute_command(self, command: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Execute a command."""
        return self.call_tool("execute_command", {"command": command, **kwargs})

    async def check_file_exists(self, path: str) -> Optional[Dict[str, Any]]:
        """Check if a file exists."""
        return self.call_tool("check_file_exists", {"path": path})

    async def list_directory(
        self, path: str = ".", include_hidden: bool = False
    ) -> Optional[Dict[str, Any]]:
        """List directory contents."""
        return self.call_tool(
            "list_directory", {"path": path, "include_hidden": include_hidden}
        )

    async def read_file_content(
        self, path: str, max_lines: int = 1000
    ) -> Optional[Dict[str, Any]]:
        """Read file content."""
        return self.call_tool(
            "read_file_content", {"path": path, "max_lines": max_lines}
        )

    async def find_files(
        self, pattern: str, directory: str = "."
    ) -> Optional[Dict[str, Any]]:
        """Find files matching a pattern."""
        return self.call_tool(
            "find_files", {"pattern": pattern, "directory": directory}
        )
