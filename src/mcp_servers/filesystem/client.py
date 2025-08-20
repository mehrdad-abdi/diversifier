#!/usr/bin/env python3
"""Filesystem MCP Client for file operations."""

from pathlib import Path
from typing import Dict, Any, Optional

from ..base_client import BaseMCPClient


class FilesystemMCPClient(BaseMCPClient):
    """MCP Client for filesystem operations."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Filesystem MCP Client."""
        super().__init__(project_root, "filesystem-server")

    def _get_server_script_path(self) -> Path:
        """Get the path to the filesystem server script."""
        return Path(__file__).parent / "server.py"

    async def read_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Read a file."""
        return self.call_tool("read_file", {"path": path})

    async def write_file(self, path: str, content: str) -> Optional[Dict[str, Any]]:
        """Write a file."""
        return self.call_tool("write_file", {"path": path, "content": content})

    async def list_files(self, path: str = ".") -> Optional[Dict[str, Any]]:
        """List files in a directory."""
        return self.call_tool("list_files", {"path": path})
