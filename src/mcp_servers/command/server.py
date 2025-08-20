#!/usr/bin/env python3
"""Command MCP Server with stdio transport for general command execution."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional
from mcp.server.stdio import stdio_server

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Tool,
    TextContent,
)


class CommandMCPServer:
    """MCP Server for general command execution with security constraints."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Command MCP Server.

        Args:
            project_root: Root directory to constrain command operations to.
                         If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.project_root = self.project_root.resolve()

        # Initialize MCP server
        self.server = Server("command-server")

        # Register tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="execute_command",
                    description="Execute a shell command in the project directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute",
                            },
                            "working_directory": {
                                "type": "string",
                                "description": "Working directory relative to project root (optional)",
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Command timeout in seconds (default: 300)",
                                "default": 300,
                            },
                            "capture_output": {
                                "type": "boolean",
                                "description": "Whether to capture command output",
                                "default": True,
                            },
                        },
                        "required": ["command"],
                    },
                ),
                Tool(
                    name="find_files",
                    description="Find files matching a pattern",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "File name pattern (e.g., '*.py', '*.toml')",
                            },
                            "directory": {
                                "type": "string",
                                "description": "Directory to search in (relative to project root, default: '.')",
                                "default": ".",
                            },
                        },
                        "required": ["pattern"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "execute_command":
                    return await self._execute_command(
                        arguments["command"],
                        arguments.get("working_directory"),
                        arguments.get("timeout", 300),
                        arguments.get("capture_output", True),
                    )
                elif name == "find_files":
                    return await self._find_files(
                        arguments["pattern"],
                        arguments.get("directory", "."),
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _validate_path(self, file_path: str) -> Path:
        """Validate that path is within project boundaries.

        Args:
            file_path: Path to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is outside project boundaries
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path

        path = path.resolve()

        # Ensure path is within project root
        if not str(path).startswith(str(self.project_root)):
            raise ValueError(f"Path {path} is outside project boundaries")

        return path

    async def _execute_command(
        self,
        command: str,
        working_directory: Optional[str],
        timeout: float,
        capture_output: bool,
    ) -> list[TextContent]:
        """Execute a shell command."""

        # Determine working directory
        if working_directory:
            work_dir = self._validate_path(working_directory)
            if not work_dir.is_dir():
                raise ValueError(f"Working directory {work_dir} does not exist")
        else:
            work_dir = self.project_root

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )

            command_result = {
                "command": command,
                "working_directory": str(work_dir.relative_to(self.project_root)),
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }

            if capture_output:
                command_result["stdout"] = result.stdout
                command_result["stderr"] = result.stderr

            return [TextContent(type="text", text=json.dumps(command_result, indent=2))]

        except subprocess.TimeoutExpired:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "command": command,
                            "error": f"Command timed out after {timeout} seconds",
                            "exit_code": -1,
                            "success": False,
                        },
                        indent=2,
                    ),
                )
            ]
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "command": command,
                            "error": str(e),
                            "exit_code": -1,
                            "success": False,
                        },
                        indent=2,
                    ),
                )
            ]

    async def _find_files(self, pattern: str, directory: str) -> list[TextContent]:
        """Find files matching a pattern."""
        try:
            search_dir = self._validate_path(directory)

            if not search_dir.exists():
                raise ValueError(f"Directory {directory} does not exist")

            if not search_dir.is_dir():
                raise ValueError(f"Path {directory} is not a directory")

            # Use glob pattern matching
            matching_files = []
            for file_path in search_dir.rglob(pattern):
                if file_path.is_file():
                    matching_files.append(
                        {
                            "path": str(file_path.relative_to(self.project_root)),
                            "name": file_path.name,
                            "directory": str(
                                file_path.parent.relative_to(self.project_root)
                            ),
                        }
                    )

            # Sort by path
            matching_files.sort(key=lambda x: x["path"])

            result = {
                "pattern": pattern,
                "search_directory": str(search_dir.relative_to(self.project_root)),
                "matches": matching_files,
                "total_matches": len(matching_files),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error finding files: {str(e)}")]

    async def run(self) -> None:
        """Run the MCP server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="command-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def main():
    """Main entry point for the Command MCP Server."""
    # Get project root from command line argument if provided
    project_root = sys.argv[1] if len(sys.argv) > 1 else None

    server = CommandMCPServer(project_root=project_root)

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
