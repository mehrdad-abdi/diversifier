#!/usr/bin/env python3
"""File System MCP Server with stdio transport."""

import ast
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Tool,
    TextContent,
)


class FileSystemMCPServer:
    """MCP Server for file system operations with security constraints."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the File System MCP Server.

        Args:
            project_root: Root directory to constrain file operations to.
                         If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.project_root = self.project_root.resolve()

        # Initialize MCP server
        self.server = Server("filesystem-server")

        # Register tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="read_file",
                    description="Read contents of a file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read",
                            }
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="write_file",
                    description="Write content to a file (creates backup)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file",
                            },
                            "create_backup": {
                                "type": "boolean",
                                "description": "Whether to create a backup before writing",
                                "default": True,
                            },
                        },
                        "required": ["file_path", "content"],
                    },
                ),
                Tool(
                    name="list_files",
                    description="List files in a directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory_path": {
                                "type": "string",
                                "description": "Path to the directory to list",
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Optional glob pattern to filter files",
                                "default": "*",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Whether to list files recursively",
                                "default": False,
                            },
                        },
                        "required": ["directory_path"],
                    },
                ),
                Tool(
                    name="analyze_python_imports",
                    description="Analyze Python file imports using AST",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the Python file to analyze",
                            }
                        },
                        "required": ["file_path"],
                    },
                ),
                Tool(
                    name="find_python_files",
                    description="Find Python files in project with import analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "library_name": {
                                "type": "string",
                                "description": "Optional library name to find files that import it",
                            }
                        },
                    },
                ),
                Tool(
                    name="get_project_structure",
                    description="Get high-level project structure analysis",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "read_file":
                    return await self._read_file(arguments["file_path"])
                elif name == "write_file":
                    return await self._write_file(
                        arguments["file_path"],
                        arguments["content"],
                        arguments.get("create_backup", True),
                    )
                elif name == "list_files":
                    return await self._list_files(
                        arguments["directory_path"],
                        arguments.get("pattern", "*"),
                        arguments.get("recursive", False),
                    )
                elif name == "analyze_python_imports":
                    return await self._analyze_python_imports(arguments["file_path"])
                elif name == "find_python_files":
                    return await self._find_python_files(arguments.get("library_name"))
                elif name == "get_project_structure":
                    return await self._get_project_structure()
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

    async def _read_file(self, file_path: str) -> list[TextContent]:
        """Read contents of a file."""
        path = self._validate_path(file_path)

        if not path.exists():
            return [TextContent(type="text", text=f"File not found: {path}")]

        if not path.is_file():
            return [TextContent(type="text", text=f"Path is not a file: {path}")]

        try:
            content = path.read_text(encoding="utf-8")
            return [TextContent(type="text", text=content)]
        except UnicodeDecodeError:
            return [TextContent(type="text", text=f"File is not text readable: {path}")]

    async def _write_file(
        self, file_path: str, content: str, create_backup: bool = True
    ) -> list[TextContent]:
        """Write content to a file with optional backup."""
        path = self._validate_path(file_path)

        # Create backup if file exists and backup is requested
        if create_backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        path.write_text(content, encoding="utf-8")

        result = f"Successfully wrote to {path}"
        if create_backup and path.exists():
            result += f" (backup created at {backup_path})"

        return [TextContent(type="text", text=result)]

    async def _list_files(
        self, directory_path: str, pattern: str = "*", recursive: bool = False
    ) -> list[TextContent]:
        """List files in a directory."""
        path = self._validate_path(directory_path)

        if not path.exists():
            return [TextContent(type="text", text=f"Directory not found: {path}")]

        if not path.is_dir():
            return [TextContent(type="text", text=f"Path is not a directory: {path}")]

        try:
            if recursive:
                files = list(path.rglob(pattern))
            else:
                files = list(path.glob(pattern))

            # Filter to only files (not directories)
            files = [f for f in files if f.is_file()]

            # Make paths relative to project root
            relative_files = []
            for file in files:
                try:
                    relative_path = file.relative_to(self.project_root)
                    relative_files.append(str(relative_path))
                except ValueError:
                    relative_files.append(str(file))

            result = {
                "directory": str(path.relative_to(self.project_root)),
                "pattern": pattern,
                "recursive": recursive,
                "files": relative_files,
                "count": len(relative_files),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error listing files: {str(e)}")]

    async def _analyze_python_imports(self, file_path: str) -> list[TextContent]:
        """Analyze Python file imports using AST."""
        path = self._validate_path(file_path)

        if not path.exists():
            return [TextContent(type="text", text=f"File not found: {path}")]

        if path.suffix not in [".py", ".pyi"]:
            return [TextContent(type="text", text=f"File is not a Python file: {path}")]

        try:
            content = path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(path))

            imports: Dict[str, Any] = {
                "direct_imports": [],
                "from_imports": [],
                "all_modules": set(),
                "file_path": str(path.relative_to(self.project_root)),
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports["direct_imports"].append(
                            {
                                "module": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno,
                            }
                        )
                        imports["all_modules"].add(alias.name.split(".")[0])

                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    for alias in node.names:
                        imports["from_imports"].append(
                            {
                                "module": module_name,
                                "name": alias.name,
                                "alias": alias.asname,
                                "level": node.level,
                                "line": node.lineno,
                            }
                        )
                        if module_name:
                            imports["all_modules"].add(module_name.split(".")[0])

            # Convert set to sorted list for JSON serialization
            imports["all_modules"] = sorted(list(imports["all_modules"]))

            return [TextContent(type="text", text=json.dumps(imports, indent=2))]

        except SyntaxError as e:
            return [TextContent(type="text", text=f"Syntax error in Python file: {e}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error analyzing imports: {str(e)}")]

    async def _find_python_files(
        self, library_name: Optional[str] = None
    ) -> list[TextContent]:
        """Find Python files in project with optional import filtering."""
        python_files = list(self.project_root.rglob("*.py"))

        results: Dict[str, Any] = {
            "project_root": str(self.project_root),
            "total_python_files": len(python_files),
            "files": [],
        }

        if library_name:
            results["filter_library"] = library_name
            results["matching_files"] = []

        for py_file in python_files:
            try:
                relative_path = py_file.relative_to(self.project_root)
                file_info = {"path": str(relative_path)}

                # If filtering by library, check imports
                if library_name:
                    content = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(content, filename=str(py_file))

                    imports_library = False
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name == library_name or alias.name.startswith(
                                    f"{library_name}."
                                ):
                                    imports_library = True
                                    break
                        elif isinstance(node, ast.ImportFrom):
                            if node.module == library_name or (
                                node.module
                                and node.module.startswith(f"{library_name}.")
                            ):
                                imports_library = True
                                break

                        if imports_library:
                            break

                    if imports_library:
                        results["matching_files"].append(file_info)

                results["files"].append(file_info)

            except Exception as e:
                file_info["error"] = str(e)
                results["files"].append(file_info)

        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    async def _get_project_structure(self) -> list[TextContent]:
        """Get high-level project structure analysis."""
        structure: Dict[str, Any] = {
            "project_root": str(self.project_root),
            "python_files": [],
            "directories": [],
            "config_files": [],
            "total_files": 0,
        }

        # Common config file patterns
        config_patterns = [
            "*.toml",
            "*.yaml",
            "*.yml",
            "*.json",
            "*.cfg",
            "*.ini",
            "requirements*.txt",
            "Dockerfile*",
            "docker-compose*",
            "Makefile",
            ".gitignore",
            "README*",
        ]

        try:
            # Get all files and directories
            all_items = list(self.project_root.rglob("*"))

            for item in all_items:
                try:
                    relative_path = item.relative_to(self.project_root)

                    if item.is_file():
                        structure["total_files"] += 1

                        if item.suffix == ".py":
                            structure["python_files"].append(str(relative_path))

                        # Check if it matches config patterns
                        for pattern in config_patterns:
                            if item.match(pattern) or item.name.startswith("."):
                                structure["config_files"].append(str(relative_path))
                                break

                    elif item.is_dir():
                        structure["directories"].append(str(relative_path))

                except ValueError:
                    # Skip files outside project root
                    continue
                except Exception:
                    # Skip files we can't process
                    continue

            # Sort lists for consistent output
            structure["python_files"].sort()
            structure["directories"].sort()
            structure["config_files"].sort()

            # Add summary stats
            structure["summary"] = {
                "python_files_count": len(structure["python_files"]),
                "directories_count": len(structure["directories"]),
                "config_files_count": len(structure["config_files"]),
                "total_files": structure["total_files"],
            }

        except Exception as e:
            structure["error"] = f"Error analyzing project structure: {str(e)}"

        return [TextContent(type="text", text=json.dumps(structure, indent=2))]

    async def run(self) -> None:
        """Run the MCP server with stdio transport."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="filesystem-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def main():
    """Main entry point for the File System MCP Server."""
    import asyncio
    import sys

    # Get project root from command line argument if provided
    project_root = sys.argv[1] if len(sys.argv) > 1 else None

    server = FileSystemMCPServer(project_root=project_root)

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
