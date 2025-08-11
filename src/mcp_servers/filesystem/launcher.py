"""File System MCP Server launcher and lifecycle management."""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any
from ..base_client import BaseMCPClient


class FileSystemMCPClient(BaseMCPClient):
    """Client for managing File System MCP Server lifecycle and communication."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Filesystem MCP client.

        Args:
            project_root: Root directory to constrain file operations to.
        """
        super().__init__(project_root, "Filesystem MCP Server")
    
    def _get_server_script_path(self) -> Path:
        """Get the path to the Filesystem server script."""
        return Path(__file__).parent / "server.py"

    def read_file(self, file_path: str) -> Optional[str]:
        """Read a file using the MCP server.

        Args:
            file_path: Path to file to read

        Returns:
            File content or None if error
        """
        result = self.call_tool("read_file", {"file_path": file_path})
        if result and "result" in result and result["result"]:
            return result["result"][0].get("text")
        return None

    def write_file(
        self, file_path: str, content: str, create_backup: bool = True
    ) -> bool:
        """Write a file using the MCP server.

        Args:
            file_path: Path to file to write
            content: Content to write
            create_backup: Whether to create backup

        Returns:
            True if successful, False otherwise
        """
        result = self.call_tool(
            "write_file",
            {
                "file_path": file_path,
                "content": content,
                "create_backup": create_backup,
            },
        )
        return result is not None and "error" not in result

    def analyze_python_imports(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze Python imports in a file.

        Args:
            file_path: Path to Python file

        Returns:
            Import analysis results
        """
        result = self.call_tool("analyze_python_imports", {"file_path": file_path})
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def find_python_files(
        self, library_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find Python files in the project.

        Args:
            library_name: Optional library name to filter by

        Returns:
            File search results
        """
        params = {}
        if library_name:
            params["library_name"] = library_name

        result = self.call_tool("find_python_files", params)
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def __enter__(self):
        """Context manager entry."""
        if self.start_server():
            return self
        else:
            raise RuntimeError("Failed to start File System MCP Server")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()


# Example usage
async def example_usage():
    """Example of how to use the File System MCP Client."""

    with FileSystemMCPClient() as client:
        # List available tools
        tools = client.list_tools()
        print("Available tools:", tools)

        # Read a file
        content = client.read_file("src/cli.py")
        if content:
            print(f"File content length: {len(content)}")

        # Analyze Python imports
        imports = client.analyze_python_imports("src/cli.py")
        if imports:
            print("Imports found:", imports.get("all_modules", []))

        # Find Python files that import a specific library
        files = client.find_python_files("argparse")
        if files:
            print(f"Files importing argparse: {len(files.get('matching_files', []))}")


if __name__ == "__main__":
    asyncio.run(example_usage())
