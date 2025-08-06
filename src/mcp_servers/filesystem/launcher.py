"""File System MCP Server launcher and lifecycle management."""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json


class FileSystemMCPClient:
    """Client for managing File System MCP Server lifecycle and communication."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the MCP client.

        Args:
            project_root: Root directory to constrain file operations to.
        """
        self.project_root = project_root or str(Path.cwd())
        self.process: Optional[subprocess.Popen] = None

    def start_server(self) -> bool:
        """Start the File System MCP Server process.

        Returns:
            True if server started successfully, False otherwise.
        """
        try:
            # Get path to server script
            server_script = Path(__file__).parent / "server.py"

            # Start server process with stdio communication
            self.process = subprocess.Popen(
                [sys.executable, str(server_script), self.project_root],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )

            return True

        except Exception as e:
            print(f"Failed to start File System MCP Server: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the File System MCP Server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC request to the server.

        Args:
            method: Method name to call
            params: Parameters for the method

        Returns:
            Response from server or None if error
        """
        if not self.process:
            return None

        request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            if self.process.stdin:
                self.process.stdin.write(request_json)
                self.process.stdin.flush()

            # Read response
            if self.process.stdout:
                response_line = self.process.stdout.readline()
                if response_line:
                    return json.loads(response_line.strip())

        except Exception as e:
            print(f"Error communicating with server: {e}")

        return None

    def list_tools(self) -> Optional[Dict[str, Any]]:
        """List available tools."""
        return self.send_request("tools/list")

    def call_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a specific tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        return self.send_request("tools/call", {"name": name, "arguments": arguments})

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
