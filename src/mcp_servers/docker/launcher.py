"""Docker MCP Server launcher with subprocess management."""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any


class DockerMCPLauncher:
    """Launcher for Docker MCP Server with stdio communication."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Docker MCP Server launcher.

        Args:
            project_root: Root directory for Docker operations.
                         If None, uses current working directory.
        """
        self.project_root = project_root or str(Path.cwd())
        self.process: Optional[subprocess.Popen] = None

    def start_server(self) -> bool:
        """Start the Docker MCP Server as a subprocess.

        Returns:
            True if server started successfully, False otherwise.
        """
        try:
            server_module = Path(__file__).parent / "server.py"

            # Start server process
            cmd = [sys.executable, str(server_module)]
            if self.project_root:
                cmd.append(self.project_root)

            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for real-time communication
            )

            return True

        except Exception as e:
            print(f"Failed to start Docker MCP Server: {e}")
            return False

    def start(self) -> subprocess.Popen:
        """Start the Docker MCP Server as a subprocess (legacy method).

        Returns:
            The subprocess.Popen object for the server process.
        """
        if self.start_server() and self.process:
            return self.process
        raise RuntimeError("Failed to start Docker MCP Server")

    def stop_server(self) -> None:
        """Stop the Docker MCP Server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None

    def stop(self) -> None:
        """Stop the Docker MCP Server process (legacy method)."""
        self.stop_server()

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
            print(f"Error communicating with Docker MCP server: {e}")

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

    def is_running(self) -> bool:
        """Check if the Docker MCP Server process is running."""
        return self.process is not None and self.process.poll() is None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """Main entry point for launching Docker MCP Server."""
    parser = argparse.ArgumentParser(description="Launch Docker MCP Server")
    parser.add_argument("--project-root", help="Root directory for Docker operations")

    args = parser.parse_args()

    launcher = DockerMCPLauncher(project_root=args.project_root)

    try:
        process = launcher.start()
        print(f"Docker MCP Server started with PID: {process.pid}")

        # Keep launcher running until interrupted
        process.wait()

    except KeyboardInterrupt:
        print("\nStopping Docker MCP Server...")
        launcher.stop()
    except Exception as e:
        print(f"Error: {e}")
        launcher.stop()


if __name__ == "__main__":
    main()
