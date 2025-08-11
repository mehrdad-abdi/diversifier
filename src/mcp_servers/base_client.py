"""Base MCP Client with proper initialization handshake."""

import json
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any


class BaseMCPClient(ABC):
    """Base class for MCP clients with proper initialization handshake."""

    def __init__(
        self, project_root: Optional[str] = None, server_name: str = "mcp-server"
    ):
        """Initialize the base MCP client.

        Args:
            project_root: Root directory to constrain operations to
            server_name: Name of the MCP server for logging/identification
        """
        self.project_root = project_root or str(Path.cwd())
        self.server_name = server_name
        self.process: Optional[subprocess.Popen] = None
        self.initialized = False

    @abstractmethod
    def _get_server_script_path(self) -> Path:
        """Get the path to the server script.

        Returns:
            Path to the server script
        """
        pass

    def start_server(self) -> bool:
        """Start the MCP Server process.

        Returns:
            True if server started successfully, False otherwise.
        """
        try:
            # Get path to server script
            server_script = self._get_server_script_path()

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
            print(f"Failed to start {self.server_name}: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the MCP Server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self.initialized = False

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
            print(f"Error communicating with {self.server_name}: {e}")

        return None

    def _initialize_mcp_session(self) -> bool:
        """Send the MCP initialize handshake."""
        if self.initialized:
            return True

        # Send initialize request as per MCP protocol
        initialize_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {
                "name": f"diversifier-{self.server_name.lower()}-client",
                "version": "1.0.0",
            },
        }

        # Send initialize request
        init_response = self.send_request("initialize", initialize_params)
        if init_response and "error" not in init_response:
            # Send initialized notification (no response expected)
            self._send_notification("notifications/initialized")
            self.initialized = True
            return True
        else:
            print(f"MCP initialization failed for {self.server_name}: {init_response}")
            return False

    def _send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process:
            return

        notification: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params

        try:
            notification_json = json.dumps(notification) + "\n"
            if self.process.stdin:
                self.process.stdin.write(notification_json)
                self.process.stdin.flush()
        except Exception as e:
            print(f"Error sending notification to {self.server_name}: {e}")

    def list_tools(self) -> Optional[Dict[str, Any]]:
        """List available tools."""
        # Ensure MCP session is initialized
        if not self._initialize_mcp_session():
            return None
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
        # Ensure MCP session is initialized
        if not self._initialize_mcp_session():
            return None
        return self.send_request("tools/call", {"name": name, "arguments": arguments})

    def __enter__(self):
        """Context manager entry."""
        if self.start_server():
            return self
        else:
            raise RuntimeError(f"Failed to start {self.server_name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()
