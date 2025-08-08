"""MCP server connection management for orchestration."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

from src.mcp_servers.filesystem.launcher import FileSystemMCPClient
from src.mcp_servers.testing.launcher import TestingMCPClient
from src.mcp_servers.git.launcher import GitMCPClient
from src.mcp_servers.docker.launcher import DockerMCPLauncher


class MCPServerType(Enum):
    """Types of MCP servers used in diversification."""

    FILESYSTEM = "filesystem"
    TESTING = "testing"
    GIT = "git"
    DOCKER = "docker"


class MCPConnection:
    """Represents a connection to an MCP server."""

    def __init__(self, server_type: MCPServerType, client: Any):
        """Initialize MCP connection.

        Args:
            server_type: Type of MCP server
            client: MCP client instance
        """
        self.server_type = server_type
        self.client = client
        self.is_connected = False
        self.logger = logging.getLogger(f"diversifier.mcp.{server_type.value}")

    async def connect(self) -> bool:
        """Connect to the MCP server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if hasattr(self.client, "start_server"):
                success = self.client.start_server()
                self.is_connected = success
                if success:
                    self.logger.info(
                        f"Connected to {self.server_type.value} MCP server"
                    )
                else:
                    self.logger.error(
                        f"Failed to connect to {self.server_type.value} MCP server"
                    )
                return success
            else:
                self.logger.warning(
                    f"Client for {self.server_type.value} does not support connection"
                )
                return False

        except Exception as e:
            self.logger.error(
                f"Error connecting to {self.server_type.value} server: {e}"
            )
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            if hasattr(self.client, "stop_server"):
                self.client.stop_server()
                self.is_connected = False
                self.logger.info(
                    f"Disconnected from {self.server_type.value} MCP server"
                )
        except Exception as e:
            self.logger.error(
                f"Error disconnecting from {self.server_type.value} server: {e}"
            )

    def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool result or None if error
        """
        if not self.is_connected:
            self.logger.error(
                f"Cannot call tool on disconnected {self.server_type.value} server"
            )
            return None

        try:
            if hasattr(self.client, "call_tool"):
                result = self.client.call_tool(tool_name, arguments)
                self.logger.debug(
                    f"Called {tool_name} on {self.server_type.value} server"
                )
                return result
            else:
                self.logger.error(
                    f"Client for {self.server_type.value} does not support tool calling"
                )
                return None

        except Exception as e:
            self.logger.error(
                f"Error calling tool {tool_name} on {self.server_type.value} server: {e}"
            )
            return None

    def list_tools(self) -> Optional[List[str]]:
        """List available tools on the MCP server.

        Returns:
            List of available tools or None if error
        """
        if not self.is_connected:
            self.logger.error(
                f"Cannot list tools on disconnected {self.server_type.value} server"
            )
            return None

        try:
            if hasattr(self.client, "list_tools"):
                result = self.client.list_tools()
                if result and "result" in result:
                    # Extract tool names from the result
                    tools = [tool.get("name", "") for tool in result.get("result", [])]
                    return tools
                return []
            else:
                self.logger.error(
                    f"Client for {self.server_type.value} does not support tool listing"
                )
                return None

        except Exception as e:
            self.logger.error(
                f"Error listing tools on {self.server_type.value} server: {e}"
            )
            return None


class MCPManager:
    """Manager for coordinating MCP server connections."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the MCP manager.

        Args:
            project_root: Root directory for file operations
        """
        self.project_root = project_root or str(Path.cwd())
        self.connections: Dict[MCPServerType, MCPConnection] = {}
        self.logger = logging.getLogger("diversifier.mcp_manager")

    async def initialize_filesystem_server(self) -> bool:
        """Initialize the filesystem MCP server.

        Returns:
            True if successful, False otherwise
        """
        try:
            client = FileSystemMCPClient(project_root=self.project_root)
            connection = MCPConnection(MCPServerType.FILESYSTEM, client)

            success = await connection.connect()
            if success:
                self.connections[MCPServerType.FILESYSTEM] = connection
                self.logger.info("Filesystem MCP server initialized")

            return success

        except Exception as e:
            self.logger.error(f"Failed to initialize filesystem server: {e}")
            return False

    async def initialize_testing_server(self) -> bool:
        """Initialize the testing MCP server.

        Returns:
            True if successful, False otherwise
        """
        try:
            client = TestingMCPClient(project_root=self.project_root)
            connection = MCPConnection(MCPServerType.TESTING, client)

            success = await connection.connect()
            if success:
                self.connections[MCPServerType.TESTING] = connection
                self.logger.info("Testing MCP server initialized")

            return success

        except Exception as e:
            self.logger.error(f"Failed to initialize testing server: {e}")
            return False

    async def initialize_git_server(self) -> bool:
        """Initialize the git MCP server.

        Returns:
            True if successful, False otherwise
        """
        try:
            client = GitMCPClient(project_root=self.project_root)
            connection = MCPConnection(MCPServerType.GIT, client)

            success = await connection.connect()
            if success:
                self.connections[MCPServerType.GIT] = connection
                self.logger.info("Git MCP server initialized")

            return success

        except Exception as e:
            self.logger.error(f"Failed to initialize git server: {e}")
            return False

    async def initialize_docker_server(self) -> bool:
        """Initialize the docker MCP server.

        Returns:
            True if successful, False otherwise
        """
        try:
            client = DockerMCPLauncher(project_root=self.project_root)
            connection = MCPConnection(MCPServerType.DOCKER, client)

            success = await connection.connect()
            if success:
                self.connections[MCPServerType.DOCKER] = connection
                self.logger.info("Docker MCP server initialized")

            return success

        except Exception as e:
            self.logger.error(f"Failed to initialize docker server: {e}")
            return False

    async def initialize_all_servers(self) -> Dict[MCPServerType, bool]:
        """Initialize all MCP servers.

        Returns:
            Dictionary mapping server types to initialization success
        """
        results = {}

        # Initialize filesystem server (implemented)
        results[MCPServerType.FILESYSTEM] = await self.initialize_filesystem_server()

        # Initialize other servers
        results[MCPServerType.TESTING] = await self.initialize_testing_server()
        results[MCPServerType.GIT] = await self.initialize_git_server()
        results[MCPServerType.DOCKER] = await self.initialize_docker_server()

        successful_count = sum(1 for success in results.values() if success)
        self.logger.info(f"Initialized {successful_count}/{len(results)} MCP servers")

        return results

    def get_connection(self, server_type: MCPServerType) -> Optional[MCPConnection]:
        """Get MCP connection for a specific server type.

        Args:
            server_type: Type of MCP server

        Returns:
            MCP connection or None if not available
        """
        connection = self.connections.get(server_type)
        if connection and connection.is_connected:
            return connection
        else:
            self.logger.warning(f"No active connection for {server_type.value} server")
            return None

    def is_server_available(self, server_type: MCPServerType) -> bool:
        """Check if a specific MCP server is available.

        Args:
            server_type: Type of MCP server

        Returns:
            True if server is available and connected
        """
        connection = self.connections.get(server_type)
        return connection is not None and connection.is_connected

    def get_available_servers(self) -> List[MCPServerType]:
        """Get list of available MCP servers.

        Returns:
            List of available server types
        """
        available = []
        for server_type, connection in self.connections.items():
            if connection.is_connected:
                available.append(server_type)
        return available

    async def call_tool(
        self, server_type: MCPServerType, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a tool on a specific MCP server.

        Args:
            server_type: Type of MCP server
            tool_name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool result or None if error
        """
        connection = self.get_connection(server_type)
        if connection:
            return connection.call_tool(tool_name, arguments)
        return None

    async def list_tools(self, server_type: MCPServerType) -> Optional[List[str]]:
        """List available tools on a specific MCP server.

        Args:
            server_type: Type of MCP server

        Returns:
            List of available tools or None if error
        """
        connection = self.get_connection(server_type)
        if connection:
            return connection.list_tools()
        return None

    async def shutdown_all_servers(self) -> None:
        """Shutdown all MCP server connections."""
        for server_type, connection in self.connections.items():
            if connection.is_connected:
                await connection.disconnect()
                self.logger.info(f"Shutdown {server_type.value} server")

        self.connections.clear()
        self.logger.info("All MCP servers shutdown")

    async def health_check(self) -> Dict[MCPServerType, bool]:
        """Perform health check on all MCP servers.

        Returns:
            Dictionary mapping server types to health status
        """
        health_status = {}

        for server_type in MCPServerType:
            connection = self.connections.get(server_type)
            if connection:
                # Try to list tools as a basic health check
                tools = connection.list_tools()
                health_status[server_type] = tools is not None
            else:
                health_status[server_type] = False

        return health_status

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.shutdown_all_servers())
