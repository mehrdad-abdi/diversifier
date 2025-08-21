"""MCP server connection management for orchestration."""

import asyncio
import logging
import subprocess
import time
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path

from src.mcp_servers.filesystem.launcher import FileSystemMCPClient
from src.mcp_servers.command.client import CommandMCPClient
from src.mcp_servers.git.launcher import GitMCPClient
from src.mcp_servers.docker.launcher import DockerMCPLauncher


class MCPServerType(Enum):
    """Types of MCP servers used in diversification."""

    FILESYSTEM = "filesystem"
    COMMAND = "command"
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
        self.last_health_check = 0.0
        self.restart_count = 0
        self.max_restarts = 3
        self.restart_backoff = 1.0
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
                    self.last_health_check = time.time()
                    self.restart_count = 0
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
                if result is not None:
                    self.last_health_check = time.time()
                    self.logger.debug(
                        f"Called {tool_name} on {self.server_type.value} server"
                    )
                    return result
                else:
                    self.logger.warning(
                        f"Tool {tool_name} returned None on {self.server_type.value} server"
                    )
                    return None
            else:
                self.logger.error(
                    f"Client for {self.server_type.value} does not support tool calling"
                )
                return None

        except Exception as e:
            self.logger.error(
                f"Error calling tool {tool_name} on {self.server_type.value} server: {e}"
            )
            self.is_connected = False
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
                    self.last_health_check = time.time()
                    # Extract tool names from the result
                    tools = [tool.get("name", "") for tool in result.get("result", [])]
                    return tools
                self.last_health_check = time.time()
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
            self.is_connected = False
            return None

    def is_healthy(self, health_check_interval: float = 30.0) -> bool:
        """Check if the server is healthy.

        Args:
            health_check_interval: Minimum seconds between health checks

        Returns:
            True if server is healthy, False otherwise
        """
        if not self.is_connected:
            return False

        current_time = time.time()
        if current_time - self.last_health_check < health_check_interval:
            return True

        try:
            tools = self.list_tools()
            return tools is not None
        except Exception as e:
            self.logger.error(
                f"Health check failed for {self.server_type.value} server: {e}"
            )
            self.is_connected = False
            return False

    async def restart(self) -> bool:
        """Restart the MCP server connection.

        Returns:
            True if restart successful, False otherwise
        """
        if self.restart_count >= self.max_restarts:
            self.logger.error(
                f"Max restart attempts ({self.max_restarts}) reached for {self.server_type.value} server"
            )
            return False

        self.logger.info(
            f"Restarting {self.server_type.value} server (attempt {self.restart_count + 1})"
        )

        await self.disconnect()

        if self.restart_count > 0:
            backoff_time = self.restart_backoff * (2 ** (self.restart_count - 1))
            self.logger.info(
                f"Waiting {backoff_time:.1f}s before restart attempt {self.restart_count + 1}"
            )
            await asyncio.sleep(backoff_time)

        self.restart_count += 1
        success = await self.connect()

        if success:
            self.logger.info(f"Successfully restarted {self.server_type.value} server")
            # Note: restart_count is reset to 0 in connect() on success
        else:
            self.logger.error(f"Failed to restart {self.server_type.value} server")

        return success


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

    async def initialize_command_server(self) -> bool:
        """Initialize the command MCP server.

        Returns:
            True if successful, False otherwise
        """
        try:
            client = CommandMCPClient(project_root=self.project_root)
            connection = MCPConnection(MCPServerType.COMMAND, client)

            success = await connection.connect()
            if success:
                self.connections[MCPServerType.COMMAND] = connection
                self.logger.info("Command MCP server initialized")

            return success

        except Exception as e:
            self.logger.error(f"Failed to initialize command server: {e}")
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
        results[MCPServerType.COMMAND] = await self.initialize_command_server()
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
            result = connection.call_tool(tool_name, arguments)
            if result is None and connection.is_connected:
                self.logger.warning(
                    f"Tool call returned None but server is still connected: {server_type.value}"
                )
            return result
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

    async def health_check(
        self, detailed: bool = False
    ) -> Dict[MCPServerType, Union[bool, Dict[str, Any]]]:
        """Perform health check on all MCP servers.

        Args:
            detailed: Return detailed health information

        Returns:
            Dictionary mapping server types to health status or detailed info
        """
        health_status: Dict[MCPServerType, Union[bool, Dict[str, Any]]] = {}

        for server_type in MCPServerType:
            connection = self.connections.get(server_type)
            if connection:
                is_healthy = connection.is_healthy()
                if detailed:
                    health_status[server_type] = {
                        "healthy": is_healthy,
                        "connected": connection.is_connected,
                        "last_check": connection.last_health_check,
                        "restart_count": connection.restart_count,
                        "max_restarts": connection.max_restarts,
                    }
                else:
                    health_status[server_type] = is_healthy
            else:
                if detailed:
                    health_status[server_type] = {
                        "healthy": False,
                        "connected": False,
                        "last_check": 0.0,
                        "restart_count": 0,
                        "max_restarts": 3,
                    }
                else:
                    health_status[server_type] = False

        return health_status

    async def restart_failed_servers(self) -> Dict[MCPServerType, bool]:
        """Restart any failed servers.

        Returns:
            Dictionary mapping server types to restart success status
        """
        restart_results = {}

        for server_type, connection in self.connections.items():
            if not connection.is_healthy():
                self.logger.info(
                    f"Attempting to restart failed {server_type.value} server"
                )
                restart_results[server_type] = await connection.restart()
            else:
                restart_results[server_type] = True

        return restart_results

    async def monitor_servers(
        self, interval: float = 60.0, max_iterations: int = -1
    ) -> None:
        """Monitor server health and restart failed servers.

        Args:
            interval: Monitoring interval in seconds
            max_iterations: Max monitoring cycles (-1 for infinite)
        """
        iteration = 0

        while max_iterations == -1 or iteration < max_iterations:
            try:
                health_status = await self.health_check()
                failed_servers = [t for t, h in health_status.items() if not h]

                if failed_servers:
                    self.logger.warning(
                        f"Failed servers detected: {[t.value for t in failed_servers]}"
                    )
                    await self.restart_failed_servers()

                await asyncio.sleep(interval)
                iteration += 1

            except Exception as e:
                self.logger.error(f"Error in server monitoring: {e}")
                await asyncio.sleep(interval)
                iteration += 1

    async def get_server_statistics(self) -> Dict[str, Any]:
        """Get statistics about server connections.

        Returns:
            Dictionary with server connection statistics
        """
        connected_count = len([c for c in self.connections.values() if c.is_connected])
        stats: Dict[str, Any] = {
            "total_servers": len(MCPServerType),
            "connected_servers": connected_count,
            "healthy_servers": 0,
            "failed_servers": 0,
            "servers": {},
        }

        health_status = await self.health_check(detailed=True)

        for server_type, health_info in health_status.items():
            if isinstance(health_info, dict):
                if health_info["healthy"]:
                    stats["healthy_servers"] += 1
                else:
                    stats["failed_servers"] += 1

                stats["servers"][server_type.value] = health_info
            else:
                if health_info:
                    stats["healthy_servers"] += 1
                else:
                    stats["failed_servers"] += 1

                stats["servers"][server_type.value] = {"healthy": health_info}

        return stats

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.shutdown_all_servers())

    def emergency_shutdown(self) -> None:
        """Emergency synchronous shutdown for signal handlers.

        This method forcefully terminates all MCP server processes without
        waiting for graceful disconnection. Use only when normal async
        shutdown is not possible (e.g., in signal handlers).
        """
        for server_type, connection in list(self.connections.items()):
            if connection.is_connected and hasattr(connection, "client"):
                client = connection.client
                if hasattr(client, "process") and client.process:
                    try:
                        # Terminate the process immediately
                        client.process.terminate()
                        try:
                            client.process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            # If terminate doesn't work, force kill
                            client.process.kill()
                        self.logger.info(
                            f"Emergency shutdown of {server_type.value} server"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error during emergency shutdown of {server_type.value}: {e}"
                        )

        # Clear all connections
        self.connections.clear()
        self.logger.info("Emergency shutdown completed")
