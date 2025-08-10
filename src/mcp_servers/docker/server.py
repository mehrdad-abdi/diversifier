#!/usr/bin/env python3
"""Docker MCP Server with stdio transport for container operations."""

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    TextContent,
    Tool,
)

import docker
from docker.errors import APIError, BuildError
from docker.models.containers import Container


class DockerMCPServer:
    """MCP Server for Docker container operations with security constraints."""

    def __init__(self, project_root: str | None = None):
        """Initialize the Docker MCP Server.

        Args:
            project_root: Root directory for project operations.
                         If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.project_root = self.project_root.resolve()

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Docker: {e}")

        # Initialize MCP server
        self.server = Server("docker-server")

        # Keep track of managed containers for cleanup
        self.managed_containers: dict[str, Container] = {}

        # Register tools
        self._register_tools()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _register_tools(self) -> None:
        """Register all available tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="build_image",
                    description="Build Docker image from Dockerfile",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to build context directory",
                            },
                            "tag": {
                                "type": "string",
                                "description": "Tag for the built image",
                            },
                            "dockerfile": {
                                "type": "string",
                                "description": "Path to Dockerfile (relative to context)",
                                "default": "Dockerfile",
                            },
                            "buildargs": {
                                "type": "object",
                                "description": "Build arguments as key-value pairs",
                                "default": {},
                            },
                        },
                        "required": ["path", "tag"],
                    },
                ),
                Tool(
                    name="run_container",
                    description="Run a container with specified configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image": {
                                "type": "string",
                                "description": "Docker image name or ID",
                            },
                            "command": {
                                "type": "string",
                                "description": "Command to run in container",
                            },
                            "environment": {
                                "type": "object",
                                "description": "Environment variables",
                                "default": {},
                            },
                            "volumes": {
                                "type": "object",
                                "description": "Volume mounts as host:container pairs",
                                "default": {},
                            },
                            "working_dir": {
                                "type": "string",
                                "description": "Working directory in container",
                            },
                            "detach": {
                                "type": "boolean",
                                "description": "Run container in background",
                                "default": False,
                            },
                            "remove": {
                                "type": "boolean",
                                "description": "Remove container when it stops",
                                "default": True,
                            },
                            "mem_limit": {
                                "type": "string",
                                "description": "Memory limit (e.g., '1g', '512m')",
                            },
                            "cpu_quota": {
                                "type": "integer",
                                "description": "CPU quota in microseconds",
                            },
                        },
                        "required": ["image"],
                    },
                ),
                Tool(
                    name="stop_container",
                    description="Stop a running container",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "container_id": {
                                "type": "string",
                                "description": "Container ID or name",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Seconds to wait before force killing",
                                "default": 10,
                            },
                        },
                        "required": ["container_id"],
                    },
                ),
                Tool(
                    name="remove_container",
                    description="Remove a container",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "container_id": {
                                "type": "string",
                                "description": "Container ID or name",
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Force remove even if running",
                                "default": False,
                            },
                        },
                        "required": ["container_id"],
                    },
                ),
                Tool(
                    name="get_container_logs",
                    description="Get logs from a container",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "container_id": {
                                "type": "string",
                                "description": "Container ID or name",
                            },
                            "tail": {
                                "type": "integer",
                                "description": "Number of lines to return from end",
                                "default": 100,
                            },
                            "follow": {
                                "type": "boolean",
                                "description": "Follow log output",
                                "default": False,
                            },
                        },
                        "required": ["container_id"],
                    },
                ),
                Tool(
                    name="list_containers",
                    description="List containers with optional filtering",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "all": {
                                "type": "boolean",
                                "description": "Show all containers (default: running only)",
                                "default": False,
                            },
                            "filters": {
                                "type": "object",
                                "description": "Filters to apply",
                                "default": {},
                            },
                        },
                    },
                ),
                Tool(
                    name="list_images",
                    description="List Docker images",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Filter by image name",
                            },
                            "filters": {
                                "type": "object",
                                "description": "Additional filters",
                                "default": {},
                            },
                        },
                    },
                ),
                Tool(
                    name="remove_image",
                    description="Remove Docker image",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image": {
                                "type": "string",
                                "description": "Image name or ID",
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Force removal",
                                "default": False,
                            },
                        },
                        "required": ["image"],
                    },
                ),
                Tool(
                    name="cleanup_resources",
                    description="Clean up managed containers and unused resources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "containers": {
                                "type": "boolean",
                                "description": "Clean up managed containers",
                                "default": True,
                            },
                            "images": {
                                "type": "boolean",
                                "description": "Remove dangling images",
                                "default": False,
                            },
                            "volumes": {
                                "type": "boolean",
                                "description": "Remove unused volumes",
                                "default": False,
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "build_image":
                    return await self._build_image(
                        arguments["path"],
                        arguments["tag"],
                        arguments.get("dockerfile", "Dockerfile"),
                        arguments.get("buildargs", {}),
                    )
                elif name == "run_container":
                    return await self._run_container(
                        arguments["image"],
                        arguments.get("command"),
                        arguments.get("environment", {}),
                        arguments.get("volumes", {}),
                        arguments.get("working_dir"),
                        arguments.get("detach", False),
                        arguments.get("remove", True),
                        arguments.get("mem_limit"),
                        arguments.get("cpu_quota"),
                    )
                elif name == "stop_container":
                    return await self._stop_container(
                        arguments["container_id"], arguments.get("timeout", 10)
                    )
                elif name == "remove_container":
                    return await self._remove_container(
                        arguments["container_id"], arguments.get("force", False)
                    )
                elif name == "get_container_logs":
                    return await self._get_container_logs(
                        arguments["container_id"],
                        arguments.get("tail", 100),
                        arguments.get("follow", False),
                    )
                elif name == "list_containers":
                    return await self._list_containers(
                        arguments.get("all", False), arguments.get("filters", {})
                    )
                elif name == "list_images":
                    return await self._list_images(
                        arguments.get("name"), arguments.get("filters", {})
                    )
                elif name == "remove_image":
                    return await self._remove_image(
                        arguments["image"], arguments.get("force", False)
                    )
                elif name == "cleanup_resources":
                    return await self._cleanup_resources(
                        arguments.get("containers", True),
                        arguments.get("images", False),
                        arguments.get("volumes", False),
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                self.logger.error(f"Error in {name}: {str(e)}")
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

    async def _build_image(
        self,
        path: str,
        tag: str,
        dockerfile: str = "Dockerfile",
        buildargs: dict[str, str] | None = None,
    ) -> list[TextContent]:
        """Build Docker image from Dockerfile."""
        if buildargs is None:
            buildargs = {}

        build_path = self._validate_path(path)
        dockerfile_path = build_path / dockerfile

        if not build_path.exists():
            return [
                TextContent(type="text", text=f"Build context not found: {build_path}")
            ]

        if not dockerfile_path.exists():
            return [
                TextContent(
                    type="text", text=f"Dockerfile not found: {dockerfile_path}"
                )
            ]

        try:
            self.logger.info(f"Building image {tag} from {build_path}")

            # Build image with streaming logs
            image, logs = self.docker_client.images.build(
                path=str(build_path),
                tag=tag,
                dockerfile=dockerfile,
                buildargs=buildargs,
                rm=True,  # Remove intermediate containers
                forcerm=True,  # Always remove intermediate containers
            )

            # Collect build logs
            build_output = []
            for log_entry in logs:
                if isinstance(log_entry, dict) and "stream" in log_entry:
                    stream_value = log_entry["stream"]
                    if isinstance(stream_value, str):
                        build_output.append(stream_value.strip())

            result = {
                "status": "success",
                "image_id": image.id,
                "tag": tag,
                "build_logs": build_output,
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except BuildError as e:
            build_logs = []
            for log in e.build_log:
                if "stream" in log:
                    build_logs.append(log["stream"].strip())

            result = {
                "status": "error",
                "error": "Build failed",
                "build_logs": build_logs,
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Build error: {str(e)}")]

    async def _run_container(
        self,
        image: str,
        command: str | None = None,
        environment: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        working_dir: str | None = None,
        detach: bool = False,
        remove: bool = True,
        mem_limit: str | None = None,
        cpu_quota: int | None = None,
    ) -> list[TextContent]:
        """Run a container with specified configuration."""
        if environment is None:
            environment = {}
        if volumes is None:
            volumes = {}

        try:
            # Validate and prepare volume mounts
            validated_volumes = {}
            for host_path, container_path in volumes.items():
                validated_host_path = self._validate_path(host_path)
                if not validated_host_path.exists():
                    return [
                        TextContent(
                            type="text",
                            text=f"Host path not found: {validated_host_path}",
                        )
                    ]
                validated_volumes[str(validated_host_path)] = {
                    "bind": container_path,
                    "mode": "rw",
                }

            # Prepare container configuration
            container_config: dict[str, Any] = {
                "image": image,
                "detach": detach,
                "remove": remove,
                "environment": environment,
                "volumes": validated_volumes,
            }

            if command:
                container_config["command"] = command
            if working_dir:
                container_config["working_dir"] = working_dir
            if mem_limit:
                container_config["mem_limit"] = mem_limit
            if cpu_quota:
                container_config["cpu_quota"] = cpu_quota

            # Run container
            container = self.docker_client.containers.run(**container_config)

            # Track managed container if detached
            if detach:
                self.managed_containers[container.id] = container

            result = {
                "status": "success",
                "container_id": container.id,
                "container_name": container.name,
                "detached": detach,
            }

            # If not detached, get output
            if not detach:
                try:
                    output = container.logs(stdout=True, stderr=True).decode("utf-8")
                    result["output"] = output
                    result["exit_code"] = container.wait()["StatusCode"]
                except Exception as e:
                    result["log_error"] = str(e)

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except APIError as e:
            return [TextContent(type="text", text=f"Docker API error: {str(e)}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Container run error: {str(e)}")]

    async def _stop_container(
        self, container_id: str, timeout: int = 10
    ) -> list[TextContent]:
        """Stop a running container."""
        try:
            container = self.docker_client.containers.get(container_id)
            container.stop(timeout=timeout)

            # Remove from managed containers if present
            if container_id in self.managed_containers:
                del self.managed_containers[container_id]

            result = {
                "status": "success",
                "container_id": container_id,
                "action": "stopped",
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Stop container error: {str(e)}")]

    async def _remove_container(
        self, container_id: str, force: bool = False
    ) -> list[TextContent]:
        """Remove a container."""
        try:
            container = self.docker_client.containers.get(container_id)
            container.remove(force=force)

            # Remove from managed containers if present
            if container_id in self.managed_containers:
                del self.managed_containers[container_id]

            result = {
                "status": "success",
                "container_id": container_id,
                "action": "removed",
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Remove container error: {str(e)}")]

    async def _get_container_logs(
        self, container_id: str, tail: int = 100, follow: bool = False
    ) -> list[TextContent]:
        """Get logs from a container."""
        try:
            container = self.docker_client.containers.get(container_id)

            if follow:
                log_stream = container.logs(
                    stdout=True,
                    stderr=True,
                    tail=tail,
                    stream=True,
                )
            else:
                logs_bytes = container.logs(
                    stdout=True,
                    stderr=True,
                    tail=tail,
                    stream=False,
                )

            if follow:
                # For streaming logs, return initial batch
                # (Full streaming would require different handling)
                log_lines = []
                for _ in range(min(tail, 50)):  # Limit streaming
                    try:
                        line = next(log_stream).decode("utf-8").strip()
                        log_lines.append(line)
                    except StopIteration:
                        break

                result = {
                    "container_id": container_id,
                    "logs": log_lines,
                    "streaming": True,
                }
            else:
                log_content = logs_bytes.decode("utf-8")
                result = {
                    "container_id": container_id,
                    "logs": log_content.strip().split("\n"),
                    "streaming": False,
                }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Get logs error: {str(e)}")]

    async def _list_containers(
        self, all_containers: bool = False, filters: dict[str, Any] | None = None
    ) -> list[TextContent]:
        """List containers with optional filtering."""
        if filters is None:
            filters = {}

        try:
            containers = self.docker_client.containers.list(
                all=all_containers, filters=filters
            )

            container_list = []
            for container in containers:
                container_info = {
                    "id": container.id,
                    "name": container.name,
                    "status": container.status,
                    "image": (
                        container.image.tags[0]
                        if container.image.tags
                        else container.image.id
                    ),
                    "created": str(container.attrs.get("Created", "")),
                    "ports": container.attrs.get("NetworkSettings", {}).get(
                        "Ports", {}
                    ),
                }
                container_list.append(container_info)

            result = {
                "containers": container_list,
                "count": len(container_list),
                "managed_containers": list(self.managed_containers.keys()),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"List containers error: {str(e)}")]

    async def _list_images(
        self, name: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[TextContent]:
        """List Docker images."""
        if filters is None:
            filters = {}

        try:
            images = self.docker_client.images.list(name=name, filters=filters)

            image_list = []
            for image in images:
                image_info = {
                    "id": image.id,
                    "tags": image.tags,
                    "size": image.attrs.get("Size", 0),
                    "created": str(image.attrs.get("Created", "")),
                }
                image_list.append(image_info)

            result = {
                "images": image_list,
                "count": len(image_list),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"List images error: {str(e)}")]

    async def _remove_image(self, image: str, force: bool = False) -> list[TextContent]:
        """Remove Docker image."""
        try:
            self.docker_client.images.remove(image=image, force=force)

            result = {
                "status": "success",
                "image": image,
                "action": "removed",
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Remove image error: {str(e)}")]

    async def _cleanup_resources(
        self, containers: bool = True, images: bool = False, volumes: bool = False
    ) -> list[TextContent]:
        """Clean up managed containers and unused resources."""
        cleanup_results = {}

        try:
            # Clean up managed containers
            if containers:
                cleaned_containers = []
                for container_id, container in self.managed_containers.copy().items():
                    try:
                        container.remove(force=True)
                        cleaned_containers.append(container_id)
                        del self.managed_containers[container_id]
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to cleanup container {container_id}: {e}"
                        )

                cleanup_results["containers"] = {
                    "cleaned": cleaned_containers,
                    "count": len(cleaned_containers),
                }

            # Clean up dangling images
            if images:
                try:
                    pruned = self.docker_client.images.prune(filters={"dangling": True})
                    cleanup_results["images"] = {
                        "reclaimed_space": pruned.get("SpaceReclaimed", 0),
                        "deleted_images": len(pruned.get("ImagesDeleted", [])),
                    }
                except Exception as e:
                    cleanup_results["images"] = {"error": str(e)}

            # Clean up unused volumes
            if volumes:
                try:
                    pruned = self.docker_client.volumes.prune()
                    cleanup_results["volumes"] = {
                        "reclaimed_space": pruned.get("SpaceReclaimed", 0),
                        "deleted_volumes": len(pruned.get("VolumesDeleted", [])),
                    }
                except Exception as e:
                    cleanup_results["volumes"] = {"error": str(e)}

            result = {
                "status": "success",
                "cleanup_results": cleanup_results,
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Cleanup error: {str(e)}")]

    async def run(self) -> None:
        """Run the MCP server with stdio transport."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="docker-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def main():
    """Main entry point for the Docker MCP Server."""
    import sys

    # Get project root from command line argument if provided
    project_root = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        import asyncio

        server = DockerMCPServer(project_root=project_root)
        asyncio.run(server.run())
    except RuntimeError as e:
        print(f"Failed to start Docker MCP Server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
