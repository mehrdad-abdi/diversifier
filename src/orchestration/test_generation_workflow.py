"""Test generation workflow for Docker-based acceptance testing."""

import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .mcp_manager import MCPManager, MCPServerType
from .acceptance_test_generator import (
    AcceptanceTestGenerator,
    AcceptanceTestGenerationResult,
)
from .doc_analyzer import DocumentationAnalysisResult
from .source_code_analyzer import SourceCodeAnalysisResult


@dataclass
class WorkflowConfiguration:
    """Configuration for test generation workflow."""

    project_root: str
    output_directory: str
    docker_registry: Optional[str] = None
    test_image_name: str = "diversifier-tests"
    test_image_tag: str = "latest"
    docker_network_name: str = "diversifier-test-network"
    test_timeout_minutes: int = 30
    parallel_test_execution: bool = True
    cleanup_containers: bool = True


@dataclass
class ContainerConfiguration:
    """Docker container configuration for testing."""

    app_container_name: str
    app_service_name: str
    app_port: int
    test_container_name: str
    test_service_name: str
    network_name: str
    volumes: Dict[str, str]
    environment_variables: Dict[str, str]
    healthcheck_config: Dict[str, Any]


@dataclass
class WorkflowExecutionResult:
    """Results of workflow execution."""

    workflow_id: str
    success: bool
    generation_result: Optional[AcceptanceTestGenerationResult]
    container_config: Optional[ContainerConfiguration]
    docker_compose_path: Optional[str]
    test_image_id: Optional[str]
    execution_logs: List[str]
    error_messages: List[str]
    execution_time_seconds: float
    timestamp: str


class TestGenerationWorkflow:
    """Orchestrates Docker-based acceptance test generation and execution."""

    def __init__(
        self,
        config: WorkflowConfiguration,
        mcp_manager: Optional[MCPManager] = None,
    ):
        """Initialize the test generation workflow.

        Args:
            config: Workflow configuration
            mcp_manager: MCP manager for server connections
        """
        self.config = config
        self.project_root = Path(config.project_root)
        self.output_dir = Path(config.output_directory)
        self.mcp_manager = mcp_manager or MCPManager(str(self.project_root))

        # Initialize components
        self.test_generator = AcceptanceTestGenerator(
            str(self.project_root), self.mcp_manager
        )

        self.logger = logging.getLogger("diversifier.workflow")

        # Workflow state
        self.workflow_id = f"workflow_{datetime.now(timezone.utc).isoformat()}"
        self.execution_logs: List[str] = []
        self.error_messages: List[str] = []

    async def initialize_mcp_servers(self) -> Dict[MCPServerType, bool]:
        """Initialize required MCP servers for the workflow.

        Returns:
            Dictionary mapping server types to initialization success
        """
        self._log("Initializing MCP servers for workflow")

        # Initialize all servers
        results = await self.mcp_manager.initialize_all_servers()

        # Check which servers are required and available
        required_servers = [
            MCPServerType.FILESYSTEM,
            MCPServerType.DOCKER,
        ]

        optional_servers = [
            MCPServerType.TESTING,
            MCPServerType.GIT,
        ]

        missing_required = []
        for server_type in required_servers:
            if not results.get(server_type, False):
                missing_required.append(server_type)

        if missing_required:
            error_msg = f"Required MCP servers failed to initialize: {[s.value for s in missing_required]}"
            self._log_error(error_msg)
            raise RuntimeError(error_msg)

        # Log status of optional servers
        for server_type in optional_servers:
            if results.get(server_type, False):
                self._log(
                    f"Optional server {server_type.value} initialized successfully"
                )
            else:
                self._log(f"Optional server {server_type.value} not available")

        self._log(
            f"MCP server initialization completed: {sum(results.values())}/{len(results)} servers active"
        )
        return results

    async def generate_acceptance_tests(
        self,
        doc_analysis: DocumentationAnalysisResult,
        source_analysis: SourceCodeAnalysisResult,
    ) -> AcceptanceTestGenerationResult:
        """Generate acceptance tests using the configured generator.

        Args:
            doc_analysis: Documentation analysis results
            source_analysis: Source code analysis results

        Returns:
            Test generation results
        """
        self._log("Generating acceptance tests")

        try:
            result = await self.test_generator.generate_acceptance_tests(
                doc_analysis, source_analysis
            )

            self._log(
                f"Generated {len(result.test_scenarios)} test scenarios in {len(result.test_suites)} suites"
            )
            self._log(f"Test generation confidence: {result.generation_confidence:.2f}")

            return result

        except Exception as e:
            error_msg = f"Test generation failed: {e}"
            self._log_error(error_msg)
            raise

    async def create_container_configuration(
        self,
        generation_result: AcceptanceTestGenerationResult,
        source_analysis: SourceCodeAnalysisResult,
    ) -> ContainerConfiguration:
        """Create Docker container configuration for testing.

        Args:
            generation_result: Test generation results
            source_analysis: Source code analysis results

        Returns:
            Container configuration
        """
        self._log("Creating container configuration")

        # Determine application port from analysis
        app_port = 8000  # Default
        if source_analysis.network_interfaces.get("ports"):
            ports = source_analysis.network_interfaces["ports"]
            if isinstance(ports, list) and ports:
                app_port = ports[0]

        # Create container configuration
        container_config = ContainerConfiguration(
            app_container_name=f"{self.config.test_image_name}_app",
            app_service_name="app",
            app_port=app_port,
            test_container_name=f"{self.config.test_image_name}_test",
            test_service_name="test",
            network_name=self.config.docker_network_name,
            volumes={
                str(self.project_root): "/app",
                str(self.output_dir): "/tests",
            },
            environment_variables={
                "BASE_URL": f"http://app:{app_port}",
                "TEST_TIMEOUT": str(self.config.test_timeout_minutes * 60),
                "PYTHONPATH": "/app:/tests",
            },
            healthcheck_config={
                "test": f"curl -f http://localhost:{app_port}/health || exit 1",
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "40s",
            },
        )

        # Add framework-specific environment variables
        framework = source_analysis.framework_detected
        if framework:
            container_config.environment_variables["DETECTED_FRAMEWORK"] = framework

        # Add database configuration if detected
        if any(
            "database" in integration.service_type.lower()
            for integration in source_analysis.external_service_integrations
        ):
            container_config.environment_variables["DATABASE_URL"] = "sqlite:///test.db"

        self._log(
            f"Container configuration created for {framework} application on port {app_port}"
        )
        return container_config

    async def generate_docker_compose(
        self,
        generation_result: AcceptanceTestGenerationResult,
        container_config: ContainerConfiguration,
    ) -> str:
        """Generate Docker Compose file for test orchestration.

        Args:
            generation_result: Test generation results
            container_config: Container configuration

        Returns:
            Path to generated docker-compose.yml file
        """
        self._log("Generating Docker Compose configuration")

        # Create Docker Compose content
        compose_content = {
            "version": "3.8",
            "networks": {container_config.network_name: {"driver": "bridge"}},
            "services": {
                container_config.app_service_name: {
                    "build": {
                        "context": str(self.project_root),
                        "dockerfile": "Dockerfile",
                    },
                    "container_name": container_config.app_container_name,
                    "networks": [container_config.network_name],
                    "ports": [
                        f"{container_config.app_port}:{container_config.app_port}"
                    ],
                    "environment": container_config.environment_variables,
                    "healthcheck": container_config.healthcheck_config,
                    "restart": "unless-stopped",
                },
                container_config.test_service_name: {
                    "build": {
                        "context": str(self.output_dir),
                        "dockerfile": "Dockerfile.test",
                    },
                    "container_name": container_config.test_container_name,
                    "networks": [container_config.network_name],
                    "environment": {
                        **container_config.environment_variables,
                        "PYTEST_ARGS": "--tb=short -v",
                    },
                    "volumes": [f"{self.output_dir}:/tests:ro"],
                    "depends_on": {
                        container_config.app_service_name: {
                            "condition": "service_healthy"
                        }
                    },
                    "command": ["python", "-m", "pytest", "/tests", "--tb=short", "-v"],
                },
            },
        }

        # Add database service if needed
        if "DATABASE_URL" in container_config.environment_variables:
            services = compose_content["services"]
            if isinstance(services, dict):
                services["db"] = {
                    "image": "postgres:13",
                    "container_name": f"{self.config.test_image_name}_db",
                    "networks": [container_config.network_name],
                    "environment": {
                        "POSTGRES_DB": "testdb",
                        "POSTGRES_USER": "testuser",
                        "POSTGRES_PASSWORD": "testpass",
                    },
                    "healthcheck": {
                        "test": ["CMD-SHELL", "pg_isready -U testuser"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5,
                    },
                }

                # Update app service to depend on database
                app_service = services.get(container_config.app_service_name)
                if isinstance(app_service, dict):
                    app_service["depends_on"] = {"db": {"condition": "service_healthy"}}

            # Update database URL
            container_config.environment_variables["DATABASE_URL"] = (
                "postgresql://testuser:testpass@db:5432/testdb"
            )

        # Write Docker Compose file
        compose_path = self.output_dir / "docker-compose.yml"
        compose_yaml = self._dict_to_yaml(compose_content)

        if self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
            await self.mcp_manager.call_tool(
                MCPServerType.FILESYSTEM,
                "write_file",
                {
                    "file_path": str(compose_path),
                    "content": compose_yaml,
                },
            )
        else:
            # Fallback to direct file write
            compose_path.write_text(compose_yaml)

        self._log(f"Docker Compose file generated: {compose_path}")
        return str(compose_path)

    async def build_test_container(
        self,
        generation_result: AcceptanceTestGenerationResult,
    ) -> Optional[str]:
        """Build Docker container with generated tests.

        Args:
            generation_result: Test generation results

        Returns:
            Docker image ID if successful, None otherwise
        """
        self._log("Building test container")

        if not self.mcp_manager.is_server_available(MCPServerType.DOCKER):
            self._log_error("Docker MCP server not available")
            return None

        try:
            # Create test Dockerfile
            dockerfile_content = self._create_test_dockerfile()
            dockerfile_path = self.output_dir / "Dockerfile.test"

            await self.mcp_manager.call_tool(
                MCPServerType.FILESYSTEM,
                "write_file",
                {
                    "file_path": str(dockerfile_path),
                    "content": dockerfile_content,
                },
            )

            # Build test container
            build_result = await self.mcp_manager.call_tool(
                MCPServerType.DOCKER,
                "build_image",
                {
                    "build_context": str(self.output_dir),
                    "dockerfile": "Dockerfile.test",
                    "image_name": self.config.test_image_name,
                    "image_tag": self.config.test_image_tag,
                },
            )

            if build_result and build_result.get("success"):
                image_id = build_result.get("image_id")
                self._log(f"Test container built successfully: {image_id}")
                return image_id
            else:
                self._log_error(f"Test container build failed: {build_result}")
                return None

        except Exception as e:
            self._log_error(f"Test container build error: {e}")
            return None

    async def execute_test_workflow(
        self,
        docker_compose_path: str,
        container_config: ContainerConfiguration,
    ) -> Dict[str, Any]:
        """Execute the complete test workflow using Docker Compose.

        Args:
            docker_compose_path: Path to Docker Compose file
            container_config: Container configuration

        Returns:
            Execution results
        """
        self._log("Executing test workflow")

        if not self.mcp_manager.is_server_available(MCPServerType.DOCKER):
            raise RuntimeError("Docker MCP server not available")

        try:
            # Start services
            start_result = await self.mcp_manager.call_tool(
                MCPServerType.DOCKER,
                "compose_up",
                {
                    "compose_file": docker_compose_path,
                    "services": [container_config.app_service_name],
                    "detached": True,
                },
            )

            if not start_result or not start_result.get("success"):
                raise RuntimeError(
                    f"Failed to start application services: {start_result}"
                )

            self._log("Application services started successfully")

            # Wait for application to be healthy
            await self._wait_for_service_health(
                container_config.app_container_name, timeout_seconds=120
            )

            # Run tests
            test_result = await self.mcp_manager.call_tool(
                MCPServerType.DOCKER,
                "compose_run",
                {
                    "compose_file": docker_compose_path,
                    "service": container_config.test_service_name,
                    "command": [
                        "python",
                        "-m",
                        "pytest",
                        "/tests",
                        "--tb=short",
                        "-v",
                        "--json-report",
                        "--json-report-file=/tests/test-report.json",
                    ],
                    "remove": True,
                },
            )

            # Collect test results
            results = {
                "test_execution_success": (
                    test_result.get("success", False) if test_result else False
                ),
                "test_output": test_result.get("output", "") if test_result else "",
                "test_exit_code": (
                    test_result.get("exit_code", -1) if test_result else -1
                ),
                "start_result": start_result,
                "container_logs": await self._collect_container_logs(container_config),
            }

            self._log(
                f"Test execution completed with exit code: {results['test_exit_code']}"
            )
            return results

        except Exception as e:
            self._log_error(f"Test workflow execution failed: {e}")
            raise
        finally:
            # Cleanup containers if configured
            if self.config.cleanup_containers:
                await self._cleanup_containers(docker_compose_path)

    async def run_complete_workflow(
        self,
        doc_analysis: DocumentationAnalysisResult,
        source_analysis: SourceCodeAnalysisResult,
    ) -> WorkflowExecutionResult:
        """Run the complete test generation and execution workflow.

        Args:
            doc_analysis: Documentation analysis results
            source_analysis: Source code analysis results

        Returns:
            Complete workflow execution results
        """
        start_time = datetime.now(timezone.utc)
        self._log(f"Starting complete workflow: {self.workflow_id}")

        try:
            # Initialize MCP servers
            await self.initialize_mcp_servers()

            # Generate acceptance tests
            generation_result = await self.generate_acceptance_tests(
                doc_analysis, source_analysis
            )

            # Export test suites
            await self.test_generator.export_test_suites(
                generation_result, str(self.output_dir)
            )

            # Create container configuration
            container_config = await self.create_container_configuration(
                generation_result, source_analysis
            )

            # Generate Docker Compose
            compose_path = await self.generate_docker_compose(
                generation_result, container_config
            )

            # Build test container
            test_image_id = await self.build_test_container(generation_result)

            # Execute workflow (optional - can be run separately)
            if hasattr(self.config, "auto_execute") and self.config.auto_execute:
                await self.execute_test_workflow(compose_path, container_config)

            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            result = WorkflowExecutionResult(
                workflow_id=self.workflow_id,
                success=True,
                generation_result=generation_result,
                container_config=container_config,
                docker_compose_path=compose_path,
                test_image_id=test_image_id,
                execution_logs=self.execution_logs.copy(),
                error_messages=self.error_messages.copy(),
                execution_time_seconds=execution_time,
                timestamp=end_time.isoformat(),
            )

            self._log(
                f"Workflow completed successfully in {execution_time:.2f} seconds"
            )
            return result

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            self._log_error(f"Workflow failed: {e}")

            return WorkflowExecutionResult(
                workflow_id=self.workflow_id,
                success=False,
                generation_result=None,
                container_config=None,
                docker_compose_path=None,
                test_image_id=None,
                execution_logs=self.execution_logs.copy(),
                error_messages=self.error_messages.copy(),
                execution_time_seconds=execution_time,
                timestamp=end_time.isoformat(),
            )

    # Helper methods

    def _log(self, message: str) -> None:
        """Log a message."""
        self.logger.info(message)
        self.execution_logs.append(
            f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        )

    def _log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
        self.error_messages.append(
            f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        )

    def _dict_to_yaml(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to YAML string."""
        try:
            import yaml  # type: ignore

            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fallback to JSON format if yaml is not available
            import json

            return json.dumps(data, indent=2)

    def _create_test_dockerfile(self) -> str:
        """Create Dockerfile content for test container."""
        return """FROM python:3.11-slim

WORKDIR /tests

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /tests/
RUN pip install --no-cache-dir -r requirements.txt

# Copy test files
COPY . /tests/

# Set environment variables
ENV PYTHONPATH="/tests"
ENV PYTHONUNBUFFERED=1

# Run tests by default
CMD ["python", "-m", "pytest", "/tests", "--tb=short", "-v"]
"""

    async def _wait_for_service_health(
        self,
        container_name: str,
        timeout_seconds: int = 120,
    ) -> None:
        """Wait for a service to become healthy."""
        self._log(f"Waiting for {container_name} to become healthy")

        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            try:
                health_result = await self.mcp_manager.call_tool(
                    MCPServerType.DOCKER,
                    "inspect_container",
                    {"container_name": container_name},
                )

                if health_result and health_result.get("health_status") == "healthy":
                    self._log(f"{container_name} is healthy")
                    return

            except Exception as e:
                self._log(f"Health check error: {e}")

            await asyncio.sleep(5)

        raise TimeoutError(
            f"{container_name} did not become healthy within {timeout_seconds} seconds"
        )

    async def _collect_container_logs(
        self,
        container_config: ContainerConfiguration,
    ) -> Dict[str, str]:
        """Collect logs from containers."""
        logs = {}

        containers = [
            container_config.app_container_name,
            container_config.test_container_name,
        ]

        for container_name in containers:
            try:
                log_result = await self.mcp_manager.call_tool(
                    MCPServerType.DOCKER,
                    "get_container_logs",
                    {"container_name": container_name},
                )

                if log_result:
                    logs[container_name] = log_result.get("logs", "")

            except Exception as e:
                logs[container_name] = f"Failed to collect logs: {e}"

        return logs

    async def _cleanup_containers(self, docker_compose_path: str) -> None:
        """Cleanup containers and resources."""
        self._log("Cleaning up containers")

        try:
            cleanup_result = await self.mcp_manager.call_tool(
                MCPServerType.DOCKER,
                "compose_down",
                {
                    "compose_file": docker_compose_path,
                    "remove_volumes": True,
                    "remove_images": "local",
                },
            )

            if cleanup_result and cleanup_result.get("success"):
                self._log("Container cleanup completed successfully")
            else:
                self._log_error(f"Container cleanup failed: {cleanup_result}")

        except Exception as e:
            self._log_error(f"Container cleanup error: {e}")


class WorkflowManager:
    """Manager for coordinating multiple test generation workflows."""

    def __init__(self, base_config: WorkflowConfiguration):
        """Initialize workflow manager.

        Args:
            base_config: Base configuration for workflows
        """
        self.base_config = base_config
        self.active_workflows: Dict[str, TestGenerationWorkflow] = {}
        self.completed_workflows: Dict[str, WorkflowExecutionResult] = {}
        self.logger = logging.getLogger("diversifier.workflow_manager")

    async def create_workflow(
        self,
        project_root: str,
        output_directory: str,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new test generation workflow.

        Args:
            project_root: Root directory of the project to test
            output_directory: Directory for test output
            custom_config: Custom configuration overrides

        Returns:
            Workflow ID
        """
        # Create workflow configuration
        config = WorkflowConfiguration(
            project_root=project_root,
            output_directory=output_directory,
            docker_registry=self.base_config.docker_registry,
            test_image_name=self.base_config.test_image_name,
            test_image_tag=self.base_config.test_image_tag,
            docker_network_name=self.base_config.docker_network_name,
            test_timeout_minutes=self.base_config.test_timeout_minutes,
            parallel_test_execution=self.base_config.parallel_test_execution,
            cleanup_containers=self.base_config.cleanup_containers,
        )

        # Apply custom configuration
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create workflow
        workflow = TestGenerationWorkflow(config)
        workflow_id = workflow.workflow_id

        self.active_workflows[workflow_id] = workflow
        self.logger.info(f"Created workflow {workflow_id}")

        return workflow_id

    async def run_workflow(
        self,
        workflow_id: str,
        doc_analysis: DocumentationAnalysisResult,
        source_analysis: SourceCodeAnalysisResult,
    ) -> WorkflowExecutionResult:
        """Run a specific workflow.

        Args:
            workflow_id: ID of the workflow to run
            doc_analysis: Documentation analysis results
            source_analysis: Source code analysis results

        Returns:
            Workflow execution results
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.active_workflows[workflow_id]
        result = await workflow.run_complete_workflow(doc_analysis, source_analysis)

        # Move to completed workflows
        self.completed_workflows[workflow_id] = result
        del self.active_workflows[workflow_id]

        return result

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Workflow status information
        """
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                "status": "active",
                "workflow_id": workflow_id,
                "logs": workflow.execution_logs[-10:],  # Last 10 logs
                "errors": workflow.error_messages,
            }
        elif workflow_id in self.completed_workflows:
            result = self.completed_workflows[workflow_id]
            return {
                "status": "completed",
                "success": result.success,
                "workflow_id": workflow_id,
                "execution_time": result.execution_time_seconds,
                "timestamp": result.timestamp,
            }
        else:
            return {"status": "not_found", "workflow_id": workflow_id}

    def list_workflows(self) -> Dict[str, List[str]]:
        """List all workflows by status.

        Returns:
            Dictionary mapping status to workflow IDs
        """
        return {
            "active": list(self.active_workflows.keys()),
            "completed": list(self.completed_workflows.keys()),
        }
