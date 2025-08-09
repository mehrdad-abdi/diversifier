"""Tests for test generation workflow functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from src.orchestration.test_generation_workflow import (
    TestGenerationWorkflow,
    WorkflowConfiguration,
    ContainerConfiguration,
    WorkflowExecutionResult,
    WorkflowManager,
)
from src.orchestration.mcp_manager import MCPManager, MCPServerType
from src.orchestration.acceptance_test_generator import (
    AcceptanceTestGenerationResult,
    AcceptanceTestSuite,
    AcceptanceTestScenario,
)
from src.orchestration.doc_analyzer import (
    DocumentationAnalysisResult,
    ExternalInterface,
    DockerServiceInfo,
)
from src.orchestration.source_code_analyzer import (
    SourceCodeAnalysisResult,
    APIEndpoint,
    ExternalServiceIntegration,
    ConfigurationUsage,
)


class TestWorkflowConfiguration:
    """Test WorkflowConfiguration dataclass."""

    def test_workflow_configuration_creation(self):
        """Test creating a workflow configuration."""
        config = WorkflowConfiguration(
            project_root="/test/project",
            output_directory="/test/output",
            docker_registry="test-registry",
            test_image_name="test-image",
            test_image_tag="v1.0",
            docker_network_name="test-network",
            test_timeout_minutes=45,
            parallel_test_execution=True,
            cleanup_containers=False,
        )

        assert config.project_root == "/test/project"
        assert config.output_directory == "/test/output"
        assert config.docker_registry == "test-registry"
        assert config.test_image_name == "test-image"
        assert config.test_image_tag == "v1.0"
        assert config.docker_network_name == "test-network"
        assert config.test_timeout_minutes == 45
        assert config.parallel_test_execution is True
        assert config.cleanup_containers is False

    def test_workflow_configuration_defaults(self):
        """Test workflow configuration with default values."""
        config = WorkflowConfiguration(
            project_root="/test/project",
            output_directory="/test/output",
        )

        assert config.project_root == "/test/project"
        assert config.output_directory == "/test/output"
        assert config.docker_registry is None
        assert config.test_image_name == "diversifier-tests"
        assert config.test_image_tag == "latest"
        assert config.docker_network_name == "diversifier-test-network"
        assert config.test_timeout_minutes == 30
        assert config.parallel_test_execution is True
        assert config.cleanup_containers is True


class TestContainerConfiguration:
    """Test ContainerConfiguration dataclass."""

    def test_container_configuration_creation(self):
        """Test creating a container configuration."""
        config = ContainerConfiguration(
            app_container_name="test-app",
            app_service_name="app",
            app_port=5000,
            test_container_name="test-container",
            test_service_name="test",
            network_name="test-network",
            volumes={"/host": "/container"},
            environment_variables={"ENV_VAR": "value"},
            healthcheck_config={"test": "curl http://localhost:5000/health"},
        )

        assert config.app_container_name == "test-app"
        assert config.app_service_name == "app"
        assert config.app_port == 5000
        assert config.test_container_name == "test-container"
        assert config.test_service_name == "test"
        assert config.network_name == "test-network"
        assert config.volumes == {"/host": "/container"}
        assert config.environment_variables == {"ENV_VAR": "value"}
        assert config.healthcheck_config == {
            "test": "curl http://localhost:5000/health"
        }


class TestWorkflowExecutionResult:
    """Test WorkflowExecutionResult dataclass."""

    def test_workflow_execution_result_creation(self):
        """Test creating a workflow execution result."""
        result = WorkflowExecutionResult(
            workflow_id="test-workflow-123",
            success=True,
            generation_result=None,
            container_config=None,
            docker_compose_path="/path/to/compose.yml",
            test_image_id="sha256:abc123",
            execution_logs=["Log 1", "Log 2"],
            error_messages=["Error 1"],
            execution_time_seconds=120.5,
            timestamp="2024-01-01T12:00:00Z",
        )

        assert result.workflow_id == "test-workflow-123"
        assert result.success is True
        assert result.generation_result is None
        assert result.container_config is None
        assert result.docker_compose_path == "/path/to/compose.yml"
        assert result.test_image_id == "sha256:abc123"
        assert result.execution_logs == ["Log 1", "Log 2"]
        assert result.error_messages == ["Error 1"]
        assert result.execution_time_seconds == 120.5
        assert result.timestamp == "2024-01-01T12:00:00Z"


class TestTestGenerationWorkflow:
    """Test TestGenerationWorkflow class."""

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = Mock(spec=MCPManager)
        manager.is_server_available.return_value = True
        manager.call_tool = AsyncMock()
        manager.initialize_all_servers = AsyncMock(
            return_value={
                MCPServerType.FILESYSTEM: True,
                MCPServerType.DOCKER: True,
                MCPServerType.TESTING: True,
                MCPServerType.GIT: True,
            }
        )
        return manager

    @pytest.fixture
    def workflow_config(self):
        """Create a test workflow configuration."""
        return WorkflowConfiguration(
            project_root="/test/project",
            output_directory="/test/output",
            test_image_name="test-workflow",
            docker_network_name="test-network",
        )

    @pytest.fixture
    def workflow(self, workflow_config, mock_mcp_manager):
        """Create a test workflow instance."""
        with patch(
            "src.orchestration.test_generation_workflow.AcceptanceTestGenerator"
        ):
            return TestGenerationWorkflow(workflow_config, mock_mcp_manager)

    @pytest.fixture
    def sample_source_analysis(self):
        """Create sample source code analysis."""
        endpoints = [
            APIEndpoint(
                path="/api/users",
                methods=["GET", "POST"],
                handler="UserView",
                authentication_required=True,
                file_location="views.py:10",
            )
        ]

        integrations = [
            ExternalServiceIntegration(
                service_type="database",
                purpose="user storage",
                connection_pattern="SQLAlchemy",
                configuration_source="DATABASE_URL",
                file_location="models.py:5",
            )
        ]

        config_usage = [
            ConfigurationUsage(
                name="DATABASE_URL",
                purpose="database connection",
                required=True,
                default_value=None,
                usage_locations=["models.py:15"],
                config_type="environment_variable",
            )
        ]

        return SourceCodeAnalysisResult(
            api_endpoints=endpoints,
            external_service_integrations=integrations,
            configuration_usage=config_usage,
            existing_test_patterns=[],
            network_interfaces={"ports": [8000]},
            security_patterns={"auth": "JWT"},
            testing_requirements={"database": "required"},
            framework_detected="flask",
            analysis_confidence=0.8,
        )

    @pytest.fixture
    def sample_doc_analysis(self):
        """Create sample documentation analysis."""
        interfaces = [
            ExternalInterface(
                type="http_api",
                name="/api/users",
                description="User management API",
                port=8000,
                protocol="HTTP",
            )
        ]

        docker_services = [
            DockerServiceInfo(
                name="web",
                container_name="app_container",
                exposed_ports=[8000],
                dependencies=["db"],
                environment_variables=["DATABASE_URL"],
            )
        ]

        return DocumentationAnalysisResult(
            external_interfaces=interfaces,
            docker_services=docker_services,
            network_configuration={"ports": {"web": 8000}},
            testing_requirements={"docker": "required"},
            deployment_patterns={"compose": True},
            analysis_confidence=0.7,
        )

    @pytest.fixture
    def sample_generation_result(self):
        """Create sample test generation result."""
        suite = AcceptanceTestSuite(
            name="http_api_tests",
            description="HTTP API tests",
            test_file_content="def test_api(): pass",
        )

        scenario = AcceptanceTestScenario(
            category="http_api",
            name="Test API",
            description="Test user endpoint",
            test_method_name="test_api",
            test_code="pass",
            dependencies=["requests"],
            docker_services=["app"],
        )

        return AcceptanceTestGenerationResult(
            test_suites=[suite],
            test_scenarios=[scenario],
            docker_configuration={},
            test_dependencies=["pytest"],
            coverage_analysis={"total": 1},
            generation_confidence=0.8,
        )

    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.config is not None
        assert workflow.project_root == Path("/test/project")
        assert workflow.output_dir == Path("/test/output")
        assert workflow.mcp_manager is not None
        assert workflow.test_generator is not None
        assert workflow.workflow_id.startswith("workflow_")
        assert isinstance(workflow.execution_logs, list)
        assert isinstance(workflow.error_messages, list)

    @pytest.mark.asyncio
    async def test_initialize_mcp_servers_success(self, workflow, mock_mcp_manager):
        """Test successful MCP server initialization."""
        results = await workflow.initialize_mcp_servers()

        assert results[MCPServerType.FILESYSTEM] is True
        assert results[MCPServerType.DOCKER] is True
        mock_mcp_manager.initialize_all_servers.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mcp_servers_missing_required(
        self, workflow, mock_mcp_manager
    ):
        """Test MCP server initialization with missing required servers."""
        mock_mcp_manager.initialize_all_servers.return_value = {
            MCPServerType.FILESYSTEM: False,  # Required server failed
            MCPServerType.DOCKER: True,
            MCPServerType.TESTING: True,
            MCPServerType.GIT: True,
        }

        with pytest.raises(
            RuntimeError, match="Required MCP servers failed to initialize"
        ):
            await workflow.initialize_mcp_servers()

    @pytest.mark.asyncio
    async def test_generate_acceptance_tests(
        self,
        workflow,
        sample_doc_analysis,
        sample_source_analysis,
        sample_generation_result,
    ):
        """Test acceptance test generation."""
        # Mock the test generator
        workflow.test_generator.generate_acceptance_tests = AsyncMock(
            return_value=sample_generation_result
        )

        result = await workflow.generate_acceptance_tests(
            sample_doc_analysis, sample_source_analysis
        )

        assert result == sample_generation_result
        workflow.test_generator.generate_acceptance_tests.assert_called_once_with(
            sample_doc_analysis, sample_source_analysis
        )

    @pytest.mark.asyncio
    async def test_generate_acceptance_tests_failure(
        self, workflow, sample_doc_analysis, sample_source_analysis
    ):
        """Test acceptance test generation failure."""
        workflow.test_generator.generate_acceptance_tests = AsyncMock(
            side_effect=Exception("Test generation failed")
        )

        with pytest.raises(Exception, match="Test generation failed"):
            await workflow.generate_acceptance_tests(
                sample_doc_analysis, sample_source_analysis
            )

    @pytest.mark.asyncio
    async def test_create_container_configuration(
        self, workflow, sample_generation_result, sample_source_analysis
    ):
        """Test container configuration creation."""
        config = await workflow.create_container_configuration(
            sample_generation_result, sample_source_analysis
        )

        assert isinstance(config, ContainerConfiguration)
        assert config.app_container_name == "test-workflow_app"
        assert config.app_service_name == "app"
        assert config.app_port == 8000  # From source analysis
        assert config.test_container_name == "test-workflow_test"
        assert config.test_service_name == "test"
        assert config.network_name == "test-network"
        assert "BASE_URL" in config.environment_variables
        assert config.environment_variables["BASE_URL"] == "http://app:8000"
        assert "DETECTED_FRAMEWORK" in config.environment_variables
        assert config.environment_variables["DETECTED_FRAMEWORK"] == "flask"

    @pytest.mark.asyncio
    async def test_create_container_configuration_with_database(
        self, workflow, sample_generation_result, sample_source_analysis
    ):
        """Test container configuration creation with database integration."""
        config = await workflow.create_container_configuration(
            sample_generation_result, sample_source_analysis
        )

        # Database should be detected from integrations
        assert "DATABASE_URL" in config.environment_variables
        assert config.environment_variables["DATABASE_URL"] == "sqlite:///test.db"

    @pytest.mark.asyncio
    async def test_generate_docker_compose(
        self, workflow, sample_generation_result, mock_mcp_manager
    ):
        """Test Docker Compose generation."""
        container_config = ContainerConfiguration(
            app_container_name="test-app",
            app_service_name="app",
            app_port=8000,
            test_container_name="test-container",
            test_service_name="test",
            network_name="test-network",
            volumes={"/test": "/app"},
            environment_variables={"BASE_URL": "http://app:8000"},
            healthcheck_config={"test": "curl -f http://localhost:8000/health"},
        )

        mock_mcp_manager.call_tool.return_value = {"success": True}

        with patch.object(
            workflow,
            "_dict_to_yaml",
            return_value="version: '3.8'\nservices:\n  app: {}",
        ):
            compose_path = await workflow.generate_docker_compose(
                sample_generation_result, container_config
            )

            assert compose_path == str(workflow.output_dir / "docker-compose.yml")
            mock_mcp_manager.call_tool.assert_called_with(
                MCPServerType.FILESYSTEM,
                "write_file",
                {
                    "file_path": str(workflow.output_dir / "docker-compose.yml"),
                    "content": "version: '3.8'\nservices:\n  app: {}",
                },
            )

    @pytest.mark.asyncio
    async def test_generate_docker_compose_with_database(
        self, workflow, sample_generation_result, mock_mcp_manager
    ):
        """Test Docker Compose generation with database service."""
        container_config = ContainerConfiguration(
            app_container_name="test-app",
            app_service_name="app",
            app_port=8000,
            test_container_name="test-container",
            test_service_name="test",
            network_name="test-network",
            volumes={"/test": "/app"},
            environment_variables={
                "BASE_URL": "http://app:8000",
                "DATABASE_URL": "sqlite:///test.db",
            },
            healthcheck_config={"test": "curl -f http://localhost:8000/health"},
        )

        mock_mcp_manager.call_tool.return_value = {"success": True}

        def mock_dict_to_yaml(data):
            # Verify database service is included
            assert "db" in data["services"]
            assert data["services"]["db"]["image"] == "postgres:13"
            return "version: '3.8'\nservices:\n  app: {}\n  db: {}\n  test: {}"

        with patch.object(workflow, "_dict_to_yaml", side_effect=mock_dict_to_yaml):
            compose_path = await workflow.generate_docker_compose(
                sample_generation_result, container_config
            )

            assert compose_path == str(workflow.output_dir / "docker-compose.yml")

    @pytest.mark.asyncio
    async def test_build_test_container(
        self, workflow, sample_generation_result, mock_mcp_manager
    ):
        """Test test container building."""
        mock_mcp_manager.call_tool.side_effect = [
            {"success": True},  # write_file call for Dockerfile
            {"success": True, "image_id": "sha256:abc123"},  # build_image call
        ]

        image_id = await workflow.build_test_container(sample_generation_result)

        assert image_id == "sha256:abc123"
        assert mock_mcp_manager.call_tool.call_count == 2

        # Check Dockerfile write call
        dockerfile_call = mock_mcp_manager.call_tool.call_args_list[0]
        assert dockerfile_call[0][0] == MCPServerType.FILESYSTEM
        assert dockerfile_call[0][1] == "write_file"
        assert "Dockerfile.test" in dockerfile_call[0][2]["file_path"]
        assert "FROM python:3.11-slim" in dockerfile_call[0][2]["content"]

        # Check build_image call
        build_call = mock_mcp_manager.call_tool.call_args_list[1]
        assert build_call[0][0] == MCPServerType.DOCKER
        assert build_call[0][1] == "build_image"

    @pytest.mark.asyncio
    async def test_build_test_container_docker_unavailable(
        self, workflow, sample_generation_result, mock_mcp_manager
    ):
        """Test test container building when Docker is unavailable."""
        mock_mcp_manager.is_server_available.return_value = False

        result = await workflow.build_test_container(sample_generation_result)

        assert result is None

    @pytest.mark.asyncio
    async def test_build_test_container_build_failure(
        self, workflow, sample_generation_result, mock_mcp_manager
    ):
        """Test test container building failure."""
        mock_mcp_manager.call_tool.side_effect = [
            {"success": True},  # write_file call
            {"success": False, "error": "Build failed"},  # build_image call
        ]

        result = await workflow.build_test_container(sample_generation_result)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_test_workflow(self, workflow, mock_mcp_manager):
        """Test test workflow execution."""
        container_config = ContainerConfiguration(
            app_container_name="test-app",
            app_service_name="app",
            app_port=8000,
            test_container_name="test-container",
            test_service_name="test",
            network_name="test-network",
            volumes={},
            environment_variables={},
            healthcheck_config={},
        )

        # Mock MCP calls
        mock_mcp_manager.call_tool.side_effect = [
            {"success": True},  # compose_up
            {"success": True, "output": "Test output", "exit_code": 0},  # compose_run
        ]

        # Mock health check
        with patch.object(workflow, "_wait_for_service_health", new_callable=AsyncMock):
            with patch.object(
                workflow,
                "_collect_container_logs",
                new_callable=AsyncMock,
                return_value={},
            ):
                with patch.object(
                    workflow, "_cleanup_containers", new_callable=AsyncMock
                ):
                    result = await workflow.execute_test_workflow(
                        "/test/docker-compose.yml", container_config
                    )

        assert result["test_execution_success"] is True
        assert result["test_output"] == "Test output"
        assert result["test_exit_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_test_workflow_docker_unavailable(
        self, workflow, mock_mcp_manager
    ):
        """Test test workflow execution when Docker is unavailable."""
        mock_mcp_manager.is_server_available.return_value = False

        container_config = ContainerConfiguration(
            app_container_name="test-app",
            app_service_name="app",
            app_port=8000,
            test_container_name="test-container",
            test_service_name="test",
            network_name="test-network",
            volumes={},
            environment_variables={},
            healthcheck_config={},
        )

        with pytest.raises(RuntimeError, match="Docker MCP server not available"):
            await workflow.execute_test_workflow(
                "/test/docker-compose.yml", container_config
            )

    @pytest.mark.asyncio
    async def test_run_complete_workflow_success(
        self,
        workflow,
        sample_doc_analysis,
        sample_source_analysis,
        sample_generation_result,
        mock_mcp_manager,
    ):
        """Test complete workflow execution success."""
        # Mock all async methods
        workflow.initialize_mcp_servers = AsyncMock(
            return_value={
                MCPServerType.FILESYSTEM: True,
                MCPServerType.DOCKER: True,
            }
        )
        workflow.generate_acceptance_tests = AsyncMock(
            return_value=sample_generation_result
        )
        workflow.test_generator.export_test_suites = AsyncMock()
        workflow.create_container_configuration = AsyncMock(
            return_value=ContainerConfiguration(
                app_container_name="test-app",
                app_service_name="app",
                app_port=8000,
                test_container_name="test-container",
                test_service_name="test",
                network_name="test-network",
                volumes={},
                environment_variables={},
                healthcheck_config={},
            )
        )
        workflow.generate_docker_compose = AsyncMock(return_value="/test/compose.yml")
        workflow.build_test_container = AsyncMock(return_value="sha256:abc123")

        result = await workflow.run_complete_workflow(
            sample_doc_analysis, sample_source_analysis
        )

        assert isinstance(result, WorkflowExecutionResult)
        assert result.success is True
        assert result.generation_result == sample_generation_result
        assert result.docker_compose_path == "/test/compose.yml"
        assert result.test_image_id == "sha256:abc123"
        assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_run_complete_workflow_failure(
        self,
        workflow,
        sample_doc_analysis,
        sample_source_analysis,
        mock_mcp_manager,
    ):
        """Test complete workflow execution failure."""
        # Mock initialization failure
        workflow.initialize_mcp_servers = AsyncMock(
            side_effect=RuntimeError("MCP initialization failed")
        )

        result = await workflow.run_complete_workflow(
            sample_doc_analysis, sample_source_analysis
        )

        assert isinstance(result, WorkflowExecutionResult)
        assert result.success is False
        assert result.generation_result is None
        assert len(result.error_messages) > 0
        assert result.execution_time_seconds > 0

    def test_log_methods(self, workflow):
        """Test logging methods."""
        workflow._log("Test message")
        workflow._log_error("Test error")

        assert len(workflow.execution_logs) == 1
        assert len(workflow.error_messages) == 1
        assert "Test message" in workflow.execution_logs[0]
        assert "Test error" in workflow.error_messages[0]

    def test_dict_to_yaml(self, workflow):
        """Test YAML conversion."""
        data = {"version": "3.8", "services": {"app": {"image": "test"}}}

        with patch("yaml.dump", return_value="mocked yaml"):
            result = workflow._dict_to_yaml(data)
            assert result == "mocked yaml"

    def test_create_test_dockerfile(self, workflow):
        """Test test Dockerfile creation."""
        dockerfile = workflow._create_test_dockerfile()

        assert "FROM python:3.11-slim" in dockerfile
        assert "WORKDIR /tests" in dockerfile
        assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
        assert (
            'CMD ["python", "-m", "pytest", "/tests", "--tb=short", "-v"]' in dockerfile
        )

    @pytest.mark.asyncio
    async def test_wait_for_service_health_success(self, workflow, mock_mcp_manager):
        """Test waiting for service health success."""
        mock_mcp_manager.call_tool.return_value = {"health_status": "healthy"}

        # Should not raise exception
        await workflow._wait_for_service_health("test-container", timeout_seconds=10)

        mock_mcp_manager.call_tool.assert_called_with(
            MCPServerType.DOCKER,
            "inspect_container",
            {"container_name": "test-container"},
        )

    @pytest.mark.asyncio
    async def test_wait_for_service_health_timeout(self, workflow, mock_mcp_manager):
        """Test waiting for service health timeout."""
        mock_mcp_manager.call_tool.return_value = {"health_status": "unhealthy"}

        with pytest.raises(TimeoutError, match="did not become healthy"):
            await workflow._wait_for_service_health("test-container", timeout_seconds=1)

    @pytest.mark.asyncio
    async def test_collect_container_logs(self, workflow, mock_mcp_manager):
        """Test collecting container logs."""
        container_config = ContainerConfiguration(
            app_container_name="test-app",
            app_service_name="app",
            app_port=8000,
            test_container_name="test-container",
            test_service_name="test",
            network_name="test-network",
            volumes={},
            environment_variables={},
            healthcheck_config={},
        )

        mock_mcp_manager.call_tool.side_effect = [
            {"logs": "App logs"},
            {"logs": "Test logs"},
        ]

        logs = await workflow._collect_container_logs(container_config)

        assert logs["test-app"] == "App logs"
        assert logs["test-container"] == "Test logs"
        assert mock_mcp_manager.call_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_collect_container_logs_failure(self, workflow, mock_mcp_manager):
        """Test collecting container logs with failure."""
        container_config = ContainerConfiguration(
            app_container_name="test-app",
            app_service_name="app",
            app_port=8000,
            test_container_name="test-container",
            test_service_name="test",
            network_name="test-network",
            volumes={},
            environment_variables={},
            healthcheck_config={},
        )

        mock_mcp_manager.call_tool.side_effect = Exception("Log collection failed")

        logs = await workflow._collect_container_logs(container_config)

        assert "Failed to collect logs" in logs["test-app"]
        assert "Failed to collect logs" in logs["test-container"]

    @pytest.mark.asyncio
    async def test_cleanup_containers_success(self, workflow, mock_mcp_manager):
        """Test container cleanup success."""
        mock_mcp_manager.call_tool.return_value = {"success": True}

        # Should not raise exception
        await workflow._cleanup_containers("/test/compose.yml")

        mock_mcp_manager.call_tool.assert_called_with(
            MCPServerType.DOCKER,
            "compose_down",
            {
                "compose_file": "/test/compose.yml",
                "remove_volumes": True,
                "remove_images": "local",
            },
        )

    @pytest.mark.asyncio
    async def test_cleanup_containers_failure(self, workflow, mock_mcp_manager):
        """Test container cleanup failure."""
        mock_mcp_manager.call_tool.return_value = {"success": False}

        # Should not raise exception, just log error
        await workflow._cleanup_containers("/test/compose.yml")

        assert len(workflow.error_messages) > 0


class TestWorkflowManager:
    """Test WorkflowManager class."""

    @pytest.fixture
    def base_config(self):
        """Create base workflow configuration."""
        return WorkflowConfiguration(
            project_root="/base/project",
            output_directory="/base/output",
            test_image_name="base-tests",
        )

    @pytest.fixture
    def manager(self, base_config):
        """Create workflow manager."""
        return WorkflowManager(base_config)

    @pytest.mark.asyncio
    async def test_create_workflow(self, manager):
        """Test creating a new workflow."""
        with patch("src.orchestration.test_generation_workflow.TestGenerationWorkflow"):
            workflow_id = await manager.create_workflow("/test/project", "/test/output")

            assert workflow_id in manager.active_workflows
            assert workflow_id not in manager.completed_workflows

    @pytest.mark.asyncio
    async def test_create_workflow_with_custom_config(self, manager):
        """Test creating workflow with custom configuration."""
        custom_config = {
            "test_timeout_minutes": 60,
            "cleanup_containers": False,
        }

        with patch(
            "src.orchestration.test_generation_workflow.TestGenerationWorkflow"
        ) as mock_workflow:
            await manager.create_workflow(
                "/test/project", "/test/output", custom_config
            )

            # Verify custom config was applied
            workflow_config = mock_workflow.call_args[0][0]
            assert workflow_config.test_timeout_minutes == 60
            assert workflow_config.cleanup_containers is False

    @pytest.mark.asyncio
    async def test_run_workflow_success(self, manager):
        """Test running a workflow successfully."""
        # Create workflow
        with patch(
            "src.orchestration.test_generation_workflow.TestGenerationWorkflow"
        ) as mock_workflow_class:
            mock_workflow = Mock()
            mock_workflow.run_complete_workflow = AsyncMock(
                return_value=WorkflowExecutionResult(
                    workflow_id="test-123",
                    success=True,
                    generation_result=None,
                    container_config=None,
                    docker_compose_path=None,
                    test_image_id=None,
                    execution_logs=[],
                    error_messages=[],
                    execution_time_seconds=10.0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            mock_workflow_class.return_value = mock_workflow

            workflow_id = await manager.create_workflow("/test/project", "/test/output")

            # Mock analyses
            doc_analysis = Mock()
            source_analysis = Mock()

            result = await manager.run_workflow(
                workflow_id, doc_analysis, source_analysis
            )

            assert result.success is True
            assert workflow_id not in manager.active_workflows
            assert workflow_id in manager.completed_workflows

    @pytest.mark.asyncio
    async def test_run_workflow_not_found(self, manager):
        """Test running a non-existent workflow."""
        doc_analysis = Mock()
        source_analysis = Mock()

        with pytest.raises(ValueError, match="Workflow .* not found"):
            await manager.run_workflow("nonexistent", doc_analysis, source_analysis)

    def test_get_workflow_status_active(self, manager):
        """Test getting status of active workflow."""
        with patch(
            "src.orchestration.test_generation_workflow.TestGenerationWorkflow"
        ) as mock_workflow_class:
            mock_workflow = Mock()
            mock_workflow.workflow_id = "test-123"
            mock_workflow.execution_logs = ["Log 1", "Log 2"]
            mock_workflow.error_messages = []
            mock_workflow_class.return_value = mock_workflow

            manager.active_workflows["test-123"] = mock_workflow

            status = manager.get_workflow_status("test-123")

            assert status["status"] == "active"
            assert status["workflow_id"] == "test-123"
            assert "logs" in status
            assert "errors" in status

    def test_get_workflow_status_completed(self, manager):
        """Test getting status of completed workflow."""
        result = WorkflowExecutionResult(
            workflow_id="test-123",
            success=True,
            generation_result=None,
            container_config=None,
            docker_compose_path=None,
            test_image_id=None,
            execution_logs=[],
            error_messages=[],
            execution_time_seconds=15.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        manager.completed_workflows["test-123"] = result

        status = manager.get_workflow_status("test-123")

        assert status["status"] == "completed"
        assert status["success"] is True
        assert status["workflow_id"] == "test-123"
        assert status["execution_time"] == 15.0

    def test_get_workflow_status_not_found(self, manager):
        """Test getting status of non-existent workflow."""
        status = manager.get_workflow_status("nonexistent")

        assert status["status"] == "not_found"
        assert status["workflow_id"] == "nonexistent"

    def test_list_workflows(self, manager):
        """Test listing workflows."""
        # Add mock workflows
        with patch("src.orchestration.test_generation_workflow.TestGenerationWorkflow"):
            manager.active_workflows["active-1"] = Mock()
            manager.active_workflows["active-2"] = Mock()

        manager.completed_workflows["completed-1"] = Mock()

        workflows = manager.list_workflows()

        assert set(workflows["active"]) == {"active-1", "active-2"}
        assert workflows["completed"] == ["completed-1"]
