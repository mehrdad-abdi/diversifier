"""Tests for workflow management functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from src.orchestration.workflow_manager import (
    WorkflowManager,
    WorkflowConfiguration,
)
from src.orchestration.acceptance_test_generator import (
    AcceptanceTestGenerator,
    WorkflowExecutionResult,
    AcceptanceTestGenerationResult,
    AcceptanceTestSuite,
    AcceptanceTestScenario,
)
from src.orchestration.mcp_manager import MCPManager, MCPServerType
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
            execute_tests=True,
            model_name="gpt-3.5-turbo",
            cleanup_containers=False,
        )

        assert config.project_root == "/test/project"
        assert config.output_directory == "/test/output"
        assert config.docker_registry == "test-registry"
        assert config.test_image_name == "test-image"
        assert config.test_image_tag == "v1.0"
        assert config.execute_tests is True
        assert config.model_name == "gpt-3.5-turbo"
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
        assert config.execute_tests is False
        assert config.model_name == "gpt-4"
        assert config.cleanup_containers is True


class TestWorkflowManager:
    """Test WorkflowManager class."""

    @pytest.fixture
    def base_config(self):
        """Create base workflow configuration."""
        return WorkflowConfiguration(
            project_root="/base/project",
            output_directory="/base/output",
            test_image_name="base-tests",
            execute_tests=False,
        )

    @pytest.fixture
    def manager(self, base_config):
        """Create workflow manager."""
        return WorkflowManager(base_config)

    @pytest.mark.asyncio
    async def test_create_workflow(self, manager):
        """Test creating a new workflow."""
        with (
            patch("src.orchestration.workflow_manager.MCPManager"),
            patch(
                "src.orchestration.workflow_manager.AcceptanceTestGenerator"
            ) as mock_generator_class,
        ):
            mock_generator = Mock()
            mock_generator.workflow_id = "test-123"
            mock_generator_class.return_value = mock_generator

            workflow_id = await manager.create_workflow("/test/project", "/test/output")

            assert workflow_id == "test-123"
            assert workflow_id in manager.active_workflows
            assert workflow_id not in manager.completed_workflows

    @pytest.mark.asyncio
    async def test_create_workflow_with_custom_config(self, manager):
        """Test creating workflow with custom configuration."""
        custom_config = {
            "execute_tests": True,
            "model_name": "gpt-3.5-turbo",
            "cleanup_containers": False,
        }

        with (
            patch("src.orchestration.workflow_manager.MCPManager"),
            patch(
                "src.orchestration.workflow_manager.AcceptanceTestGenerator"
            ) as mock_generator_class,
        ):
            mock_generator = Mock()
            mock_generator.workflow_id = "test-456"
            mock_generator_class.return_value = mock_generator

            workflow_id = await manager.create_workflow(
                "/test/project", "/test/output", custom_config
            )

            assert workflow_id in manager.active_workflows

    @pytest.mark.asyncio
    async def test_run_workflow_success(self, manager):
        """Test running a workflow successfully."""
        # Create workflow
        with (
            patch("src.orchestration.workflow_manager.MCPManager"),
            patch(
                "src.orchestration.workflow_manager.AcceptanceTestGenerator"
            ) as mock_generator_class,
        ):
            mock_generator = Mock()
            mock_generator.workflow_id = "test-123"
            mock_generator.run_complete_workflow = AsyncMock(
                return_value=WorkflowExecutionResult(
                    workflow_id="test-123",
                    success=True,
                    generation_result=None,
                    docker_compose_path=None,
                    test_image_id=None,
                    execution_logs=[],
                    error_messages=[],
                    execution_time_seconds=10.0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            mock_generator_class.return_value = mock_generator

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
        mock_generator = Mock()
        mock_generator.workflow_id = "test-123"
        mock_generator.execution_logs = ["Log 1", "Log 2"]
        mock_generator.error_messages = []

        manager.active_workflows["test-123"] = mock_generator

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
        manager.active_workflows["active-1"] = Mock()
        manager.active_workflows["active-2"] = Mock()
        manager.completed_workflows["completed-1"] = Mock()

        workflows = manager.list_workflows()

        assert set(workflows["active"]) == {"active-1", "active-2"}
        assert workflows["completed"] == ["completed-1"]

    @pytest.mark.asyncio
    async def test_get_workflow_generator(self, manager):
        """Test getting workflow generator."""
        mock_generator = Mock()
        manager.active_workflows["test-123"] = mock_generator

        generator = await manager.get_workflow_generator("test-123")
        assert generator == mock_generator

        # Test non-existent workflow
        generator = await manager.get_workflow_generator("nonexistent")
        assert generator is None

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, manager):
        """Test cancelling a workflow."""
        mock_generator = Mock()
        mock_generator.workflow_id = "test-123"
        mock_generator.execution_logs = ["Log 1"]
        mock_generator.error_messages = []
        manager.active_workflows["test-123"] = mock_generator

        success = await manager.cancel_workflow("test-123")

        assert success is True
        assert "test-123" not in manager.active_workflows
        assert "test-123" in manager.completed_workflows

        # Verify cancelled workflow result
        result = manager.completed_workflows["test-123"]
        assert result.success is False
        assert "Workflow cancelled" in result.error_messages

    @pytest.mark.asyncio
    async def test_cancel_workflow_not_found(self, manager):
        """Test cancelling a non-existent workflow."""
        success = await manager.cancel_workflow("nonexistent")
        assert success is False


class TestAcceptanceTestGeneratorWorkflowMethods:
    """Test the new workflow methods added to AcceptanceTestGenerator."""

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
    def generator(self, mock_mcp_manager):
        """Create an AcceptanceTestGenerator instance."""
        return AcceptanceTestGenerator("/test/project", mock_mcp_manager)

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
            docker_compose_content="version: '3.8'",
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

    def test_generator_workflow_initialization(self, generator):
        """Test generator has workflow state management."""
        assert generator.workflow_id.startswith("workflow_")
        assert isinstance(generator.execution_logs, list)
        assert isinstance(generator.error_messages, list)
        assert len(generator.execution_logs) == 0
        assert len(generator.error_messages) == 0

    def test_log_methods(self, generator):
        """Test logging methods."""
        generator._log("Test message")
        generator._log_error("Test error")

        assert len(generator.execution_logs) == 1
        assert len(generator.error_messages) == 1
        assert "Test message" in generator.execution_logs[0]
        assert "Test error" in generator.error_messages[0]

    @pytest.mark.asyncio
    async def test_initialize_mcp_servers_success(self, generator, mock_mcp_manager):
        """Test successful MCP server initialization."""
        results = await generator.initialize_mcp_servers()

        assert results[MCPServerType.FILESYSTEM] is True
        assert results[MCPServerType.DOCKER] is True
        mock_mcp_manager.initialize_all_servers.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_mcp_servers_missing_required(
        self, generator, mock_mcp_manager
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
            await generator.initialize_mcp_servers()

    @pytest.mark.asyncio
    async def test_build_test_container_success(
        self, generator, sample_generation_result, mock_mcp_manager
    ):
        """Test successful test container building."""
        mock_mcp_manager.call_tool.return_value = {
            "success": True,
            "image_id": "sha256:abc123",
        }

        result = await generator.build_test_container(
            sample_generation_result, "/test/output"
        )

        assert result == "sha256:abc123"
        mock_mcp_manager.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_test_container_docker_unavailable(
        self, generator, sample_generation_result, mock_mcp_manager
    ):
        """Test test container building when Docker is unavailable."""
        mock_mcp_manager.is_server_available.return_value = False

        result = await generator.build_test_container(
            sample_generation_result, "/test/output"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_run_complete_workflow_success(
        self,
        generator,
        sample_doc_analysis,
        sample_source_analysis,
        sample_generation_result,
        mock_mcp_manager,
    ):
        """Test complete workflow execution success."""
        # Mock methods
        generator.initialize_mcp_servers = AsyncMock(
            return_value={
                MCPServerType.FILESYSTEM: True,
                MCPServerType.DOCKER: True,
            }
        )
        generator.generate_acceptance_tests = AsyncMock(
            return_value=sample_generation_result
        )
        generator.export_test_suites = AsyncMock(return_value="/test/output")

        result = await generator.run_complete_workflow(
            sample_doc_analysis, sample_source_analysis
        )

        assert isinstance(result, WorkflowExecutionResult)
        assert result.success is True
        assert result.generation_result == sample_generation_result
        assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_run_complete_workflow_failure(
        self,
        generator,
        sample_doc_analysis,
        sample_source_analysis,
        mock_mcp_manager,
    ):
        """Test complete workflow execution failure."""
        # Mock initialization failure
        generator.initialize_mcp_servers = AsyncMock(
            side_effect=RuntimeError("MCP initialization failed")
        )

        result = await generator.run_complete_workflow(
            sample_doc_analysis, sample_source_analysis
        )

        assert isinstance(result, WorkflowExecutionResult)
        assert result.success is False
        assert result.generation_result is None
        assert len(result.error_messages) > 0
        assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_wait_for_service_health_success(self, generator, mock_mcp_manager):
        """Test waiting for service health success."""
        mock_mcp_manager.call_tool.return_value = {"health_status": "healthy"}

        # Should not raise exception
        await generator._wait_for_service_health("test-container", timeout_seconds=10)

        mock_mcp_manager.call_tool.assert_called_with(
            MCPServerType.DOCKER,
            "inspect_container",
            {"container_name": "test-container"},
        )

    @pytest.mark.asyncio
    async def test_wait_for_service_health_timeout(self, generator, mock_mcp_manager):
        """Test waiting for service health timeout."""
        mock_mcp_manager.call_tool.return_value = {"health_status": "unhealthy"}

        with pytest.raises(TimeoutError, match="did not become healthy"):
            await generator._wait_for_service_health(
                "test-container", timeout_seconds=1
            )

    @pytest.mark.asyncio
    async def test_collect_container_logs(self, generator, mock_mcp_manager):
        """Test collecting container logs."""
        mock_mcp_manager.call_tool.side_effect = [
            {"logs": "App logs"},
            {"logs": "Test logs"},
        ]

        logs = await generator._collect_container_logs(
            ["app-container", "test-container"]
        )

        assert logs["app-container"] == "App logs"
        assert logs["test-container"] == "Test logs"
        assert mock_mcp_manager.call_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_containers_success(self, generator, mock_mcp_manager):
        """Test container cleanup success."""
        mock_mcp_manager.call_tool.return_value = {"success": True}

        # Should not raise exception
        await generator._cleanup_containers("/test/compose.yml")

        mock_mcp_manager.call_tool.assert_called_with(
            MCPServerType.DOCKER,
            "compose_down",
            {
                "compose_file": "/test/compose.yml",
                "remove_volumes": True,
                "remove_images": "local",
            },
        )
