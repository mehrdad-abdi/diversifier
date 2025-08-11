"""Tests for acceptance test generator functionality."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.orchestration.acceptance_test_generator import (
    AcceptanceTestGenerator,
    AcceptanceTestSuite,
    AcceptanceTestScenario,
    AcceptanceTestGenerationResult,
)
from src.orchestration.config import DiversifierConfig, LLMConfig
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


class TestAcceptanceTestSuite:
    """Test AcceptanceTestSuite dataclass."""

    def test_acceptance_test_suite_creation(self):
        """Test creating an acceptance test suite."""
        suite = AcceptanceTestSuite(
            name="http_api_tests",
            description="HTTP API acceptance tests",
            test_file_content="import pytest\n\ndef test_example():\n    pass",
            docker_compose_content="version: '3.8'",
            dockerfile_content="FROM python:3.11",
            requirements_content="pytest>=7.0.0",
        )

        assert suite.name == "http_api_tests"
        assert suite.description == "HTTP API acceptance tests"
        assert "import pytest" in suite.test_file_content
        assert suite.docker_compose_content == "version: '3.8'"
        assert suite.dockerfile_content == "FROM python:3.11"
        assert suite.requirements_content == "pytest>=7.0.0"

    def test_acceptance_test_suite_defaults(self):
        """Test acceptance test suite with default values."""
        suite = AcceptanceTestSuite(
            name="basic_tests",
            description="Basic tests",
            test_file_content="def test_basic(): pass",
        )

        assert suite.docker_compose_content is None
        assert suite.dockerfile_content is None
        assert suite.requirements_content is None


class TestAcceptanceTestScenario:
    """Test TestScenario dataclass."""

    def test_test_scenario_creation(self):
        """Test creating a test scenario."""
        scenario = AcceptanceTestScenario(
            category="http_api",
            name="Test user creation endpoint",
            description="Test POST /api/users endpoint",
            test_method_name="test_create_user",
            test_code="def test_create_user():\n    pass",
            dependencies=["requests"],
            docker_services=["app", "db"],
        )

        assert scenario.category == "http_api"
        assert scenario.name == "Test user creation endpoint"
        assert scenario.description == "Test POST /api/users endpoint"
        assert scenario.test_method_name == "test_create_user"
        assert scenario.dependencies == ["requests"]
        assert scenario.docker_services == ["app", "db"]


class TestAcceptanceTestGenerationResult:
    """Test AcceptanceTestGenerationResult dataclass."""

    def test_generation_result_creation(self):
        """Test creating a generation result."""
        suite = AcceptanceTestSuite(
            name="test_suite", description="Test", test_file_content="pass"
        )
        scenario = AcceptanceTestScenario(
            category="http_api",
            name="Test API",
            description="Test",
            test_method_name="test_api",
            test_code="pass",
            dependencies=[],
            docker_services=[],
        )

        result = AcceptanceTestGenerationResult(
            test_suites=[suite],
            test_scenarios=[scenario],
            docker_configuration={"network": "test"},
            test_dependencies=["pytest"],
            coverage_analysis={"total": 1},
            generation_confidence=0.8,
        )

        assert len(result.test_suites) == 1
        assert len(result.test_scenarios) == 1
        assert result.docker_configuration == {"network": "test"}
        assert result.test_dependencies == ["pytest"]
        assert result.coverage_analysis == {"total": 1}
        assert result.generation_confidence == 0.8


class TestAcceptanceTestGenerator:
    """Test AcceptanceTestGenerator class."""

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = Mock(spec=MCPManager)
        manager.is_server_available.return_value = True
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
                path="/api/v1/users",
                methods=["GET", "POST"],
                handler="UserListView",
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

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.project_root == Path("/test/project")
        assert generator.mcp_manager is not None
        assert generator._framework_config_cache is None
        assert generator._docker_config_cache is None

    def test_create_file_system_tools(self, generator):
        """Test creating file system tools."""
        tools = generator._create_file_system_tools()

        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "write_test_file" in tool_names
        assert "read_existing_test" in tool_names

    @pytest.mark.asyncio
    async def test_write_test_file_tool(self, generator, mock_mcp_manager):
        """Test the write_test_file tool."""
        tools = generator._create_file_system_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_test_file")

        # Mock successful file write
        mock_mcp_manager.call_tool.return_value = {"success": True}

        result = await write_tool.ainvoke(
            {"file_path": "/test/test_api.py", "content": "def test_api(): pass"}
        )

        assert "Successfully wrote test file" in result
        mock_mcp_manager.call_tool.assert_called_with(
            MCPServerType.FILESYSTEM,
            "write_file",
            {"file_path": "/test/test_api.py", "content": "def test_api(): pass"},
        )

    @pytest.mark.asyncio
    async def test_read_existing_test_tool(self, generator, mock_mcp_manager):
        """Test the read_existing_test tool."""
        tools = generator._create_file_system_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_existing_test")

        # Mock file read response
        mock_mcp_manager.call_tool.return_value = {
            "result": [{"text": "def test_example(): assert True"}]
        }

        result = await read_tool.ainvoke({"file_path": "/test/existing_test.py"})

        assert "def test_example(): assert True" in result
        mock_mcp_manager.call_tool.assert_called_with(
            MCPServerType.FILESYSTEM,
            "read_file",
            {"file_path": "/test/existing_test.py"},
        )

    def test_extract_response_text(self, generator):
        """Test extracting response text from different result formats."""
        # Test output format
        result_output = {"output": "Generated test code"}
        assert generator._extract_response_text(result_output) == "Generated test code"

        # Test messages format
        mock_message = Mock()
        mock_message.content = "Test content from message"
        result_messages = {"messages": [mock_message]}
        assert (
            generator._extract_response_text(result_messages)
            == "Test content from message"
        )

        # Test fallback format
        result_other = {"data": "some other format"}
        assert "data" in generator._extract_response_text(result_other)

    def test_parse_test_response(self, generator):
        """Test parsing LLM response into test scenarios."""
        # Valid JSON response
        json_response = """```json
{
    "test_scenarios": [
        {
            "name": "Test User API",
            "description": "Test user creation endpoint",
            "test_method": "test_create_user",
            "test_code": "def test_create_user(): pass",
            "dependencies": ["requests"],
            "docker_services": ["app"]
        }
    ]
}
```"""

        result = generator._parse_test_response(json_response, "http_api")

        assert "scenarios" in result
        scenarios = result["scenarios"]
        assert len(scenarios) == 1

        scenario = scenarios[0]
        assert scenario.category == "http_api"
        assert scenario.name == "Test User API"
        assert scenario.test_method_name == "test_create_user"
        assert scenario.dependencies == ["requests"]
        assert scenario.docker_services == ["app"]

    def test_parse_test_response_invalid_json(self, generator):
        """Test parsing invalid JSON response."""
        invalid_response = "This is not JSON"

        result = generator._parse_test_response(invalid_response, "http_api")

        assert result == {"scenarios": []}

    def test_generate_test_file_content(self, generator, sample_source_analysis):
        """Test generating complete test file content."""
        scenarios = [
            AcceptanceTestScenario(
                category="http_api",
                name="Test Users API",
                description="Test user endpoint functionality",
                test_method_name="test_users_endpoint",
                test_code="        response = requests.get(f'{BASE_URL}/api/v1/users')\n        assert response.status_code == 200",
                dependencies=["requests"],
                docker_services=["app"],
            )
        ]

        framework_config = {
            "default_port": 5000,
            "health_endpoint": "/health",
            "python_version": "3.11",
        }

        content = generator._generate_test_file_content(
            "http_api", scenarios, sample_source_analysis, framework_config
        )

        # Verify content structure
        assert "import pytest" in content
        assert "import requests" in content
        assert "class TestHttpApi:" in content
        assert "def test_users_endpoint(self):" in content
        assert "Test user endpoint functionality" in content
        assert "BASE_URL" in content
        assert "http://app:5000" in content  # Check dynamic port
        assert "get_app_container_name()" in content
        assert "return 5000" in content  # Check dynamic port in helper function

    @pytest.mark.asyncio
    async def test_generate_docker_compose_content(
        self, generator, sample_source_analysis, sample_doc_analysis
    ):
        """Test generating Docker Compose content."""
        with patch(
            "src.orchestration.acceptance_test_generator.DiversificationAgent"
        ) as mock_agent_class:
            mock_agent = Mock()
            mock_agent.invoke.return_value = {
                "output": """version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
  test:
    build:
      dockerfile: Dockerfile.test"""
            }
            mock_agent_class.return_value = mock_agent

            framework_config = {
                "default_port": 5000,
                "health_endpoint": "/health",
                "python_version": "3.11",
                "base_docker_image": "python:3.11-slim",
            }

            content = await generator._generate_docker_compose_content(
                mock_agent,
                sample_doc_analysis,
                sample_source_analysis,
                framework_config,
            )

            assert "version:" in content
            assert "services:" in content
            assert "app:" in content
            assert "5000:5000" in content  # Flask default port

    def test_generate_requirements_content(self, generator):
        """Test generating requirements.txt content."""
        content = generator._generate_requirements_content()

        assert "pytest" in content
        assert "requests" in content
        assert "websocket-client" in content
        assert "docker" in content

    def test_get_test_dependencies(self, generator):
        """Test getting test dependencies."""
        dependencies = generator._get_test_dependencies()

        assert "pytest" in dependencies
        assert "requests" in dependencies
        assert "websocket-client" in dependencies
        assert "docker" in dependencies
        assert "pytest-asyncio" in dependencies

    @pytest.mark.asyncio
    async def test_generate_docker_configuration(
        self, generator, sample_doc_analysis, sample_source_analysis
    ):
        """Test generating Docker configuration."""
        with patch(
            "src.orchestration.acceptance_test_generator.DiversificationAgent"
        ) as mock_agent_class:
            mock_agent = Mock()
            # Mock framework discovery response
            mock_agent.invoke.side_effect = [
                {
                    "output": json.dumps(
                        {
                            "default_port": 5000,
                            "health_endpoint": "/health",
                            "python_version": "3.11",
                            "base_docker_image": "python:3.11-slim",
                        }
                    )
                },
                {"output": "version: '3.8'\nservices:\n  app:\n    build: ."},
            ]
            mock_agent_class.return_value = mock_agent

            config = await generator._generate_docker_configuration(
                mock_agent, sample_doc_analysis, sample_source_analysis
            )

            assert "test_dockerfile" in config
            assert "docker_compose" in config
            assert "network_config" in config
            assert "environment_variables" in config

            # Verify structure
            assert config["network_config"]["app_service"] == "app"
            assert config["network_config"]["test_service"] == "test"
            assert "BASE_URL" in config["environment_variables"]
            assert "http://app:5000" in config["environment_variables"]["BASE_URL"]

    def test_analyze_test_coverage(self, generator, sample_source_analysis):
        """Test analyzing test coverage."""
        scenarios = [
            AcceptanceTestScenario(
                "http_api", "Test 1", "desc", "test1", "code", [], []
            ),
            AcceptanceTestScenario(
                "http_api", "Test 2", "desc", "test2", "code", [], []
            ),
            AcceptanceTestScenario(
                "network", "Test 3", "desc", "test3", "code", [], []
            ),
            AcceptanceTestScenario("config", "Test 4", "desc", "test4", "code", [], []),
            AcceptanceTestScenario("error", "Test 5", "desc", "test5", "code", [], []),
        ]

        coverage = generator._analyze_test_coverage(scenarios, sample_source_analysis)

        assert coverage["total_scenarios"] == 5
        assert coverage["categories_covered"] == 4
        assert coverage["category_breakdown"]["http_api"] == 2
        assert coverage["category_breakdown"]["network"] == 1

        # Check coverage percentages
        assert "api_coverage_percentage" in coverage
        assert "service_coverage_percentage" in coverage
        assert "config_coverage_percentage" in coverage

    def test_calculate_generation_confidence(
        self, generator, sample_doc_analysis, sample_source_analysis
    ):
        """Test calculating generation confidence."""
        # High confidence - many scenarios across categories
        high_scenarios = [
            AcceptanceTestScenario(
                "http_api", "Test 1", "desc", "test1", "code", [], []
            ),
            AcceptanceTestScenario(
                "http_api", "Test 2", "desc", "test2", "code", [], []
            ),
            AcceptanceTestScenario(
                "network", "Test 3", "desc", "test3", "code", [], []
            ),
            AcceptanceTestScenario("config", "Test 4", "desc", "test4", "code", [], []),
            AcceptanceTestScenario("error", "Test 5", "desc", "test5", "code", [], []),
            AcceptanceTestScenario("docker", "Test 6", "desc", "test6", "code", [], []),
            AcceptanceTestScenario(
                "service_lifecycle", "Test 7", "desc", "test7", "code", [], []
            ),
            AcceptanceTestScenario(
                "http_api", "Test 8", "desc", "test8", "code", [], []
            ),
            AcceptanceTestScenario(
                "network", "Test 9", "desc", "test9", "code", [], []
            ),
            AcceptanceTestScenario(
                "config", "Test 10", "desc", "test10", "code", [], []
            ),
            AcceptanceTestScenario(
                "error", "Test 11", "desc", "test11", "code", [], []
            ),
        ]

        confidence = generator._calculate_generation_confidence(
            high_scenarios, sample_doc_analysis, sample_source_analysis
        )

        # Should be high confidence (close to 1.0)
        assert confidence >= 0.8

        # Medium confidence - fewer scenarios
        medium_scenarios = [
            AcceptanceTestScenario(
                "http_api", "Test 1", "desc", "test1", "code", [], []
            ),
            AcceptanceTestScenario(
                "network", "Test 2", "desc", "test2", "code", [], []
            ),
            AcceptanceTestScenario("config", "Test 3", "desc", "test3", "code", [], []),
        ]

        confidence_med = generator._calculate_generation_confidence(
            medium_scenarios, sample_doc_analysis, sample_source_analysis
        )

        # Should be medium confidence
        assert 0.3 <= confidence_med <= 0.8

        # Low confidence - very few scenarios
        low_scenarios = [
            AcceptanceTestScenario("other", "Test 1", "desc", "test1", "code", [], [])
        ]

        confidence_low = generator._calculate_generation_confidence(
            low_scenarios, sample_doc_analysis, sample_source_analysis
        )

        # Should be low confidence
        assert confidence_low <= 0.3

    def test_create_test_suites(self, generator, sample_source_analysis):
        """Test creating organized test suites from scenarios."""
        scenarios = [
            AcceptanceTestScenario(
                "http_api", "API Test 1", "desc1", "test1", "code1", [], []
            ),
            AcceptanceTestScenario(
                "http_api", "API Test 2", "desc2", "test2", "code2", [], []
            ),
            AcceptanceTestScenario(
                "network", "Network Test", "desc3", "test3", "code3", [], []
            ),
            AcceptanceTestScenario(
                "config", "Config Test", "desc4", "test4", "code4", [], []
            ),
        ]

        framework_config = {
            "default_port": 5000,
            "health_endpoint": "/health",
            "python_version": "3.11",
            "base_docker_image": "python:3.11-slim",
        }

        test_suites = generator._create_test_suites(
            scenarios, sample_source_analysis, framework_config
        )

        # Should have suites for each category plus Docker config
        suite_names = [suite.name for suite in test_suites]

        assert "http_api_acceptance_tests" in suite_names
        assert "network_acceptance_tests" in suite_names
        assert "config_acceptance_tests" in suite_names
        assert "docker_configuration" in suite_names

        # Check that HTTP API suite contains both scenarios
        http_suite = next(
            s for s in test_suites if s.name == "http_api_acceptance_tests"
        )
        assert "test1" in http_suite.test_file_content
        assert "test2" in http_suite.test_file_content
        assert "http://app:5000" in http_suite.test_file_content  # Check dynamic port

        # Check Docker configuration suite
        docker_suite = next(s for s in test_suites if s.name == "docker_configuration")
        # Docker compose content will be empty initially, set by caller
        assert docker_suite.dockerfile_content is not None
        assert docker_suite.requirements_content is not None
        assert (
            "python:3.11-slim" in docker_suite.dockerfile_content
        )  # Check dynamic base image

    @patch("src.orchestration.acceptance_test_generator.DiversificationAgent")
    @pytest.mark.asyncio
    async def test_generate_http_api_tests(
        self, mock_agent_class, generator, sample_source_analysis
    ):
        """Test generating HTTP API tests."""
        # Mock agent response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": json.dumps(
                {
                    "test_scenarios": [
                        {
                            "name": "Test User List API",
                            "description": "Test GET /api/v1/users endpoint",
                            "test_method": "test_get_users",
                            "test_code": "response = requests.get(f'{BASE_URL}/api/v1/users')",
                            "dependencies": ["requests"],
                            "docker_services": ["app"],
                        }
                    ]
                }
            )
        }
        mock_agent_class.return_value = mock_agent

        result = await generator._generate_http_api_tests(
            mock_agent, sample_source_analysis
        )

        # Verify agent was called with appropriate prompt
        mock_agent.invoke.assert_called_once()
        call_args = mock_agent.invoke.call_args[0][0]
        assert "API Endpoints Discovered:" in call_args
        assert "/api/v1/users" in call_args
        assert "flask" in call_args

        # Verify result structure
        assert "scenarios" in result
        scenarios = result["scenarios"]
        assert len(scenarios) == 1

        scenario = scenarios[0]
        assert scenario.category == "http_api"
        assert scenario.name == "Test User List API"
        assert scenario.test_method_name == "test_get_users"

    @patch("src.orchestration.acceptance_test_generator.DiversificationAgent")
    @pytest.mark.asyncio
    async def test_generate_service_lifecycle_tests(
        self, mock_agent_class, generator, sample_doc_analysis, sample_source_analysis
    ):
        """Test generating service lifecycle tests."""
        # Mock agent response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": json.dumps(
                {
                    "test_scenarios": [
                        {
                            "name": "Test Application Startup",
                            "description": "Test that application starts successfully",
                            "test_method": "test_app_startup",
                            "test_code": "wait_for_service_ready(f'{BASE_URL}/health')",
                            "dependencies": ["requests"],
                            "docker_services": ["app"],
                        }
                    ]
                }
            )
        }
        mock_agent_class.return_value = mock_agent

        result = await generator._generate_service_lifecycle_tests(
            mock_agent, sample_doc_analysis, sample_source_analysis
        )

        # Verify agent was called
        mock_agent.invoke.assert_called_once()
        call_args = mock_agent.invoke.call_args[0][0]
        assert "service lifecycle tests" in call_args
        assert "Framework: flask" in call_args

        # Verify result
        assert "scenarios" in result
        scenarios = result["scenarios"]
        assert len(scenarios) == 1
        assert scenarios[0].category == "service_lifecycle"

    @pytest.mark.asyncio
    async def test_generate_acceptance_tests_integration(
        self, generator, sample_doc_analysis, sample_source_analysis
    ):
        """Test full acceptance test generation integration."""
        with (
            patch(
                "src.orchestration.acceptance_test_generator.DiversificationAgent"
            ) as mock_agent_class,
            patch(
                "src.orchestration.acceptance_test_generator.get_config"
            ) as mock_get_config,
        ):
            # Mock configuration
            mock_llm_config = Mock(spec=LLMConfig)
            mock_config = Mock(spec=DiversifierConfig)
            mock_config.llm = mock_llm_config
            mock_get_config.return_value = mock_config

            # Mock agent responses for all test generation methods
            mock_agent = Mock()

            # Return different responses for each category + framework discovery + docker generation
            mock_agent.invoke.side_effect = [
                {"output": json.dumps({"test_scenarios": []})},  # HTTP tests
                {"output": json.dumps({"test_scenarios": []})},  # Lifecycle tests
                {"output": json.dumps({"test_scenarios": []})},  # Network tests
                {"output": json.dumps({"test_scenarios": []})},  # Config tests
                {"output": json.dumps({"test_scenarios": []})},  # Error tests
                {"output": json.dumps({"test_scenarios": []})},  # Docker tests
                {
                    "output": json.dumps(
                        {  # Framework discovery
                            "default_port": 5000,
                            "health_endpoint": "/health",
                            "python_version": "3.11",
                            "base_docker_image": "python:3.11-slim",
                        }
                    )
                },
                {
                    "output": "version: '3.8'\nservices:\n  app:\n    build: ."
                },  # Docker compose generation
            ]

            mock_agent_class.return_value = mock_agent

            # Run full generation
            result = await generator.generate_acceptance_tests(
                sample_doc_analysis, sample_source_analysis
            )

            # Verify result structure
            assert isinstance(result, AcceptanceTestGenerationResult)
            assert isinstance(result.test_suites, list)
            assert isinstance(result.test_scenarios, list)
            assert isinstance(result.docker_configuration, dict)
            assert isinstance(result.test_dependencies, list)
            assert isinstance(result.coverage_analysis, dict)
            assert 0.0 <= result.generation_confidence <= 1.0

            # Verify agent was called multiple times for different test categories + framework discovery + docker
            assert mock_agent.invoke.call_count == 8

    @pytest.mark.asyncio
    async def test_export_test_suites(self, generator, mock_mcp_manager):
        """Test exporting test suites to files."""
        # Create sample test generation result
        suite1 = AcceptanceTestSuite(
            name="http_api_tests",
            description="HTTP API tests",
            test_file_content="def test_api(): pass",
        )

        suite2 = AcceptanceTestSuite(
            name="docker_configuration",
            description="Docker config",
            test_file_content="",
            docker_compose_content="version: '3.8'",
            dockerfile_content="FROM python:3.11",
            requirements_content="pytest>=7.0.0",
        )

        result = AcceptanceTestGenerationResult(
            test_suites=[suite1, suite2],
            test_scenarios=[],
            docker_configuration={},
            test_dependencies=["pytest"],
            coverage_analysis={"total": 1},
            generation_confidence=0.8,
        )

        # Mock successful file writes
        mock_mcp_manager.call_tool.return_value = {"success": True}

        output_dir = await generator.export_test_suites(result, "/test/output")

        # Verify MCP calls were made for each file
        expected_calls = (
            5  # http_api_tests + compose + dockerfile + requirements + metadata
        )
        assert mock_mcp_manager.call_tool.call_count == expected_calls

        # Verify output directory
        assert output_dir == "/test/output"

    @pytest.mark.asyncio
    async def test_export_test_suites_mcp_unavailable(
        self, generator, mock_mcp_manager
    ):
        """Test exporting test suites when MCP server is unavailable."""
        mock_mcp_manager.is_server_available.return_value = False

        suite = AcceptanceTestSuite(
            name="basic_tests",
            description="Basic tests",
            test_file_content="def test_basic(): pass",
        )

        result = AcceptanceTestGenerationResult(
            test_suites=[suite],
            test_scenarios=[],
            docker_configuration={},
            test_dependencies=[],
            coverage_analysis={},
            generation_confidence=0.5,
        )

        # Should not raise exception, should use fallback
        output_dir = await generator.export_test_suites(result, "/test/fallback")

        assert output_dir == "/test/fallback"
        # MCP should not have been called
        assert mock_mcp_manager.call_tool.call_count == 0

    @pytest.mark.asyncio
    async def test_discover_framework_configuration(
        self, generator, sample_doc_analysis, sample_source_analysis
    ):
        """Test dynamic framework configuration discovery."""
        with patch(
            "src.orchestration.acceptance_test_generator.DiversificationAgent"
        ) as mock_agent_class:
            mock_agent = Mock()
            mock_agent.invoke.return_value = {
                "output": json.dumps(
                    {
                        "default_port": 5000,
                        "health_endpoint": "/health",
                        "api_prefix": "/api",
                        "test_patterns": ["test_*.py"],
                        "python_version": "3.11",
                        "base_docker_image": "python:3.11-slim",
                        "framework_specific": {"wsgi_server": "flask"},
                    }
                )
            }
            mock_agent_class.return_value = mock_agent

            config = await generator._discover_framework_configuration(
                mock_agent, sample_doc_analysis, sample_source_analysis
            )

            # Verify configuration structure
            assert config["default_port"] == 5000
            assert config["health_endpoint"] == "/health"
            assert config["python_version"] == "3.11"
            assert "framework_specific" in config

            # Verify caching
            assert generator._framework_config_cache == config

            # Second call should use cache
            config2 = await generator._discover_framework_configuration(
                mock_agent, sample_doc_analysis, sample_source_analysis
            )
            assert config2 == config
            # Agent should only be called once due to caching
            assert mock_agent.invoke.call_count == 1

    @pytest.mark.asyncio
    async def test_discover_framework_configuration_fallback(
        self, generator, sample_doc_analysis, sample_source_analysis
    ):
        """Test framework configuration discovery with fallback."""
        with patch(
            "src.orchestration.acceptance_test_generator.DiversificationAgent"
        ) as mock_agent_class:
            mock_agent = Mock()
            # Simulate JSON decode error
            mock_agent.invoke.return_value = {"output": "Invalid JSON response"}
            mock_agent_class.return_value = mock_agent

            config = await generator._discover_framework_configuration(
                mock_agent, sample_doc_analysis, sample_source_analysis
            )

            # Should fall back to Flask configuration (from sample_source_analysis)
            assert config["default_port"] == 5000
            assert config["health_endpoint"] == "/health"
            assert config["framework_specific"]["wsgi_server"] == "flask"

    def test_get_fallback_framework_config(self, generator):
        """Test fallback framework configuration generation."""
        # Test Flask fallback
        flask_config = generator._get_fallback_framework_config("flask")
        assert flask_config["default_port"] == 5000
        assert flask_config["framework_specific"]["wsgi_server"] == "flask"

        # Test Django fallback
        django_config = generator._get_fallback_framework_config("django")
        assert django_config["default_port"] == 8000
        assert django_config["framework_specific"]["wsgi_server"] == "django"

        # Test FastAPI fallback
        fastapi_config = generator._get_fallback_framework_config("fastapi")
        assert fastapi_config["default_port"] == 8000
        assert fastapi_config["framework_specific"]["asgi_server"] == "fastapi"

        # Test unknown framework fallback
        unknown_config = generator._get_fallback_framework_config("unknown")
        assert unknown_config["default_port"] == 8000
        assert unknown_config["framework_specific"]["generic"] is True

    def test_generate_dockerfile_content(self, generator):
        """Test Dockerfile content generation."""
        framework_config = {
            "python_version": "3.11",
            "base_docker_image": "python:3.11-slim",
        }

        dockerfile = generator._generate_dockerfile_content(framework_config)

        assert "FROM python:3.11-slim" in dockerfile
        assert "WORKDIR /app" in dockerfile
        assert "pip install -r requirements.txt" in dockerfile
        assert 'CMD ["python", "-m", "pytest"' in dockerfile

    def test_get_fallback_docker_compose(self, generator):
        """Test fallback Docker Compose configuration generation."""
        framework_config = {
            "default_port": 5000,
            "health_endpoint": "/health",
            "python_version": "3.11",
        }

        compose_content = generator._get_fallback_docker_compose(framework_config)

        assert "version: '3.8'" in compose_content
        assert "5000:5000" in compose_content  # Dynamic port
        assert "http://localhost:5000/health" in compose_content  # Dynamic health check
        assert "BASE_URL=http://app:5000" in compose_content  # Dynamic base URL


@pytest.mark.integration
class TestAcceptanceTestGeneratorIntegration:
    """Integration tests requiring actual MCP servers."""

    def test_real_project_test_generation(self):
        """Test test generation on a real project structure.

        Note: This test requires actual MCP servers to be running.
        It's marked as integration and may be skipped in unit test runs.
        """
        pytest.skip("Integration test - requires running MCP servers")

        # This would test with actual MCP servers:
        # from src.orchestration.mcp_manager import MCPManager
        # mcp_manager = MCPManager()
        # await mcp_manager.initialize_filesystem_server()
        # generator = AcceptanceTestGenerator("/real/project/path", mcp_manager)
        #
        # # Use real analysis results
        # result = await generator.generate_acceptance_tests(doc_analysis, source_analysis)
        # assert result.generation_confidence > 0.0
