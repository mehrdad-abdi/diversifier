"""Tests for documentation analyzer functionality."""

import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.orchestration.config import DiversifierConfig, LLMConfig
from src.orchestration.doc_analyzer import (
    DocumentationAnalyzer,
    ExternalInterface,
    DockerServiceInfo,
    DocumentationAnalysisResult,
)
from src.orchestration.mcp_manager import MCPManager, MCPServerType


class TestExternalInterface:
    """Test ExternalInterface dataclass."""

    def test_external_interface_creation(self):
        """Test creating an external interface."""
        interface = ExternalInterface(
            type="http_api",
            name="/api/v1/users",
            description="User management API",
            port=8080,
            protocol="HTTP",
            authentication="Bearer token",
            required_for_testing=True,
        )

        assert interface.type == "http_api"
        assert interface.name == "/api/v1/users"
        assert interface.description == "User management API"
        assert interface.port == 8080
        assert interface.protocol == "HTTP"
        assert interface.authentication == "Bearer token"
        assert interface.required_for_testing is True

    def test_external_interface_defaults(self):
        """Test external interface with default values."""
        interface = ExternalInterface(
            type="database", name="user_db", description="User database"
        )

        assert interface.port is None
        assert interface.protocol is None
        assert interface.authentication is None
        assert interface.required_for_testing is True


class TestDockerServiceInfo:
    """Test DockerServiceInfo dataclass."""

    def test_docker_service_info_creation(self):
        """Test creating Docker service info."""
        service = DockerServiceInfo(
            name="web-app",
            container_name="myapp_web",
            exposed_ports=[8000, 8080],
            dependencies=["db", "redis"],
            environment_variables=["DATABASE_URL", "REDIS_URL"],
            health_check="/health",
        )

        assert service.name == "web-app"
        assert service.container_name == "myapp_web"
        assert service.exposed_ports == [8000, 8080]
        assert service.dependencies == ["db", "redis"]
        assert service.environment_variables == ["DATABASE_URL", "REDIS_URL"]
        assert service.health_check == "/health"

    def test_docker_service_info_defaults(self):
        """Test Docker service info with default values."""
        service = DockerServiceInfo(
            name="api",
            container_name=None,
            exposed_ports=[8000],
            dependencies=[],
            environment_variables=[],
        )

        assert service.health_check is None


class TestDocumentationAnalysisResult:
    """Test DocumentationAnalysisResult dataclass."""

    def test_analysis_result_creation(self):
        """Test creating analysis result."""
        interfaces = [
            ExternalInterface("http_api", "/api/users", "Users API"),
            ExternalInterface("database", "postgres", "Main database"),
        ]

        services = [DockerServiceInfo("web", "web_1", [8000], ["db"], ["DATABASE_URL"])]

        result = DocumentationAnalysisResult(
            external_interfaces=interfaces,
            docker_services=services,
            network_configuration={"ports": [8000]},
            testing_requirements={"mock_services": ["payment"]},
            deployment_patterns={"orchestration": "docker-compose"},
            analysis_confidence=0.8,
        )

        assert len(result.external_interfaces) == 2
        assert len(result.docker_services) == 1
        assert result.network_configuration == {"ports": [8000]}
        assert result.testing_requirements == {"mock_services": ["payment"]}
        assert result.deployment_patterns == {"orchestration": "docker-compose"}
        assert result.analysis_confidence == 0.8


class TestDocumentationAnalyzer:
    """Test DocumentationAnalyzer class."""

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = Mock(spec=MCPManager)
        manager.is_server_available.return_value = True
        return manager

    @pytest.fixture
    def analyzer(self, mock_mcp_manager):
        """Create a DocumentationAnalyzer instance."""
        return DocumentationAnalyzer("/test/project", mock_mcp_manager)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.project_root == Path("/test/project")
        assert analyzer.mcp_manager is not None
        assert len(analyzer.doc_patterns) > 0
        assert len(analyzer.config_patterns) > 0
        assert "README*" in analyzer.doc_patterns
        assert "docker-compose.yml" in analyzer.doc_patterns
        assert "pyproject.toml" in analyzer.config_patterns

    @pytest.mark.asyncio
    async def test_collect_documentation_files(self, analyzer, mock_mcp_manager):
        """Test collecting documentation files."""
        # Mock project structure response
        structure_response = {
            "result": [
                {
                    "text": json.dumps(
                        {
                            "config_files": ["README.md", "docs/api.md"],
                            "total_files": 10,
                        }
                    )
                }
            ]
        }

        # Mock file listing response
        file_list_response = {
            "result": [
                {
                    "text": json.dumps(
                        {
                            "files": ["README.md", "docker-compose.yml", "docs/api.md"],
                            "count": 3,
                        }
                    )
                }
            ]
        }

        mock_mcp_manager.call_tool.side_effect = [
            structure_response,
            file_list_response,
            file_list_response,  # Called multiple times for different patterns
        ]

        doc_files = await analyzer._collect_documentation_files()

        # Verify MCP calls were made
        assert mock_mcp_manager.call_tool.called
        assert len(doc_files) > 0

    @pytest.mark.asyncio
    async def test_collect_configuration_files(self, analyzer, mock_mcp_manager):
        """Test collecting configuration files."""
        file_list_response = {
            "result": [
                {
                    "text": json.dumps(
                        {"files": ["pyproject.toml", "requirements.txt"], "count": 2}
                    )
                }
            ]
        }

        mock_mcp_manager.call_tool.return_value = file_list_response

        config_files = await analyzer._collect_configuration_files()

        assert mock_mcp_manager.call_tool.called
        # Due to deduplication, length might vary
        assert isinstance(config_files, list)

    @pytest.mark.asyncio
    async def test_collect_files_mcp_unavailable(self, analyzer, mock_mcp_manager):
        """Test file collection when MCP server is unavailable."""
        mock_mcp_manager.is_server_available.return_value = False

        doc_files = await analyzer._collect_documentation_files()
        config_files = await analyzer._collect_configuration_files()

        assert doc_files == []
        assert config_files == []

    def test_create_file_system_tools(self, analyzer):
        """Test creating file system tools."""
        tools = analyzer._create_file_system_tools()

        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "read_documentation_file" in tool_names
        assert "list_project_files" in tool_names

    @pytest.mark.asyncio
    async def test_read_documentation_file_tool(self, analyzer, mock_mcp_manager):
        """Test the read_documentation_file tool."""
        tools = analyzer._create_file_system_tools()
        read_tool = next(
            tool for tool in tools if tool.name == "read_documentation_file"
        )

        # Mock file read response
        mock_mcp_manager.call_tool.return_value = {
            "result": [{"text": "# My Project\n\nThis is a web API..."}]
        }

        result = await read_tool.ainvoke({"file_path": "README.md"})

        assert "# My Project" in result
        mock_mcp_manager.call_tool.assert_called_with(
            MCPServerType.FILESYSTEM, "read_file", {"file_path": "README.md"}
        )

    @pytest.mark.asyncio
    async def test_list_project_files_tool(self, analyzer, mock_mcp_manager):
        """Test the list_project_files tool."""
        tools = analyzer._create_file_system_tools()
        list_tool = next(tool for tool in tools if tool.name == "list_project_files")

        # Mock file list response
        mock_response = {
            "result": [
                {"text": json.dumps({"files": ["file1.py", "file2.py"], "count": 2})}
            ]
        }
        mock_mcp_manager.call_tool.return_value = mock_response

        result = await list_tool.ainvoke(
            {"directory_path": "/test/project", "pattern": "*.py"}
        )
        result_data = json.loads(result)

        assert result_data["files"] == ["file1.py", "file2.py"]
        assert result_data["count"] == 2

    @patch("src.orchestration.doc_analyzer.DiversificationAgent")
    def test_analyze_documentation_files(self, mock_agent_class, analyzer):
        """Test analyzing documentation files with LLM agent."""
        # Mock agent response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": json.dumps(
                {
                    "external_interfaces": {
                        "http_endpoints": [
                            {
                                "path": "/api/v1/users",
                                "methods": ["GET", "POST"],
                                "description": "User API",
                                "port": 8000,
                            }
                        ],
                        "databases": [
                            {
                                "name": "user_db",
                                "type": "postgresql",
                                "default_port": 5432,
                            }
                        ],
                    },
                    "network_configuration": {"exposed_ports": [8000]},
                }
            )
        }
        mock_agent_class.return_value = mock_agent

        doc_files = [Path("README.md"), Path("API.md")]
        result = analyzer._analyze_documentation_files(mock_agent, doc_files)

        # Verify agent was called
        mock_agent.invoke.assert_called_once()

        # Verify result structure
        assert "external_interfaces" in result
        assert "network_configuration" in result
        assert len(result["external_interfaces"]["http_endpoints"]) == 1
        assert len(result["external_interfaces"]["databases"]) == 1

    @patch("src.orchestration.doc_analyzer.DiversificationAgent")
    def test_analyze_docker_configuration(self, mock_agent_class, analyzer):
        """Test analyzing Docker configuration files."""
        # Mock agent response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": json.dumps(
                {
                    "service_architecture": {
                        "primary_service": {
                            "name": "web-app",
                            "exposed_ports": [8000],
                            "dependencies": ["db"],
                        }
                    },
                    "network_configuration": {"networks": [{"name": "app-network"}]},
                }
            )
        }
        mock_agent_class.return_value = mock_agent

        config_files = [Path("docker-compose.yml"), Path("Dockerfile")]
        result = analyzer._analyze_docker_configuration(mock_agent, config_files)

        # Verify agent was called
        mock_agent.invoke.assert_called_once()

        # Verify result structure
        assert "service_architecture" in result
        assert "network_configuration" in result
        assert result["service_architecture"]["primary_service"]["name"] == "web-app"

    def test_combine_analysis_results(self, analyzer):
        """Test combining documentation and Docker analysis results."""
        doc_analysis = {
            "external_interfaces": {
                "http_endpoints": [
                    {
                        "path": "/api/users",
                        "methods": ["GET"],
                        "description": "User API",
                        "port": 8000,
                        "authentication": "Bearer",
                    }
                ],
                "databases": [
                    {"name": "user_db", "type": "postgresql", "default_port": 5432}
                ],
            },
            "network_configuration": {"exposed_ports": [8000]},
            "testing_considerations": {"mock_services": ["payment"]},
            "docker_requirements": {"base_image": "python:3.11"},
        }

        docker_analysis = {
            "service_architecture": {
                "primary_service": {
                    "name": "web-app",
                    "container_name": "app_web",
                    "exposed_ports": [8000],
                    "dependencies": ["db"],
                    "health_check": "/health",
                }
            },
            "network_configuration": {"networks": [{"name": "app-net"}]},
            "testing_setup": {"test_containers": ["test-runner"]},
            "deployment_patterns": {"orchestration": "docker-compose"},
        }

        result = analyzer._combine_analysis_results(doc_analysis, docker_analysis)

        # Verify external interfaces
        assert len(result.external_interfaces) == 2  # HTTP endpoint + database
        http_interface = next(
            i for i in result.external_interfaces if i.type == "http_api"
        )
        assert http_interface.name == "/api/users"
        assert http_interface.port == 8000
        assert http_interface.authentication == "Bearer"

        db_interface = next(
            i for i in result.external_interfaces if i.type == "database"
        )
        assert db_interface.name == "user_db"
        assert db_interface.port == 5432

        # Verify Docker services
        assert len(result.docker_services) == 1
        service = result.docker_services[0]
        assert service.name == "web-app"
        assert service.container_name == "app_web"
        assert service.exposed_ports == [8000]
        assert service.dependencies == ["db"]
        assert service.health_check == "/health"

        # Verify other fields
        assert result.network_configuration["doc_analysis"]["exposed_ports"] == [8000]
        assert result.network_configuration["docker_analysis"]["networks"] == [
            {"name": "app-net"}
        ]
        assert result.testing_requirements["doc_considerations"]["mock_services"] == [
            "payment"
        ]
        assert result.testing_requirements["docker_setup"]["test_containers"] == [
            "test-runner"
        ]
        assert (
            result.deployment_patterns["docker_requirements"]["base_image"]
            == "python:3.11"
        )
        assert (
            result.deployment_patterns["deployment_config"]["orchestration"]
            == "docker-compose"
        )

    def test_calculate_analysis_confidence(self, analyzer):
        """Test confidence score calculation."""
        # High confidence case - all factors present
        doc_analysis_high = {
            "external_interfaces": {
                "http_endpoints": [{"path": "/api"}],
                "databases": [{"name": "db"}],
            },
            "network_configuration": {"ports": [8000]},
            "testing_considerations": {"mock_services": ["payment"]},
        }

        docker_analysis_high = {
            "service_architecture": {"primary_service": {"name": "web"}},
            "network_configuration": {"networks": []},
            "testing_setup": {"containers": ["test"]},
        }

        confidence_high = analyzer._calculate_analysis_confidence(
            doc_analysis_high, docker_analysis_high
        )
        assert confidence_high == 1.0  # All factors present

        # Medium confidence case - some factors missing
        doc_analysis_med = {
            "external_interfaces": {"http_endpoints": [{"path": "/api"}]}
        }

        docker_analysis_med = {
            "service_architecture": {"primary_service": {"name": "web"}}
        }

        confidence_med = analyzer._calculate_analysis_confidence(
            doc_analysis_med, docker_analysis_med
        )
        assert confidence_med == 0.6  # Interface discovery (0.3) + Docker config (0.3)

        # Low confidence case - minimal information
        doc_analysis_low = {}
        docker_analysis_low = {}

        confidence_low = analyzer._calculate_analysis_confidence(
            doc_analysis_low, docker_analysis_low
        )
        assert confidence_low == 0.0

    @patch("src.orchestration.doc_analyzer.get_config")
    @patch("src.orchestration.doc_analyzer.DiversificationAgent")
    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    @pytest.mark.asyncio
    async def test_analyze_project_documentation_integration(
        self, mock_agent_class, mock_get_config, analyzer, mock_mcp_manager
    ):
        """Test full project documentation analysis integration."""
        # Mock configuration
        mock_llm_config = Mock(spec=LLMConfig)
        mock_llm_config.model_name = "test-model"
        mock_config = Mock(spec=DiversifierConfig)
        mock_config.llm = mock_llm_config
        mock_get_config.return_value = mock_config

        # Patch dataclasses.replace to handle mock objects
        def mock_replace(obj, **changes):
            for key, value in changes.items():
                setattr(obj, key, value)
            return obj

        with patch("dataclasses.replace", side_effect=mock_replace):
            # Mock file collection
            structure_response = {
                "result": [{"text": json.dumps({"config_files": [], "total_files": 0})}]
            }
            file_list_response = {
                "result": [{"text": json.dumps({"files": ["README.md"], "count": 1})}]
            }
            mock_mcp_manager.call_tool.side_effect = [
                structure_response,  # get_project_structure
                file_list_response,  # list_files calls for doc patterns
                file_list_response,
                file_list_response,  # list_files calls for config patterns
            ]

            # Mock agent responses
            mock_agent = Mock()
            mock_agent.invoke.side_effect = [
                {  # Documentation analysis response
                    "output": json.dumps(
                        {
                            "external_interfaces": {
                                "http_endpoints": [{"path": "/api", "port": 8000}]
                            },
                            "network_configuration": {"exposed_ports": [8000]},
                        }
                    )
                },
                {  # Docker analysis response
                    "output": json.dumps(
                        {
                            "service_architecture": {
                                "primary_service": {
                                    "name": "web",
                                    "exposed_ports": [8000],
                                }
                            },
                            "network_configuration": {"networks": []},
                        }
                    )
                },
            ]
            mock_agent_class.return_value = mock_agent

            # Create test LLM config
            test_llm_config = LLMConfig(
                provider="anthropic",
                model_name="claude-3-5-sonnet-20241022",
                api_key_env_var="TEST_API_KEY",
            )

            # Run analysis
            result = await analyzer.analyze_project_documentation(test_llm_config)

            # Verify result
            assert isinstance(result, DocumentationAnalysisResult)
            assert len(result.external_interfaces) > 0
            assert len(result.docker_services) > 0
            assert result.analysis_confidence > 0.0

    @pytest.mark.asyncio
    async def test_export_analysis_results(self, analyzer, mock_mcp_manager):
        """Test exporting analysis results to JSON."""
        # Create test result
        interfaces = [
            ExternalInterface("http_api", "/api/users", "Users API", 8000, "HTTP")
        ]
        services = [DockerServiceInfo("web", "web_1", [8000], ["db"], ["DATABASE_URL"])]
        result = DocumentationAnalysisResult(
            external_interfaces=interfaces,
            docker_services=services,
            network_configuration={},
            testing_requirements={},
            deployment_patterns={},
            analysis_confidence=0.8,
        )

        # Mock MCP call for file write
        mock_mcp_manager.call_tool.return_value = {"success": True}

        output_path = await analyzer.export_analysis_results(
            result, "/test/output.json"
        )

        # Verify MCP call was made
        mock_mcp_manager.call_tool.assert_called_once()
        call_args = mock_mcp_manager.call_tool.call_args
        assert call_args[0][0] == MCPServerType.FILESYSTEM
        assert call_args[0][1] == "write_file"
        assert call_args[0][2]["file_path"] == "/test/output.json"

        # Verify JSON structure
        written_content = call_args[0][2]["content"]
        parsed_content = json.loads(written_content)
        assert len(parsed_content["external_interfaces"]) == 1
        assert len(parsed_content["docker_services"]) == 1
        assert parsed_content["analysis_confidence"] == 0.8

        # Verify timestamp is current and properly formatted ISO 8601 with timezone
        generated_at = parsed_content["generated_at"]
        assert isinstance(generated_at, str)
        assert (
            generated_at.endswith("Z")
            or "+" in generated_at
            or generated_at.endswith(":00")
        )  # Valid timezone format
        # Parse to ensure it's a valid ISO format timestamp
        datetime.fromisoformat(generated_at.replace("Z", "+00:00"))

        assert output_path == "/test/output.json"

    @patch("builtins.open", new_callable=mock_open)
    @pytest.mark.asyncio
    async def test_export_analysis_results_fallback(
        self, mock_file, analyzer, mock_mcp_manager
    ):
        """Test exporting results with MCP server unavailable."""
        # Make MCP server unavailable
        mock_mcp_manager.is_server_available.return_value = False

        # Create test result
        interfaces = [ExternalInterface("http_api", "/api", "API")]
        result = DocumentationAnalysisResult(
            external_interfaces=interfaces,
            docker_services=[],
            network_configuration={},
            testing_requirements={},
            deployment_patterns={},
            analysis_confidence=0.5,
        )

        output_path = await analyzer.export_analysis_results(
            result, "/test/fallback.json"
        )

        # Verify fallback file write was used
        mock_file.assert_called_once_with("/test/fallback.json", "w")
        assert output_path == "/test/fallback.json"


@pytest.mark.integration
class TestDocumentationAnalyzerIntegration:
    """Integration tests requiring actual MCP servers."""

    def test_real_project_analysis(self):
        """Test analysis on a real project structure.

        Note: This test requires actual MCP servers to be running.
        It's marked as integration and may be skipped in unit test runs.
        """
        pytest.skip("Integration test - requires running MCP servers")

        # This would test with actual MCP servers:
        # from src.orchestration.mcp_manager import MCPManager
        # mcp_manager = MCPManager()
        # await mcp_manager.initialize_filesystem_server()
        # analyzer = DocumentationAnalyzer("/real/project/path", mcp_manager)
        # result = analyzer.analyze_project_documentation()
        # assert result.analysis_confidence > 0.0
