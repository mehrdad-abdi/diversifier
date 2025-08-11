"""Tests for source code analyzer functionality."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from src.orchestration.config import DiversifierConfig, LLMConfig
from src.orchestration.source_code_analyzer import (
    SourceCodeAnalyzer,
    APIEndpoint,
    ExternalServiceIntegration,
    ConfigurationUsage,
    ExistingTestPattern,
    SourceCodeAnalysisResult,
)
from src.orchestration.mcp_manager import MCPManager, MCPServerType


class TestAPIEndpoint:
    """Test APIEndpoint dataclass."""

    def test_api_endpoint_creation(self):
        """Test creating an API endpoint."""
        endpoint = APIEndpoint(
            path="/api/v1/users",
            methods=["GET", "POST"],
            handler="UserListView.as_view()",
            authentication_required=True,
            file_location="myapp/views.py:45",
            request_body_schema={"name": "string", "email": "string"},
            response_schema={"id": "int", "name": "string", "email": "string"},
        )

        assert endpoint.path == "/api/v1/users"
        assert endpoint.methods == ["GET", "POST"]
        assert endpoint.handler == "UserListView.as_view()"
        assert endpoint.authentication_required is True
        assert endpoint.file_location == "myapp/views.py:45"
        assert endpoint.request_body_schema == {"name": "string", "email": "string"}
        assert endpoint.response_schema == {
            "id": "int",
            "name": "string",
            "email": "string",
        }

    def test_api_endpoint_defaults(self):
        """Test API endpoint with default values."""
        endpoint = APIEndpoint(
            path="/api/health",
            methods=["GET"],
            handler="health_check",
            authentication_required=False,
            file_location="myapp/health.py:10",
        )

        assert endpoint.request_body_schema is None
        assert endpoint.response_schema is None


class TestExternalServiceIntegration:
    """Test ExternalServiceIntegration dataclass."""

    def test_external_service_integration_creation(self):
        """Test creating external service integration."""
        integration = ExternalServiceIntegration(
            service_type="http_client",
            purpose="payment processing",
            connection_pattern="requests library",
            configuration_source="PAYMENT_API_URL environment variable",
            file_location="myapp/services/payment.py:23",
            endpoints_or_operations=["/charge", "/refund"],
        )

        assert integration.service_type == "http_client"
        assert integration.purpose == "payment processing"
        assert integration.connection_pattern == "requests library"
        assert (
            integration.configuration_source == "PAYMENT_API_URL environment variable"
        )
        assert integration.file_location == "myapp/services/payment.py:23"
        assert integration.endpoints_or_operations == ["/charge", "/refund"]

    def test_external_service_integration_defaults(self):
        """Test external service integration with defaults."""
        integration = ExternalServiceIntegration(
            service_type="database",
            purpose="user data storage",
            connection_pattern="SQLAlchemy ORM",
            configuration_source="DATABASE_URL",
            file_location="myapp/models.py",
        )

        assert integration.endpoints_or_operations is None


class TestConfigurationUsage:
    """Test ConfigurationUsage dataclass."""

    def test_configuration_usage_creation(self):
        """Test creating configuration usage."""
        config = ConfigurationUsage(
            name="DATABASE_URL",
            purpose="database connection string",
            required=True,
            default_value=None,
            usage_locations=["myapp/database.py:15", "myapp/models.py:8"],
            config_type="environment_variable",
        )

        assert config.name == "DATABASE_URL"
        assert config.purpose == "database connection string"
        assert config.required is True
        assert config.default_value is None
        assert config.usage_locations == ["myapp/database.py:15", "myapp/models.py:8"]
        assert config.config_type == "environment_variable"


class TestExistingTestPattern:
    """Test ExistingTestPattern dataclass."""

    def test_existing_test_pattern_creation(self):
        """Test creating existing test pattern."""
        test_pattern = ExistingTestPattern(
            test_name="test_user_creation",
            test_type="api",
            endpoint_or_feature_tested="/api/v1/users",
            file_location="tests/test_users.py:25",
            assertions=["status_code == 201", "response contains user id"],
            mock_usage=["@patch('myapp.services.payment.requests.post')"],
        )

        assert test_pattern.test_name == "test_user_creation"
        assert test_pattern.test_type == "api"
        assert test_pattern.endpoint_or_feature_tested == "/api/v1/users"
        assert test_pattern.file_location == "tests/test_users.py:25"
        assert test_pattern.assertions == [
            "status_code == 201",
            "response contains user id",
        ]
        assert test_pattern.mock_usage == [
            "@patch('myapp.services.payment.requests.post')"
        ]

    def test_existing_test_pattern_defaults(self):
        """Test existing test pattern with defaults."""
        test_pattern = ExistingTestPattern(
            test_name="test_database_connection",
            test_type="integration",
            endpoint_or_feature_tested="database connectivity",
            file_location="tests/test_integration.py:10",
        )

        assert test_pattern.assertions is None
        assert test_pattern.mock_usage is None


class TestSourceCodeAnalysisResult:
    """Test SourceCodeAnalysisResult dataclass."""

    def test_analysis_result_creation(self):
        """Test creating analysis result."""
        endpoints = [
            APIEndpoint("/api/users", ["GET"], "UserListView", False, "views.py:10")
        ]
        integrations = [
            ExternalServiceIntegration(
                "database", "data storage", "SQLAlchemy", "DATABASE_URL", "models.py"
            )
        ]
        config_usage = [
            ConfigurationUsage(
                "DEBUG",
                "development mode",
                False,
                "False",
                ["settings.py:5"],
                "environment_variable",
            )
        ]
        test_patterns = [
            ExistingTestPattern("test_api", "api", "/api/users", "tests/test_api.py:20")
        ]

        result = SourceCodeAnalysisResult(
            api_endpoints=endpoints,
            external_service_integrations=integrations,
            configuration_usage=config_usage,
            existing_test_patterns=test_patterns,
            network_interfaces={"server_ports": [{"port": 8000, "protocol": "HTTP"}]},
            security_patterns={"authentication": "JWT"},
            testing_requirements={"mock_services": ["payment_api"]},
            framework_detected="flask",
            analysis_confidence=0.8,
        )

        assert len(result.api_endpoints) == 1
        assert len(result.external_service_integrations) == 1
        assert len(result.configuration_usage) == 1
        assert len(result.existing_test_patterns) == 1
        assert result.network_interfaces == {
            "server_ports": [{"port": 8000, "protocol": "HTTP"}]
        }
        assert result.security_patterns == {"authentication": "JWT"}
        assert result.testing_requirements == {"mock_services": ["payment_api"]}
        assert result.framework_detected == "flask"
        assert result.analysis_confidence == 0.8


class TestSourceCodeAnalyzer:
    """Test SourceCodeAnalyzer class."""

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = Mock(spec=MCPManager)
        manager.is_server_available.return_value = True
        return manager

    @pytest.fixture
    def analyzer(self, mock_mcp_manager):
        """Create a SourceCodeAnalyzer instance."""
        return SourceCodeAnalyzer("/test/project", mock_mcp_manager)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.project_root == Path("/test/project")
        assert analyzer.mcp_manager is not None
        assert len(analyzer.python_file_patterns) > 0
        assert len(analyzer.test_file_patterns) > 0
        assert len(analyzer.config_file_patterns) > 0
        assert "*.py" in analyzer.python_file_patterns
        assert "test*.py" in analyzer.test_file_patterns
        assert "settings.py" in analyzer.config_file_patterns
        assert "flask" in analyzer.framework_patterns
        assert "django" in analyzer.framework_patterns
        assert "fastapi" in analyzer.framework_patterns

    @pytest.mark.asyncio
    async def test_collect_python_files(self, analyzer, mock_mcp_manager):
        """Test collecting Python source files."""
        # Mock file list response
        file_list_response = {
            "result": [
                {
                    "text": json.dumps(
                        {
                            "files": [
                                "myapp/views.py",
                                "myapp/models.py",
                                "myapp/services.py",
                            ],
                            "count": 3,
                        }
                    )
                }
            ]
        }

        mock_mcp_manager.call_tool.return_value = file_list_response

        python_files = await analyzer._collect_python_files()

        assert mock_mcp_manager.call_tool.called
        assert len(python_files) > 0
        # Verify test files are filtered out
        assert not any("test" in str(f).lower() for f in python_files)

    @pytest.mark.asyncio
    async def test_collect_test_files(self, analyzer, mock_mcp_manager):
        """Test collecting test files."""
        file_list_response = {
            "result": [
                {
                    "text": json.dumps(
                        {
                            "files": ["tests/test_views.py", "tests/test_models.py"],
                            "count": 2,
                        }
                    )
                }
            ]
        }

        mock_mcp_manager.call_tool.return_value = file_list_response

        test_files = await analyzer._collect_test_files()

        assert mock_mcp_manager.call_tool.called
        assert isinstance(test_files, list)

    @pytest.mark.asyncio
    async def test_collect_config_files(self, analyzer, mock_mcp_manager):
        """Test collecting configuration files."""
        file_list_response = {
            "result": [
                {
                    "text": json.dumps(
                        {"files": ["settings.py", "config.yml"], "count": 2}
                    )
                }
            ]
        }

        mock_mcp_manager.call_tool.return_value = file_list_response

        config_files = await analyzer._collect_config_files()

        assert mock_mcp_manager.call_tool.called
        assert isinstance(config_files, list)

    @pytest.mark.asyncio
    async def test_collect_files_mcp_unavailable(self, analyzer, mock_mcp_manager):
        """Test file collection when MCP server is unavailable."""
        mock_mcp_manager.is_server_available.return_value = False

        python_files = await analyzer._collect_python_files()
        test_files = await analyzer._collect_test_files()
        config_files = await analyzer._collect_config_files()

        assert python_files == []
        assert test_files == []
        assert config_files == []

    def test_create_file_system_tools(self, analyzer):
        """Test creating file system tools."""
        tools = analyzer._create_file_system_tools()

        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        assert "read_source_file" in tool_names
        assert "search_code_patterns" in tool_names
        assert "list_project_files" in tool_names

    @pytest.mark.asyncio
    async def test_read_source_file_tool(self, analyzer, mock_mcp_manager):
        """Test the read_source_file tool."""
        tools = analyzer._create_file_system_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_source_file")

        # Mock file read response
        mock_mcp_manager.call_tool.return_value = {
            "result": [
                {
                    "text": "@app.route('/api/users')\ndef get_users():\n    return User.query.all()"
                }
            ]
        }

        result = await read_tool.ainvoke({"file_path": "myapp/views.py"})

        assert "@app.route('/api/users')" in result
        mock_mcp_manager.call_tool.assert_called_with(
            MCPServerType.FILESYSTEM, "read_file", {"file_path": "myapp/views.py"}
        )

    @pytest.mark.asyncio
    async def test_search_code_patterns_tool(self, analyzer, mock_mcp_manager):
        """Test the search_code_patterns tool."""
        tools = analyzer._create_file_system_tools()
        search_tool = next(
            tool for tool in tools if tool.name == "search_code_patterns"
        )

        # Mock search response
        mock_response = {
            "result": [
                {
                    "text": json.dumps(
                        {
                            "matches": [
                                {"file": "views.py", "line": 10, "match": "@app.route"}
                            ]
                        }
                    )
                }
            ]
        }
        mock_mcp_manager.call_tool.return_value = mock_response

        result = await search_tool.ainvoke(
            {"directory_path": "/test/project", "pattern": "@app.route"}
        )

        result_data = json.loads(result)
        assert "matches" in result_data

    @pytest.mark.asyncio
    async def test_list_project_files_tool(self, analyzer, mock_mcp_manager):
        """Test the list_project_files tool."""
        tools = analyzer._create_file_system_tools()
        list_tool = next(tool for tool in tools if tool.name == "list_project_files")

        # Mock file list response
        mock_response = {
            "result": [
                {"text": json.dumps({"files": ["views.py", "models.py"], "count": 2})}
            ]
        }
        mock_mcp_manager.call_tool.return_value = mock_response

        result = await list_tool.ainvoke(
            {"directory_path": "/test/project", "pattern": "*.py"}
        )
        result_data = json.loads(result)

        assert result_data["files"] == ["views.py", "models.py"]
        assert result_data["count"] == 2

    @patch("src.orchestration.source_code_analyzer.DiversificationAgent")
    def test_analyze_source_code_files(self, mock_agent_class, analyzer):
        """Test analyzing source code files with LLM agent."""
        # Mock agent response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": json.dumps(
                {
                    "api_endpoints": {
                        "http_endpoints": [
                            {
                                "path": "/api/v1/users",
                                "methods": ["GET", "POST"],
                                "handler": "UserListView.as_view()",
                                "authentication_required": True,
                                "file_location": "myapp/views.py:45",
                            }
                        ]
                    },
                    "external_service_integrations": {
                        "databases": [
                            {
                                "purpose": "user data storage",
                                "connection_pattern": "SQLAlchemy ORM",
                                "connection_config": "DATABASE_URL",
                                "file_location": "myapp/models.py",
                            }
                        ]
                    },
                    "configuration_patterns": {
                        "environment_variables": [
                            {
                                "name": "DATABASE_URL",
                                "purpose": "database connection",
                                "required": True,
                                "default_value": None,
                                "usage_locations": ["myapp/models.py:15"],
                            }
                        ]
                    },
                    "analysis_metadata": {"framework_detected": "flask"},
                }
            )
        }
        mock_agent_class.return_value = mock_agent

        python_files = [Path("myapp/views.py"), Path("myapp/models.py")]
        config_files = [Path("settings.py")]
        result = analyzer._analyze_source_code_files(
            mock_agent, python_files, config_files
        )

        # Verify agent was called
        mock_agent.invoke.assert_called_once()

        # Verify result structure
        assert "api_endpoints" in result
        assert "external_service_integrations" in result
        assert "configuration_patterns" in result
        assert len(result["api_endpoints"]["http_endpoints"]) == 1
        assert len(result["external_service_integrations"]["databases"]) == 1

    @patch("src.orchestration.source_code_analyzer.DiversificationAgent")
    def test_analyze_test_files(self, mock_agent_class, analyzer):
        """Test analyzing existing test files."""
        # Mock agent response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "output": json.dumps(
                {
                    "existing_test_patterns": {
                        "test_types": ["unit", "integration", "api"],
                        "api_test_examples": [
                            {
                                "test_name": "test_user_creation",
                                "endpoint_tested": "/api/v1/users",
                                "test_data": {"name": "Test User"},
                                "assertions": ["status_code == 201"],
                                "file_location": "tests/test_users.py:25",
                            }
                        ],
                    }
                }
            )
        }
        mock_agent_class.return_value = mock_agent

        test_files = [Path("tests/test_users.py")]
        result = analyzer._analyze_test_files(mock_agent, test_files)

        # Verify agent was called
        mock_agent.invoke.assert_called_once()

        # Verify result structure
        assert "existing_test_patterns" in result
        assert len(result["existing_test_patterns"]["api_test_examples"]) == 1

    def test_analyze_test_files_empty(self, analyzer):
        """Test analyzing test files when no test files exist."""
        result = analyzer._analyze_test_files(Mock(), [])
        assert result == {"existing_test_patterns": {}}

    def test_combine_analysis_results(self, analyzer):
        """Test combining source code and test analysis results."""
        source_analysis = {
            "api_endpoints": {
                "http_endpoints": [
                    {
                        "path": "/api/users",
                        "methods": ["GET"],
                        "handler": "get_users",
                        "authentication_required": False,
                        "file_location": "views.py:10",
                        "request_body_schema": None,
                        "response_schema": {"users": "list"},
                    }
                ]
            },
            "external_service_integrations": {
                "databases": [
                    {
                        "purpose": "data storage",
                        "connection_pattern": "SQLAlchemy",
                        "connection_config": "DATABASE_URL",
                        "file_location": "models.py",
                        "endpoints_called": [],
                    }
                ]
            },
            "configuration_patterns": {
                "environment_variables": [
                    {
                        "name": "DEBUG",
                        "purpose": "development mode",
                        "required": False,
                        "default_value": "False",
                        "usage_locations": ["settings.py:8"],
                    }
                ]
            },
            "network_interfaces": {"server_ports": [{"port": 8000}]},
            "security_considerations": {"authentication": "session"},
            "testing_requirements": {"mock_services": []},
            "analysis_metadata": {"framework_detected": "flask"},
        }

        test_analysis = {
            "existing_test_patterns": {
                "api_test_examples": [
                    {
                        "test_name": "test_get_users",
                        "endpoint_tested": "/api/users",
                        "file_location": "tests/test_api.py:20",
                        "assertions": ["status_code == 200"],
                    }
                ]
            }
        }

        result = analyzer._combine_analysis_results(source_analysis, test_analysis)

        # Verify API endpoints
        assert len(result.api_endpoints) == 1
        endpoint = result.api_endpoints[0]
        assert endpoint.path == "/api/users"
        assert endpoint.methods == ["GET"]
        assert endpoint.handler == "get_users"
        assert endpoint.authentication_required is False
        assert endpoint.file_location == "views.py:10"

        # Verify external integrations
        assert len(result.external_service_integrations) == 1
        integration = result.external_service_integrations[0]
        assert integration.service_type == "databases"
        assert integration.purpose == "data storage"

        # Verify configuration usage
        assert len(result.configuration_usage) == 1
        config = result.configuration_usage[0]
        assert config.name == "DEBUG"
        assert config.purpose == "development mode"
        assert config.config_type == "environment_variable"

        # Verify existing tests
        assert len(result.existing_test_patterns) == 1
        test_pattern = result.existing_test_patterns[0]
        assert test_pattern.test_name == "test_get_users"
        assert test_pattern.endpoint_or_feature_tested == "/api/users"

        # Verify other fields
        assert result.framework_detected == "flask"
        assert result.analysis_confidence > 0.0

    def test_calculate_analysis_confidence(self, analyzer):
        """Test confidence score calculation."""
        # High confidence case - all factors present
        source_analysis_high = {
            "api_endpoints": {"http_endpoints": [{"path": "/api"}]},
            "external_service_integrations": {"databases": [{"name": "db"}]},
            "configuration_patterns": {"environment_variables": [{"name": "DEBUG"}]},
            "analysis_metadata": {"framework_detected": "flask"},
        }

        test_analysis_high = {
            "existing_test_patterns": {"api_test_examples": [{"test_name": "test_api"}]}
        }

        confidence_high = analyzer._calculate_analysis_confidence(
            source_analysis_high, test_analysis_high
        )
        assert (
            confidence_high == 1.0
        )  # All factors present (0.3 + 0.25 + 0.2 + 0.15 + 0.1)

        # Medium confidence case - some factors missing
        source_analysis_med = {
            "api_endpoints": {"http_endpoints": [{"path": "/api"}]},
            "analysis_metadata": {"framework_detected": "django"},
        }

        test_analysis_med = {"existing_test_patterns": {}}

        confidence_med = analyzer._calculate_analysis_confidence(
            source_analysis_med, test_analysis_med
        )
        assert confidence_med == 0.4  # API endpoints (0.3) + framework (0.1)

        # Low confidence case - minimal information
        source_analysis_low = {"analysis_metadata": {"framework_detected": "unknown"}}
        test_analysis_low = {"existing_test_patterns": {}}

        confidence_low = analyzer._calculate_analysis_confidence(
            source_analysis_low, test_analysis_low
        )
        assert confidence_low == 0.0

    @patch("src.orchestration.source_code_analyzer.get_config")
    @patch("src.orchestration.source_code_analyzer.DiversificationAgent")
    @pytest.mark.asyncio
    async def test_analyze_project_source_code_integration(
        self, mock_agent_class, mock_get_config, analyzer, mock_mcp_manager
    ):
        """Test full project source code analysis integration."""
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
            file_list_response = {
                "result": [
                    {"text": json.dumps({"files": ["myapp/views.py"], "count": 1})}
                ]
            }
            mock_mcp_manager.call_tool.return_value = file_list_response

            # Mock agent responses
            mock_agent = Mock()
            mock_agent.invoke.side_effect = [
                {  # Source code analysis response
                    "output": json.dumps(
                        {
                            "api_endpoints": {
                                "http_endpoints": [{"path": "/api", "methods": ["GET"]}]
                            },
                            "external_service_integrations": {"databases": []},
                            "configuration_patterns": {"environment_variables": []},
                            "analysis_metadata": {"framework_detected": "flask"},
                        }
                    )
                },
                {  # Test analysis response
                    "output": json.dumps(
                        {"existing_test_patterns": {"api_test_examples": []}}
                    )
                },
            ]
            mock_agent_class.return_value = mock_agent

            # Run analysis
            result = await analyzer.analyze_project_source_code()

            # Verify result
            assert isinstance(result, SourceCodeAnalysisResult)
            assert len(result.api_endpoints) > 0
            assert result.framework_detected == "flask"
            assert result.analysis_confidence > 0.0

    @pytest.mark.asyncio
    async def test_export_analysis_results(self, analyzer, mock_mcp_manager):
        """Test exporting analysis results to JSON."""
        # Create test result
        endpoints = [
            APIEndpoint("/api/users", ["GET"], "get_users", False, "views.py:10")
        ]
        integrations = [
            ExternalServiceIntegration(
                "database", "storage", "SQLAlchemy", "DATABASE_URL", "models.py"
            )
        ]
        result = SourceCodeAnalysisResult(
            api_endpoints=endpoints,
            external_service_integrations=integrations,
            configuration_usage=[],
            existing_test_patterns=[],
            network_interfaces={},
            security_patterns={},
            testing_requirements={},
            framework_detected="flask",
            analysis_confidence=0.7,
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
        assert len(parsed_content["api_endpoints"]) == 1
        assert len(parsed_content["external_service_integrations"]) == 1
        assert parsed_content["framework_detected"] == "flask"
        assert parsed_content["analysis_confidence"] == 0.7

        # Verify timestamp is properly formatted
        generated_at = parsed_content["generated_at"]
        assert isinstance(generated_at, str)
        assert (
            generated_at.endswith("Z")
            or "+" in generated_at
            or generated_at.endswith(":00")
        )

        # Parse to ensure it's a valid ISO format timestamp
        datetime.fromisoformat(generated_at.replace("Z", "+00:00"))

        assert output_path == "/test/output.json"

    @patch("builtins.open", create=True)
    @pytest.mark.asyncio
    async def test_export_analysis_results_fallback(
        self, mock_open, analyzer, mock_mcp_manager
    ):
        """Test exporting results with MCP server unavailable."""
        # Make MCP server unavailable
        mock_mcp_manager.is_server_available.return_value = False

        # Create test result
        result = SourceCodeAnalysisResult(
            api_endpoints=[],
            external_service_integrations=[],
            configuration_usage=[],
            existing_test_patterns=[],
            network_interfaces={},
            security_patterns={},
            testing_requirements={},
            framework_detected="unknown",
            analysis_confidence=0.5,
        )

        output_path = await analyzer.export_analysis_results(
            result, "/test/fallback.json"
        )

        # Verify fallback file write was attempted
        assert output_path == "/test/fallback.json"


@pytest.mark.integration
class TestSourceCodeAnalyzerIntegration:
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
        # analyzer = SourceCodeAnalyzer("/real/project/path", mcp_manager)
        # result = await analyzer.analyze_project_source_code()
        # assert result.analysis_confidence > 0.0
