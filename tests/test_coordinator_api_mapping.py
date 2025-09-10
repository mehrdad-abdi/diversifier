"""Tests for DiversificationCoordinator API mapping functionality."""

import pytest
import tempfile
from unittest.mock import Mock, patch

from src.orchestration.coordinator import DiversificationCoordinator
from src.orchestration.config import LLMConfig, MigrationConfig


class TestCoordinatorAPIMapping:
    """Test cases for _find_api_mapping step in DiversificationCoordinator."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = self.temp_dir.name

        # Mock configurations
        self.llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key_env_var="OPENAI_API_KEY",
            temperature=0.2,
        )
        self.migration_config = MigrationConfig()

        # Create coordinator
        self.coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=self.llm_config,
            migration_config=self.migration_config,
        )

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @pytest.mark.asyncio
    async def test_find_api_mapping_web_server_not_available(self):
        """Test _find_api_mapping when Web MCP server is not available."""
        # Mock MCP manager to return False for web server availability
        self.coordinator.mcp_manager.is_server_available = Mock(return_value=False)

        result = await self.coordinator._find_api_mapping()

        assert result["success"] is False
        assert "Web MCP server not available" in result["error"]

    @pytest.mark.asyncio
    async def test_find_api_mapping_no_test_results(self):
        """Test _find_api_mapping when no test selection results are available."""
        # Mock MCP manager to return True for web server
        self.coordinator.mcp_manager.is_server_available = Mock(return_value=True)

        result = await self.coordinator._find_api_mapping()

        assert result["success"] is False
        assert "No test selection results available" in result["error"]

    @pytest.mark.asyncio
    async def test_find_api_mapping_no_api_usage_found(self):
        """Test _find_api_mapping when no API usage is found."""
        # Mock MCP manager
        self.coordinator.mcp_manager.is_server_available = Mock(return_value=True)

        # Mock library usage result with no matching APIs
        mock_module_usage = Mock()
        mock_module_usage.module_name = "os"  # Different library
        mock_module_usage.function_calls = ["listdir", "getcwd"]

        mock_library_usage_result = Mock()
        mock_library_usage_result.module_usages = [mock_module_usage]

        mock_selection_result = Mock()
        mock_selection_result.library_usage_result = mock_library_usage_result

        # Add to step results
        self.coordinator.step_results["select_tests"] = {
            "selection_result": mock_selection_result
        }

        result = await self.coordinator._find_api_mapping()

        assert result["success"] is True
        assert result["api_mappings"] == {}
        assert "No requests API usage found" in result["note"]

    @pytest.mark.asyncio
    async def test_find_api_mapping_successful(self):
        """Test successful _find_api_mapping execution."""
        # Mock MCP manager
        self.coordinator.mcp_manager.is_server_available = Mock(return_value=True)

        # Mock web connection
        mock_web_connection = Mock()
        mock_web_connection.call_tool = Mock(side_effect=self._mock_web_tool_calls)
        self.coordinator.mcp_manager.get_connection = Mock(
            return_value=mock_web_connection
        )

        # Mock library usage result with requests APIs
        mock_module_usage = Mock()
        mock_module_usage.module_name = "requests"
        mock_module_usage.function_calls = ["get", "post"]
        mock_module_usage.method_calls = ["session"]
        mock_module_usage.attribute_accesses = ["Session"]

        mock_library_usage_result = Mock()
        mock_library_usage_result.module_usages = [mock_module_usage]

        mock_selection_result = Mock()
        mock_selection_result.library_usage_result = mock_library_usage_result

        # Add to step results
        self.coordinator.step_results["select_tests"] = {
            "selection_result": mock_selection_result
        }

        # Mock LLM analysis
        with patch.object(
            self.coordinator, "_analyze_api_mappings_with_llm"
        ) as mock_llm:
            mock_llm.return_value = {
                "get": "get",
                "post": "post",
                "session": "AsyncClient",
                "Session": "AsyncClient",
            }

            result = await self.coordinator._find_api_mapping()

        assert result["success"] is True
        assert "get" in result["api_mappings"]
        assert "post" in result["api_mappings"]
        assert result["source_library"] == "requests"
        assert result["target_library"] == "httpx"
        assert len(result["used_apis"]) == 4

    def _mock_web_tool_calls(self, tool_name, params):
        """Mock web tool calls for testing."""
        if tool_name == "get_pypi_info":
            if params["package_name"] == "requests":
                return {
                    "info": {
                        "name": "requests",
                        "docs_url": "https://requests.readthedocs.io/",
                    }
                }
            elif params["package_name"] == "httpx":
                return {
                    "info": {
                        "name": "httpx",
                        "docs_url": "https://www.python-httpx.org/",
                    }
                }
        elif tool_name == "fetch_documentation":
            return "# Sample Documentation\nThis is sample documentation content."
        return None

    def test_extract_docs_url_with_docs_url(self):
        """Test extracting docs URL when docs_url field is present."""
        pypi_info = {"info": {"docs_url": "https://example.readthedocs.io/"}}

        result = self.coordinator._extract_docs_url(pypi_info, "example")

        assert result == "https://example.readthedocs.io/"

    def test_extract_docs_url_with_project_urls(self):
        """Test extracting docs URL from project_urls."""
        pypi_info = {
            "info": {
                "project_urls": {
                    "Documentation": "https://example-docs.com/",
                    "Homepage": "https://example.com/",
                }
            }
        }

        result = self.coordinator._extract_docs_url(pypi_info, "example")

        assert result == "https://example-docs.com/"

    def test_extract_docs_url_fallback(self):
        """Test docs URL fallback behavior."""
        pypi_info = {"info": {}}

        result = self.coordinator._extract_docs_url(pypi_info, "example")

        assert result == "https://example.readthedocs.io/"

    def test_extract_docs_url_no_pypi_info(self):
        """Test extracting docs URL when pypi_info is None."""
        result = self.coordinator._extract_docs_url(None, "example")

        assert result == ""

    @pytest.mark.asyncio
    @patch("src.orchestration.coordinator.init_chat_model")
    async def test_analyze_api_mappings_with_llm(self, mock_init_chat_model):
        """Test LLM API mapping analysis."""
        # Mock LLM model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = (
            '{"get": "get", "post": "post", "Session": "AsyncClient"}'
        )
        mock_model.invoke = Mock(return_value=mock_response)
        mock_init_chat_model.return_value = mock_model

        used_apis = {"get", "post", "Session"}
        source_docs = "Source documentation content"
        target_docs = "Target documentation content"

        result = await self.coordinator._analyze_api_mappings_with_llm(
            used_apis, source_docs, target_docs
        )

        assert result == {"get": "get", "post": "post", "Session": "AsyncClient"}
        mock_init_chat_model.assert_called_once()
        mock_model.invoke.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.orchestration.coordinator.init_chat_model")
    async def test_analyze_api_mappings_with_llm_json_parsing_error(
        self, mock_init_chat_model
    ):
        """Test LLM API mapping with JSON parsing error."""
        # Mock LLM model with invalid JSON response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = "Invalid JSON response"
        mock_model.invoke = Mock(return_value=mock_response)
        mock_init_chat_model.return_value = mock_model

        used_apis = {"get", "post"}
        source_docs = ""
        target_docs = ""

        result = await self.coordinator._analyze_api_mappings_with_llm(
            used_apis, source_docs, target_docs
        )

        assert result == {}

    @pytest.mark.asyncio
    async def test_find_api_mapping_llm_failure(self):
        """Test _find_api_mapping when LLM analysis fails."""
        # Mock MCP manager and web connection
        self.coordinator.mcp_manager.is_server_available = Mock(return_value=True)
        mock_web_connection = Mock()
        mock_web_connection.call_tool = Mock(return_value={"info": {}})
        self.coordinator.mcp_manager.get_connection = Mock(
            return_value=mock_web_connection
        )

        # Mock library usage
        mock_module_usage = Mock()
        mock_module_usage.module_name = "requests"
        mock_module_usage.function_calls = ["get"]
        mock_module_usage.method_calls = []
        mock_module_usage.attribute_accesses = []

        mock_library_usage_result = Mock()
        mock_library_usage_result.module_usages = [mock_module_usage]

        mock_selection_result = Mock()
        mock_selection_result.library_usage_result = mock_library_usage_result

        self.coordinator.step_results["select_tests"] = {
            "selection_result": mock_selection_result
        }

        # Mock LLM analysis to return empty mappings
        with patch.object(
            self.coordinator, "_analyze_api_mappings_with_llm"
        ) as mock_llm:
            mock_llm.return_value = {}

            result = await self.coordinator._find_api_mapping()

        assert result["success"] is False
        assert "Failed to find API mappings" in result["error"]
