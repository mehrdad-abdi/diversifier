"""Tests for DiversificationCoordinator API mapping functionality."""

import pytest
import tempfile
from unittest.mock import Mock, patch

from src.orchestration.coordinator import DiversificationCoordinator
from src.orchestration.config import LLMConfig, MigrationConfig
from src.orchestration.api_mapping import APIMappingResult, DocumentationSource


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
    async def test_find_api_mapping_no_test_results(self):
        """Test _find_api_mapping when no test selection results are available."""
        result = await self.coordinator._find_api_mapping()

        assert result["success"] is False
        assert "No test selection results available" in result["error"]

    @pytest.mark.asyncio
    async def test_find_api_mapping_no_library_usage_data(self):
        """Test _find_api_mapping when no library usage data is available."""
        # Add empty test selection results
        self.coordinator.step_results["select_tests"] = {"selection_result": None}

        result = await self.coordinator._find_api_mapping()

        assert result["success"] is False
        assert "No library usage data available" in result["error"]

    @pytest.mark.asyncio
    async def test_find_api_mapping_successful(self):
        """Test successful _find_api_mapping execution using APIMapper."""
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

        # Mock APIMapper
        source_docs = DocumentationSource(
            "requests", "https://docs.requests.com", "source docs"
        )
        target_docs = DocumentationSource(
            "httpx", "https://docs.httpx.com", "target docs"
        )

        mock_mapping_result = APIMappingResult(
            source_library="requests",
            target_library="httpx",
            used_apis=["get", "post", "session", "Session"],
            api_mappings={
                "get": "get",
                "post": "post",
                "session": "AsyncClient",
                "Session": "AsyncClient",
            },
            unmapped_apis=[],
            source_docs=source_docs,
            target_docs=target_docs,
            success=True,
        )

        with patch("src.orchestration.coordinator.APIMapper") as mock_api_mapper:
            mock_mapper_instance = Mock()

            # Mock the async method properly
            async def mock_find_api_mappings(used_apis):
                return mock_mapping_result

            mock_mapper_instance.find_api_mappings = mock_find_api_mappings
            mock_api_mapper.return_value = mock_mapper_instance

            result = await self.coordinator._find_api_mapping()

        assert result["success"] is True
        assert "get" in result["api_mappings"]
        assert "post" in result["api_mappings"]
        assert result["source_library"] == "requests"
        assert result["target_library"] == "httpx"
        assert len(result["used_apis"]) == 4
        assert result["source_docs_url"] == "https://docs.requests.com"
        assert result["target_docs_url"] == "https://docs.httpx.com"

    @pytest.mark.asyncio
    async def test_find_api_mapping_no_api_usage_found(self):
        """Test _find_api_mapping when no API usage is found."""
        # Mock library usage result with no matching APIs
        mock_module_usage = Mock()
        mock_module_usage.module_name = "os"  # Different library
        mock_module_usage.function_calls = ["listdir", "getcwd"]
        mock_module_usage.method_calls = []
        mock_module_usage.attribute_accesses = []

        mock_library_usage_result = Mock()
        mock_library_usage_result.module_usages = [mock_module_usage]

        mock_selection_result = Mock()
        mock_selection_result.library_usage_result = mock_library_usage_result

        # Add to step results
        self.coordinator.step_results["select_tests"] = {
            "selection_result": mock_selection_result
        }

        # Mock APIMapper for empty API set
        mock_mapping_result = APIMappingResult(
            source_library="requests",
            target_library="httpx",
            used_apis=[],
            api_mappings={},
            unmapped_apis=[],
            source_docs=None,
            target_docs=None,
            success=True,
        )

        with patch("src.orchestration.coordinator.APIMapper") as mock_api_mapper:
            mock_mapper_instance = Mock()

            # Mock the async method properly
            async def mock_find_api_mappings(used_apis):
                return mock_mapping_result

            mock_mapper_instance.find_api_mappings = mock_find_api_mappings
            mock_api_mapper.return_value = mock_mapper_instance

            result = await self.coordinator._find_api_mapping()

        assert result["success"] is True
        assert result["api_mappings"] == {}

    @pytest.mark.asyncio
    async def test_find_api_mapping_api_mapper_failure(self):
        """Test _find_api_mapping when APIMapper fails."""
        # Mock library usage result
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

        # Mock APIMapper to return failure
        mock_mapping_result = APIMappingResult(
            source_library="requests",
            target_library="httpx",
            used_apis=["get"],
            api_mappings={},
            unmapped_apis=["get"],
            source_docs=None,
            target_docs=None,
            success=False,
            error="Failed to find API mappings between requests and httpx",
        )

        with patch("src.orchestration.coordinator.APIMapper") as mock_api_mapper:
            mock_mapper_instance = Mock()

            # Mock the async method properly
            async def mock_find_api_mappings(used_apis):
                return mock_mapping_result

            mock_mapper_instance.find_api_mappings = mock_find_api_mappings
            mock_api_mapper.return_value = mock_mapper_instance

            result = await self.coordinator._find_api_mapping()

        assert result["success"] is False
        assert "Failed to find API mappings" in result["error"]

    @pytest.mark.asyncio
    async def test_find_api_mapping_exception_handling(self):
        """Test _find_api_mapping exception handling."""
        # Mock library usage result with proper structure but cause APIMapper to fail
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

        with patch("src.orchestration.coordinator.APIMapper") as mock_api_mapper:
            mock_api_mapper.side_effect = Exception("Test exception")

            result = await self.coordinator._find_api_mapping()

        assert result["success"] is False
        assert "API mapping failed: Test exception" in result["error"]
