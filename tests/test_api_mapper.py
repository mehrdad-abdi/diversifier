"""Tests for APIMapper functionality."""

import pytest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

from src.orchestration.api_mapping import (
    APIMapper,
    APIMappingResult,
    DocumentationSource,
    APIMappingsResult,
    APIMigrationGuide,
)
from src.orchestration.config import LLMConfig
from src.orchestration.mcp_manager import MCPManager


class TestAPIMapper:
    """Test cases for APIMapper."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # Mock configurations
        self.llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key_env_var="OPENAI_API_KEY",
            temperature=0.2,
        )

        # Mock MCP manager
        self.mcp_manager = Mock(spec=MCPManager)

        # Create API mapper
        self.api_mapper = APIMapper(
            source_library="requests",
            target_library="httpx",
            llm_config=self.llm_config,
            mcp_manager=self.mcp_manager,
        )

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test APIMapper initialization."""
        assert self.api_mapper.source_library == "requests"
        assert self.api_mapper.target_library == "httpx"
        assert self.api_mapper.llm_config == self.llm_config
        assert self.api_mapper.mcp_manager == self.mcp_manager
        assert self.api_mapper.prompts_dir.name == "prompts"

    def test_init_with_custom_prompts_dir(self):
        """Test APIMapper initialization with custom prompts directory."""
        custom_prompts_dir = "/custom/prompts"
        mapper = APIMapper(
            source_library="requests",
            target_library="httpx",
            llm_config=self.llm_config,
            mcp_manager=self.mcp_manager,
            prompts_dir=custom_prompts_dir,
        )

        assert str(mapper.prompts_dir) == custom_prompts_dir

    @pytest.mark.asyncio
    async def test_find_api_mappings_web_server_not_available(self):
        """Test find_api_mappings when Web MCP server is not available."""
        self.mcp_manager.is_server_available.return_value = False
        used_apis = {"get", "post"}

        result = await self.api_mapper.find_api_mappings(used_apis)

        assert not result.success
        assert "Web MCP server not available" in result.error
        assert set(result.used_apis) == {"get", "post"}
        assert set(result.unmapped_apis) == {"get", "post"}

    @pytest.mark.asyncio
    async def test_find_api_mappings_empty_apis(self):
        """Test find_api_mappings with empty API set."""
        self.mcp_manager.is_server_available.return_value = True
        used_apis = set()

        result = await self.api_mapper.find_api_mappings(used_apis)

        assert result.success
        assert result.used_apis == []
        assert result.api_mappings == {}
        assert result.unmapped_apis == []

    @pytest.mark.asyncio
    async def test_find_api_mappings_successful(self):
        """Test successful API mapping."""
        self.mcp_manager.is_server_available.return_value = True
        used_apis = {"get", "post", "Session"}

        # Mock the fetch documentation method
        source_docs = DocumentationSource(
            "requests", "https://docs.requests.com", "source docs"
        )
        target_docs = DocumentationSource(
            "httpx", "https://docs.httpx.com", "target docs"
        )

        with patch.object(
            self.api_mapper, "_fetch_library_documentation"
        ) as mock_fetch:
            mock_fetch.side_effect = [source_docs, target_docs]

            with patch.object(
                self.api_mapper, "_analyze_api_mappings_with_llm"
            ) as mock_llm:
                mock_llm.return_value = {
                    "get": "get",
                    "post": "post",
                    "Session": "AsyncClient",
                }

                result = await self.api_mapper.find_api_mappings(used_apis)

        assert result.success
        assert result.api_mappings == {
            "get": "get",
            "post": "post",
            "Session": "AsyncClient",
        }
        assert len(result.used_apis) == 3
        assert result.unmapped_apis == []
        assert result.source_docs == source_docs
        assert result.target_docs == target_docs

    @pytest.mark.asyncio
    async def test_find_api_mappings_llm_failure(self):
        """Test API mapping when LLM analysis fails."""
        self.mcp_manager.is_server_available.return_value = True
        used_apis = {"get"}

        with patch.object(
            self.api_mapper, "_fetch_library_documentation"
        ) as mock_fetch:
            mock_fetch.return_value = None

            with patch.object(
                self.api_mapper, "_analyze_api_mappings_with_llm"
            ) as mock_llm:
                mock_llm.return_value = {}  # Empty mappings

                result = await self.api_mapper.find_api_mappings(used_apis)

        assert not result.success
        assert "Failed to find API mappings" in result.error
        assert result.unmapped_apis == ["get"]

    @pytest.mark.asyncio
    async def test_fetch_library_documentation_success(self):
        """Test successful documentation fetching."""
        # Mock web connection
        mock_web_connection = Mock()
        mock_web_connection.call_tool.side_effect = [
            {"info": {"docs_url": "https://docs.example.com"}},  # PyPI info
            {
                "result": [{"text": "Sample documentation content"}]
            },  # Documentation content
        ]
        self.mcp_manager.get_connection.return_value = mock_web_connection

        result = await self.api_mapper._fetch_library_documentation("requests")

        assert result is not None
        assert result.library_name == "requests"
        assert result.docs_url == "https://docs.example.com"
        assert result.content == "Sample documentation content"

    @pytest.mark.asyncio
    async def test_fetch_library_documentation_no_web_connection(self):
        """Test documentation fetching without web connection."""
        self.mcp_manager.get_connection.return_value = None

        result = await self.api_mapper._fetch_library_documentation("requests")

        assert result is None

    def test_extract_docs_url_with_docs_url(self):
        """Test extracting docs URL when docs_url field is present."""
        pypi_info = {"info": {"docs_url": "https://example.readthedocs.io/"}}

        result = self.api_mapper._extract_docs_url(pypi_info, "example")

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

        result = self.api_mapper._extract_docs_url(pypi_info, "example")

        assert result == "https://example-docs.com/"

    def test_extract_docs_url_fallback(self):
        """Test docs URL fallback behavior."""
        pypi_info = {"info": {}}

        result = self.api_mapper._extract_docs_url(pypi_info, "example")

        assert result == "https://example.readthedocs.io/"

    def test_extract_docs_url_no_pypi_info(self):
        """Test extracting docs URL when pypi_info is None."""
        result = self.api_mapper._extract_docs_url(None, "example")

        assert result == ""

    @pytest.mark.asyncio
    @patch("src.orchestration.api_mapping.api_mapper.init_chat_model")
    async def test_analyze_api_mappings_with_llm(self, mock_init_chat_model):
        """Test LLM API mapping analysis."""
        # Create temporary prompt files
        prompts_dir = Path(self.temp_dir.name) / "prompts"
        prompts_dir.mkdir()

        system_prompt = "Test system prompt: {source_library} {target_library} {source_docs} {target_docs}"
        human_prompt = (
            "Test human prompt: {source_library} {target_library} {apis_list}"
        )

        (prompts_dir / "api_mapping_system.txt").write_text(system_prompt)
        (prompts_dir / "api_mapping_human.txt").write_text(human_prompt)

        # Update mapper to use temp prompts directory
        self.api_mapper.prompts_dir = prompts_dir

        # Mock LLM model with structured output
        mock_model = Mock()
        mock_structured_model = Mock()
        mock_structured_response = APIMappingsResult(
            migration_guides=[
                APIMigrationGuide(
                    source_api="get",
                    target_api="get",
                    confidence="HIGH",
                    source_example="requests.get('https://api.example.com')",
                    target_example="httpx.get('https://api.example.com')",
                ),
                APIMigrationGuide(
                    source_api="post",
                    target_api="post",
                    confidence="HIGH",
                    source_example="requests.post('https://api.example.com', data={'key': 'value'})",
                    target_example="httpx.post('https://api.example.com', data={'key': 'value'})",
                ),
                APIMigrationGuide(
                    source_api="Session",
                    target_api="AsyncClient",
                    confidence="MEDIUM",
                    source_example="session = requests.Session()",
                    target_example="client = httpx.AsyncClient()",
                    migration_notes="Note: AsyncClient is async and requires await",
                ),
            ],
            general_migration_notes="Migration from requests to httpx requires async/await patterns",
        )
        mock_structured_model.invoke = Mock(return_value=mock_structured_response)
        mock_model.with_structured_output = Mock(return_value=mock_structured_model)
        mock_init_chat_model.return_value = mock_model

        used_apis = {"get", "post", "Session"}
        source_docs = DocumentationSource("requests", "url", "source content")
        target_docs = DocumentationSource("httpx", "url", "target content")

        result = await self.api_mapper._analyze_api_mappings_with_llm(
            used_apis, source_docs, target_docs
        )

        assert result == {"get": "get", "post": "post", "Session": "AsyncClient"}
        mock_init_chat_model.assert_called_once()
        mock_model.with_structured_output.assert_called_once_with(APIMappingsResult)
        mock_structured_model.invoke.assert_called_once()

    def test_load_prompt_template_existing_file(self):
        """Test loading prompt template from existing file."""
        # Create temporary prompt file
        prompts_dir = Path(self.temp_dir.name) / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "test_prompt.txt"
        prompt_file.write_text("Test prompt content")

        # Update mapper to use temp prompts directory
        self.api_mapper.prompts_dir = prompts_dir

        result = self.api_mapper._load_prompt_template("test_prompt.txt")

        assert result == "Test prompt content"

    def test_load_prompt_template_missing_file(self):
        """Test loading prompt template from missing file."""
        result = self.api_mapper._load_prompt_template("nonexistent.txt")

        assert result == ""


class TestDocumentationSource:
    """Test cases for DocumentationSource dataclass."""

    def test_documentation_source_creation(self):
        """Test DocumentationSource creation."""
        doc_source = DocumentationSource(
            library_name="requests",
            docs_url="https://docs.requests.com",
            content="Documentation content",
        )

        assert doc_source.library_name == "requests"
        assert doc_source.docs_url == "https://docs.requests.com"
        assert doc_source.content == "Documentation content"


class TestAPIMappingResult:
    """Test cases for APIMappingResult dataclass."""

    def test_api_mapping_result_creation(self):
        """Test APIMappingResult creation."""
        result = APIMappingResult(
            source_library="requests",
            target_library="httpx",
            used_apis=["get", "post"],
            api_mappings={"get": "get", "post": "post"},
            unmapped_apis=[],
            source_docs=None,
            target_docs=None,
            success=True,
        )

        assert result.source_library == "requests"
        assert result.target_library == "httpx"
        assert result.used_apis == ["get", "post"]
        assert result.api_mappings == {"get": "get", "post": "post"}
        assert result.unmapped_apis == []
        assert result.success is True
        assert result.error is None
