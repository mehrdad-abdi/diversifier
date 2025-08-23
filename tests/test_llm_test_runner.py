"""Tests for LLMTestRunner class."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.orchestration.test_running.llm_test_runner import LLMTestRunner
from src.orchestration.config import LLMConfig, MigrationConfig
from src.orchestration.mcp_manager import MCPManager, MCPServerType


@pytest.fixture
def llm_config():
    """Create a test LLM configuration."""
    return LLMConfig(
        provider="openai",
        model_name="gpt-4",
        api_key_env_var="OPENAI_API_KEY",
        max_tokens=1000,
        additional_params={},
    )


@pytest.fixture
def migration_config():
    """Create a test migration configuration."""
    return MigrationConfig(
        common_project_files=["pyproject.toml", "requirements.txt"],
        test_paths=["tests/", "test/"],
    )


@pytest.fixture
def mock_mcp_manager():
    """Create a mock MCP manager."""
    manager = Mock(spec=MCPManager)

    # Mock connection objects
    mock_command_connection = Mock()
    mock_command_connection.client = Mock()
    mock_filesystem_connection = Mock()
    mock_filesystem_connection.client = Mock()

    manager.get_connection.side_effect = lambda server_type: {
        MCPServerType.COMMAND: mock_command_connection,
        MCPServerType.FILESYSTEM: mock_filesystem_connection,
    }.get(server_type)

    return manager


class TestLLMTestRunnerInitialization:
    """Test LLMTestRunner initialization."""

    def test_init_with_valid_parameters(
        self, llm_config, migration_config, mock_mcp_manager
    ):
        """Test successful initialization with all required parameters."""
        project_path = "/test/project"

        with patch(
            "src.orchestration.test_running.llm_test_runner.init_chat_model"
        ) as mock_init_chat:
            mock_llm = Mock()
            mock_init_chat.return_value = mock_llm

            runner = LLMTestRunner(
                project_path=project_path,
                llm_config=llm_config,
                migration_config=migration_config,
                mcp_manager=mock_mcp_manager,
            )

            assert runner.project_path == Path(project_path).resolve()
            assert runner.llm_config == llm_config
            assert runner.migration_config == migration_config
            assert runner.mcp_manager == mock_mcp_manager
            assert runner.llm == mock_llm

            # Verify LLM was initialized correctly
            mock_init_chat.assert_called_once_with(
                model="openai:gpt-4", temperature=0, max_tokens=1000, **{}
            )

    def test_init_with_relative_path_resolves_to_absolute(
        self, llm_config, migration_config, mock_mcp_manager
    ):
        """Test that relative paths are resolved to absolute paths."""
        with patch("src.orchestration.test_running.llm_test_runner.init_chat_model"):
            runner = LLMTestRunner(
                project_path="./relative/path",
                llm_config=llm_config,
                migration_config=migration_config,
                mcp_manager=mock_mcp_manager,
            )

            assert runner.project_path.is_absolute()


class TestLLMTestRunnerMethods:
    """Test LLMTestRunner methods."""

    @pytest.fixture
    def runner(self, llm_config, migration_config, mock_mcp_manager):
        """Create a test runner instance."""
        with patch("src.orchestration.test_running.llm_test_runner.init_chat_model"):
            return LLMTestRunner(
                project_path="/test/project",
                llm_config=llm_config,
                migration_config=migration_config,
                mcp_manager=mock_mcp_manager,
            )

    def test_analyze_project_structure_gets_clients_from_manager(self, runner):
        """Test that analyze_project_structure gets clients from MCP manager."""
        # Mock the call_tool method on the command client for find_files
        command_connection = runner.mcp_manager.get_connection(MCPServerType.COMMAND)

        # Create a mock response matching actual MCP response format
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": '{"pattern": "pyproject.toml", "search_directory": ".", "matches": [{"path": "pyproject.toml", "name": "pyproject.toml", "directory": "."}], "total_matches": 1}',
                }
            ]
        }
        command_connection.client.call_tool.return_value = mock_response

        # Test that the method gets clients from manager
        # Note: This is now a sync method since we removed the async pattern
        runner.analyze_project_structure()

        # Verify connection was obtained from manager
        runner.mcp_manager.get_connection.assert_called_with(MCPServerType.COMMAND)

    def test_no_initialize_mcp_clients_method(self, runner):
        """Test that initialize_mcp_clients method no longer exists."""
        assert not hasattr(runner, "initialize_mcp_clients")

    def test_no_optional_client_attributes(self, runner):
        """Test that MCP clients are not stored as optional attributes."""
        # The runner should not have command_client or filesystem_client attributes
        assert not hasattr(runner, "command_client")
        assert not hasattr(runner, "filesystem_client")
