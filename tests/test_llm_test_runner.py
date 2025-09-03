"""Tests for LLMTestRunner class."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.orchestration.test_running.llm_test_runner import (
    LLMTestRunner,
    UnrecoverableTestRunnerError,
)
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
                model="openai:gpt-4", temperature=0, **{}
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

        # Create a mock response matching actual JSON-RPC MCP response format
        mock_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": '{"pattern": "pyproject.toml", "search_directory": ".", "matches": [{"path": "pyproject.toml", "name": "pyproject.toml", "directory": "."}], "total_matches": 1}',
                    }
                ],
                "isError": False,
            },
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


class TestLLMTestRunnerRetryBehavior:
    """Test LLM test runner retry behavior for 5XX errors."""

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

    def test_is_5xx_error_detection(self, runner):
        """Test 5XX error detection helper method."""
        # Test actual 5XX error messages
        assert runner._is_5xx_error(
            "503 The model is overloaded. Please try again later."
        )
        assert runner._is_5xx_error("500 Internal Server Error")
        assert runner._is_5xx_error("502 Bad Gateway")
        assert runner._is_5xx_error("504 Gateway Timeout")

        # Test non-5XX errors should not be retried
        assert not runner._is_5xx_error("404 Not Found")
        assert not runner._is_5xx_error("401 Unauthorized")
        assert not runner._is_5xx_error("400 Bad Request")
        assert not runner._is_5xx_error("Connection timeout")
        assert not runner._is_5xx_error("Invalid API key")

        # Test edge cases
        assert not runner._is_5xx_error("Error 503 in the middle but not at start")
        assert runner._is_5xx_error("599 Custom Server Error")  # 5XX range end

    @pytest.mark.asyncio
    async def test_setup_and_run_tests_retries_on_503_error(self, runner):
        """Test that 503 errors during agent execution are retried."""
        # Mock project structure
        project_structure = {
            "project_path": "/test/project",
            "project_files": [{"name": "pyproject.toml", "type": "file"}],
            "test_directories": ["tests/"],
        }

        # Mock filesystem connection for reading project files
        filesystem_connection = runner.mcp_manager.get_connection(
            MCPServerType.FILESYSTEM
        )
        filesystem_connection.client.call_tool.return_value = {
            "result": {"content": [{"text": "mock_file_content"}]}
        }

        # Mock prompt file
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="test prompt {input}"),
        ):

            # Mock agent executor to fail twice with 503, then succeed
            mock_agent_executor = Mock()
            mock_agent_executor.ainvoke = AsyncMock(
                side_effect=[
                    Exception("503 The model is overloaded. Please try again later."),
                    Exception("503 Service Unavailable"),
                    {
                        "output": "Test execution completed successfully"
                    },  # Success on 3rd attempt
                ]
            )

            # Mock extraction method to return success
            runner._extract_structured_results = AsyncMock(
                return_value={
                    "testing_framework": "pytest",
                    "setup_successful": True,
                    "tests_executed": 2,
                    "tests_passed": 2,
                    "tests_failed": 0,
                }
            )

            with (
                patch(
                    "src.orchestration.test_running.llm_test_runner.create_react_agent"
                ),
                patch(
                    "src.orchestration.test_running.llm_test_runner.AgentExecutor",
                    return_value=mock_agent_executor,
                ),
            ):

                result = await runner.setup_and_run_tests(
                    project_structure, ["test_func"]
                )

                # Verify it succeeded after retries
                assert result["setup_successful"] is True
                assert result["tests_executed"] == 2

                # Verify agent was called 3 times (2 failures + 1 success)
                assert mock_agent_executor.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_setup_and_run_tests_retries_on_500_error(self, runner):
        """Test that 500 errors during agent execution are retried."""
        # Mock project structure
        project_structure = {
            "project_path": "/test/project",
            "project_files": [{"name": "pyproject.toml", "type": "file"}],
            "test_directories": ["tests/"],
        }

        # Mock filesystem connection
        filesystem_connection = runner.mcp_manager.get_connection(
            MCPServerType.FILESYSTEM
        )
        filesystem_connection.client.call_tool.return_value = {
            "result": {"content": [{"text": "mock_file_content"}]}
        }

        # Mock prompt file
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="test prompt {input}"),
        ):

            # Mock agent executor to fail with 500, then succeed
            mock_agent_executor = Mock()
            mock_agent_executor.ainvoke = AsyncMock(
                side_effect=[
                    Exception("500 Internal Server Error"),
                    {"output": "Test execution completed"},  # Success on 2nd attempt
                ]
            )

            # Mock extraction method
            runner._extract_structured_results = AsyncMock(
                return_value={
                    "testing_framework": "pytest",
                    "setup_successful": True,
                    "tests_executed": 1,
                    "tests_passed": 1,
                    "tests_failed": 0,
                }
            )

            with (
                patch(
                    "src.orchestration.test_running.llm_test_runner.create_react_agent"
                ),
                patch(
                    "src.orchestration.test_running.llm_test_runner.AgentExecutor",
                    return_value=mock_agent_executor,
                ),
            ):

                result = await runner.setup_and_run_tests(
                    project_structure, ["test_func"]
                )

                # Verify it succeeded after retry
                assert result["setup_successful"] is True
                assert mock_agent_executor.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_setup_and_run_tests_fails_after_max_retries(self, runner):
        """Test that after max retries on 5XX errors, it still fails."""
        # Mock project structure
        project_structure = {
            "project_path": "/test/project",
            "project_files": [{"name": "pyproject.toml", "type": "file"}],
            "test_directories": ["tests/"],
        }

        # Mock filesystem connection
        filesystem_connection = runner.mcp_manager.get_connection(
            MCPServerType.FILESYSTEM
        )
        filesystem_connection.client.call_tool.return_value = {
            "result": {"content": [{"text": "mock_file_content"}]}
        }

        # Mock prompt file
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="test prompt {input}"),
        ):

            # Mock agent executor to always fail with 503
            mock_agent_executor = Mock()
            mock_agent_executor.ainvoke = AsyncMock(
                side_effect=Exception("503 Service Unavailable")
            )

            with (
                patch(
                    "src.orchestration.test_running.llm_test_runner.create_react_agent"
                ),
                patch(
                    "src.orchestration.test_running.llm_test_runner.AgentExecutor",
                    return_value=mock_agent_executor,
                ),
            ):

                # Should still raise UnrecoverableTestRunnerError after retries
                with pytest.raises(UnrecoverableTestRunnerError) as exc_info:
                    await runner.setup_and_run_tests(project_structure, ["test_func"])

                # Verify error details
                assert "503 Service Unavailable" in str(exc_info.value.message)
                assert exc_info.value.error_type == "agent_execution"

                # Verify it tried 3 times (max retries)
                assert mock_agent_executor.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_setup_and_run_tests_no_retry_on_non_5xx_error(self, runner):
        """Test that non-5XX errors are not retried."""
        # Mock project structure
        project_structure = {
            "project_path": "/test/project",
            "project_files": [{"name": "pyproject.toml", "type": "file"}],
            "test_directories": ["tests/"],
        }

        # Mock filesystem connection
        filesystem_connection = runner.mcp_manager.get_connection(
            MCPServerType.FILESYSTEM
        )
        filesystem_connection.client.call_tool.return_value = {
            "result": {"content": [{"text": "mock_file_content"}]}
        }

        # Mock prompt file
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value="test prompt {input}"),
        ):

            # Mock agent executor to fail with 4XX error (should not retry)
            mock_agent_executor = Mock()
            mock_agent_executor.ainvoke = AsyncMock(
                side_effect=Exception("401 Unauthorized")
            )

            with (
                patch(
                    "src.orchestration.test_running.llm_test_runner.create_react_agent"
                ),
                patch(
                    "src.orchestration.test_running.llm_test_runner.AgentExecutor",
                    return_value=mock_agent_executor,
                ),
            ):

                # Should immediately fail without retries
                with pytest.raises(UnrecoverableTestRunnerError) as exc_info:
                    await runner.setup_and_run_tests(project_structure, ["test_func"])

                # Verify error details
                assert "401 Unauthorized" in str(exc_info.value.message)

                # Verify it was only called once (no retries)
                assert mock_agent_executor.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_structured_results_retries_on_5xx_error(self, runner):
        """Test that structured result extraction retries on 5XX errors."""
        # Mock the result object with model_dump method
        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "testing_framework": "pytest",
            "setup_successful": True,
            "tests_executed": 1,
        }

        # Mock the LLM to fail with 502, then succeed
        mock_extraction_llm = Mock()
        mock_extraction_llm.ainvoke = AsyncMock(
            side_effect=[
                Exception("502 Bad Gateway"),
                mock_result,  # Success with properly mocked result
            ]
        )

        runner.llm = Mock()
        runner.llm.with_structured_output.return_value = mock_extraction_llm

        # Test extraction with retry
        result = await runner._extract_structured_results(
            "mock agent output", ["test_func"]
        )

        # Verify it succeeded after retry
        assert result["setup_successful"] is True
        assert mock_extraction_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_structured_results_fails_after_max_retries(self, runner):
        """Test that structured result extraction fails after max retries."""
        # Mock the LLM to always fail with 504
        mock_extraction_llm = Mock()
        mock_extraction_llm.ainvoke = AsyncMock(
            side_effect=Exception("504 Gateway Timeout")
        )

        runner.llm = Mock()
        runner.llm.with_structured_output.return_value = mock_extraction_llm

        # Should raise exception after max retries
        with pytest.raises(Exception) as exc_info:
            await runner._extract_structured_results("mock agent output", ["test_func"])

        # Verify error message and retry count
        assert "504 Gateway Timeout" in str(exc_info.value)
        assert mock_extraction_llm.ainvoke.call_count == 3
