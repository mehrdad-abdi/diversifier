"""Tests for the simple sequential workflow."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.orchestration.simple_workflow import MigrationWorkflow, RateLimitError
from src.orchestration.config import LLMConfig, MigrationConfig
from src.orchestration.mcp_manager import MCPServerType
from src.orchestration.test_running.llm_test_runner import UnrecoverableTestRunnerError


@pytest.fixture
def llm_config():
    """Mock LLM configuration for testing."""
    return LLMConfig(
        provider="openai",
        model_name="gpt-4",
        api_key_env_var="OPENAI_API_KEY",
        temperature=0.1,
        max_tokens=1000,
        additional_params={},
    )


@pytest.fixture
def migration_config():
    """Mock migration configuration for testing."""
    return MigrationConfig(test_paths=["tests/"])


@pytest.fixture
def workflow(llm_config, migration_config, tmp_path):
    """Create a workflow instance for testing."""
    return MigrationWorkflow(
        project_path=str(tmp_path),
        source_library="requests",
        target_library="httpx",
        llm_config=llm_config,
        migration_config=migration_config,
    )


class TestMigrationWorkflow:
    """Test cases for MigrationWorkflow."""

    def test_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.source_library == "requests"
        assert workflow.target_library == "httpx"
        assert isinstance(workflow.project_path, Path)
        assert workflow.step_results == {}

    @pytest.mark.asyncio
    async def test_execute_success(self, workflow):
        """Test successful workflow execution."""
        # Mock all step methods to return success
        with (
            patch.object(
                workflow, "_initialize_environment", new_callable=AsyncMock
            ) as mock_init,
            patch.object(
                workflow, "_create_backup", new_callable=AsyncMock
            ) as mock_backup,
            patch.object(
                workflow, "_select_tests", new_callable=AsyncMock
            ) as mock_select,
            patch.object(
                workflow, "_run_baseline_tests", new_callable=AsyncMock
            ) as mock_baseline,
            patch.object(
                workflow, "_migrate_code", new_callable=AsyncMock
            ) as mock_migrate,
            patch.object(
                workflow, "_validate_migration", new_callable=AsyncMock
            ) as mock_validate,
            patch.object(
                workflow, "_repair_issues", new_callable=AsyncMock
            ) as mock_repair,
            patch.object(
                workflow, "_finalize_migration", new_callable=AsyncMock
            ) as mock_finalize,
            patch.object(workflow, "_cleanup", new_callable=AsyncMock) as mock_cleanup,
        ):

            # Configure all steps to return success
            for mock_step in [
                mock_init,
                mock_backup,
                mock_select,
                mock_baseline,
                mock_migrate,
                mock_validate,
                mock_repair,
                mock_finalize,
            ]:
                mock_step.return_value = {"success": True}

            result = await workflow.execute()

            assert result is True
            assert len(workflow.step_results) == 8
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_failure(self, workflow):
        """Test workflow execution with step failure."""
        with (
            patch.object(
                workflow, "_initialize_environment", new_callable=AsyncMock
            ) as mock_init,
            patch.object(
                workflow, "_create_backup", new_callable=AsyncMock
            ) as mock_backup,
            patch.object(workflow, "_cleanup", new_callable=AsyncMock) as mock_cleanup,
        ):

            mock_init.return_value = {"success": True}
            mock_backup.return_value = {"success": False, "error": "Backup failed"}

            result = await workflow.execute()

            assert result is False
            assert len(workflow.step_results) == 1  # Only init step completed
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_step_with_retry_rate_limit(self, workflow):
        """Test step execution with rate limit retry."""
        mock_step = AsyncMock()
        mock_step.side_effect = [
            RateLimitError("Rate limit exceeded"),
            RateLimitError("Rate limit exceeded"),
            {"success": True},
        ]

        with patch.object(asyncio, "sleep", new_callable=AsyncMock) as mock_sleep:
            result = await workflow._execute_step_with_retry(mock_step)

            assert result == {"success": True}
            assert mock_step.call_count == 3
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_step_with_retry_other_error(self, workflow):
        """Test step execution with non-rate-limit error (no retry)."""
        mock_step = AsyncMock()
        mock_step.side_effect = ValueError("Some other error")

        with pytest.raises(ValueError):
            await workflow._execute_step_with_retry(mock_step)

        assert mock_step.call_count == 1

    @pytest.mark.asyncio
    async def test_initialize_environment_success(self, workflow):
        """Test successful environment initialization."""
        with (
            patch.object(workflow, "_validate_llm_config", return_value=True),
            patch.object(
                workflow.mcp_manager, "initialize_all_servers", new_callable=AsyncMock
            ) as mock_init,
        ):

            mock_init.return_value = {
                MCPServerType.FILESYSTEM: True,
                MCPServerType.GIT: True,
            }

            result = await workflow._initialize_environment()

            assert result["success"] is True
            assert "mcp_servers" in result
            assert "available_servers" in result

    @pytest.mark.asyncio
    async def test_initialize_environment_invalid_config(self, workflow):
        """Test environment initialization with invalid LLM config."""
        with patch.object(workflow, "_validate_llm_config", return_value=False):
            result = await workflow._initialize_environment()

            assert result["success"] is False
            assert "Invalid LLM configuration" in result["error"]

    @pytest.mark.asyncio
    async def test_initialize_environment_filesystem_failure(self, workflow):
        """Test environment initialization with filesystem server failure."""
        with (
            patch.object(workflow, "_validate_llm_config", return_value=True),
            patch.object(
                workflow.mcp_manager, "initialize_all_servers", new_callable=AsyncMock
            ) as mock_init,
        ):

            mock_init.return_value = {
                MCPServerType.FILESYSTEM: False,
                MCPServerType.GIT: True,
            }

            result = await workflow._initialize_environment()

            assert result["success"] is False
            assert "Filesystem MCP server failed to initialize" in result["error"]

    def test_validate_llm_config_success(self, workflow):
        """Test successful LLM configuration validation."""
        with patch("src.orchestration.simple_workflow.init_chat_model") as mock_init:
            mock_init.return_value = Mock()

            result = workflow._validate_llm_config()

            assert result is True
            mock_init.assert_called_once_with(
                model="openai:gpt-4", temperature=0.1, max_tokens=1000
            )

    def test_validate_llm_config_failure(self, workflow):
        """Test LLM configuration validation failure."""
        with patch("src.orchestration.simple_workflow.init_chat_model") as mock_init:
            mock_init.side_effect = ValueError("Invalid model")

            result = workflow._validate_llm_config()

            assert result is False

    @pytest.mark.asyncio
    async def test_create_backup_success(self, workflow):
        """Test successful backup creation."""
        with (
            patch.object(
                workflow.mcp_manager, "is_server_available", return_value=True
            ),
            patch.object(
                workflow.mcp_manager, "call_tool", new_callable=AsyncMock
            ) as mock_call,
        ):

            mock_call.side_effect = [
                {"branch": "main", "status": "clean"},  # get_status
                {
                    "status": "success",
                    "temp_branch": "diversifier-backup-123",
                    "base_branch": "main",
                },  # create_temp_branch
            ]

            result = await workflow._create_backup()

            assert result["success"] is True
            assert result["backup_method"] == "git_branch"
            assert "diversifier-backup-123" in result["backup_path"]

    @pytest.mark.asyncio
    async def test_create_backup_git_unavailable(self, workflow):
        """Test backup creation when git server is unavailable."""
        with patch.object(
            workflow.mcp_manager, "is_server_available", return_value=False
        ):
            result = await workflow._create_backup()

            assert result["success"] is False
            assert "Git MCP server not available" in result["error"]

    @pytest.mark.asyncio
    async def test_select_tests_success(self, workflow):
        """Test successful test selection."""
        mock_selection_result = Mock()
        mock_selection_result.pipeline_success = True
        mock_selection_result.total_execution_time = 1.5

        mock_summary = {
            "test_coverage": {
                "covered_usages": 5,
                "uncovered_usages": 2,
                "coverage_percentage": 71.4,
            }
        }

        with (
            patch.object(
                workflow.test_coverage_selector,
                "select_test_coverage",
                new_callable=AsyncMock,
            ) as mock_select,
            patch.object(
                workflow.test_coverage_selector,
                "get_selection_summary",
                return_value=mock_summary,
            ),
        ):

            mock_select.return_value = mock_selection_result

            result = await workflow._select_tests()

            assert result["success"] is True
            assert result["selection_result"] == mock_selection_result
            assert result["summary"] == mock_summary

    @pytest.mark.asyncio
    async def test_select_tests_failure(self, workflow):
        """Test test selection failure."""
        mock_selection_result = Mock()
        mock_selection_result.pipeline_success = False

        with patch.object(
            workflow.test_coverage_selector,
            "select_test_coverage",
            new_callable=AsyncMock,
        ) as mock_select:
            mock_select.return_value = mock_selection_result

            result = await workflow._select_tests()

            assert result["success"] is False
            assert "Test coverage selection pipeline failed" in result["error"]

    @pytest.mark.asyncio
    async def test_run_baseline_tests_no_coverage(self, workflow):
        """Test baseline test execution with no test coverage."""
        # Set up step results
        workflow.step_results["select_tests"] = {
            "selection_result": Mock(test_discovery_result=Mock(coverage_paths=[]))
        }

        result = await workflow._run_baseline_tests()

        assert result["success"] is True
        assert result["test_results"]["tests_executed"] == 0
        assert (
            "No tests cover the selected library usage"
            in result["test_results"]["note"]
        )

    @pytest.mark.asyncio
    async def test_run_tests_with_llm_success(self, workflow):
        """Test LLM-powered test execution success."""
        test_functions = {
            "test_file.py::test_function1",
            "test_file.py::test_function2",
        }

        mock_runner = Mock()
        mock_runner.analyze_project_structure = AsyncMock(
            return_value={"test_files": ["test_file.py"]}
        )
        mock_runner.detect_dev_requirements = AsyncMock(
            return_value={"testing_framework": "pytest"}
        )
        mock_runner.setup_test_environment = AsyncMock(return_value={"success": True})
        mock_runner.run_tests = AsyncMock(
            return_value={
                "overall_success": True,
                "summary": {"successful_commands": 1, "total_commands": 1},
                "test_commands_executed": [{"stdout": "2 passed", "stderr": ""}],
            }
        )

        with patch(
            "src.orchestration.simple_workflow.LLMTestRunner", return_value=mock_runner
        ):
            result = await workflow._run_tests_with_llm(test_functions)

            assert result["success"] is True
            assert result["test_results"]["passed"] == 2
            assert result["test_results"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_run_tests_with_llm_unrecoverable_error(self, workflow):
        """Test LLM test execution with unrecoverable error."""
        test_functions = {"test_file.py::test_function"}

        mock_runner = Mock()
        mock_runner.analyze_project_structure = AsyncMock(
            side_effect=UnrecoverableTestRunnerError("Fatal error", "FATAL", None)
        )

        with (
            patch(
                "src.orchestration.simple_workflow.LLMTestRunner",
                return_value=mock_runner,
            ),
            pytest.raises(RuntimeError, match="Test runner encountered unrecoverable"),
        ):
            await workflow._run_tests_with_llm(test_functions)

    @pytest.mark.asyncio
    async def test_migrate_code_success(self, workflow):
        """Test successful code migration."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Migration completed successfully"}

        with patch.object(workflow.agent_manager, "get_agent", return_value=mock_agent):
            result = await workflow._migrate_code()

            assert result["success"] is True
            assert "Migration completed successfully" in result["migration_result"]

    @pytest.mark.asyncio
    async def test_validate_migration_success(self, workflow):
        """Test successful migration validation."""
        workflow.step_results["select_tests"] = {"selection_result": Mock()}

        result = await workflow._validate_migration()

        assert result["success"] is True
        assert "Test coverage selection only" in result["test_results"]["note"]

    @pytest.mark.asyncio
    async def test_repair_issues_no_repair_needed(self, workflow):
        """Test repair step when no issues found."""
        workflow.step_results["validate_migration"] = {"test_results": {"failed": 0}}

        result = await workflow._repair_issues()

        assert result["success"] is True
        assert result["repairs_needed"] is False

    @pytest.mark.asyncio
    async def test_repair_issues_with_failures(self, workflow):
        """Test repair step when issues are found."""
        workflow.step_results["validate_migration"] = {"test_results": {"failed": 2}}

        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Repairs applied successfully"}

        with patch.object(workflow.agent_manager, "get_agent", return_value=mock_agent):
            result = await workflow._repair_issues()

            assert result["success"] is True
            assert "Repairs applied successfully" in result["repair_result"]

    @pytest.mark.asyncio
    async def test_finalize_migration_success(self, workflow):
        """Test successful migration finalization."""
        workflow.step_results["migrate_code"] = {
            "files_modified": ["file1.py", "file2.py"]
        }

        with (
            patch.object(
                workflow.mcp_manager, "is_server_available", return_value=True
            ),
            patch.object(
                workflow.mcp_manager, "call_tool", new_callable=AsyncMock
            ) as mock_call,
        ):

            mock_call.side_effect = [
                {"status": "success"},  # add_files
                {"status": "success", "commit_hash": "abc123"},  # commit_changes
            ]

            result = await workflow._finalize_migration()

            assert result["success"] is True
            assert result["migration_finalized"] is True
            assert result["commit_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_finalize_migration_no_git(self, workflow):
        """Test migration finalization without git server."""
        with patch.object(
            workflow.mcp_manager, "is_server_available", return_value=False
        ):
            result = await workflow._finalize_migration()

            assert result["success"] is True
            assert result["migration_finalized"] is True

    @pytest.mark.asyncio
    async def test_cleanup(self, workflow):
        """Test workflow cleanup."""
        with (
            patch.object(
                workflow.mcp_manager, "shutdown_all_servers", new_callable=AsyncMock
            ) as mock_shutdown,
            patch.object(workflow.agent_manager, "clear_all_memories") as mock_clear,
        ):

            await workflow._cleanup()

            mock_shutdown.assert_called_once()
            mock_clear.assert_called_once()

    def test_get_workflow_summary(self, workflow):
        """Test workflow summary generation."""
        workflow.step_results = {
            "step1": {"success": True},
            "step2": {"success": False, "error": "Failed"},
        }

        summary = workflow.get_workflow_summary()

        assert summary["source_library"] == "requests"
        assert summary["target_library"] == "httpx"
        assert summary["total_steps"] == 8
        assert summary["completed_steps"] == 1
        assert summary["failed_steps"] == 1
        assert summary["is_complete"] is False


class TestRateLimitError:
    """Test cases for RateLimitError."""

    def test_rate_limit_error_creation(self):
        """Test creating RateLimitError."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, Exception)
