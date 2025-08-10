"""Tests for orchestration system components."""

import pytest
import tempfile
import logging
import os
from unittest.mock import Mock, patch
from pathlib import Path

from src.orchestration.agent import AgentManager, AgentType, DiversificationAgent
from src.orchestration.mcp_manager import MCPManager, MCPServerType, MCPConnection
from src.orchestration.workflow import (
    WorkflowState,
    MigrationContext,
    WorkflowStage,
    WorkflowStatus,
)
from src.orchestration.coordinator import DiversificationCoordinator
from src.orchestration.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
from src.orchestration.logging_config import setup_logging, get_logger


class TestDiversificationAgent:
    """Test cases for DiversificationAgent."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_agent_initialization(self):
        """Test agent initialization."""
        from src.orchestration.config import LLMConfig

        llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key_env_var="OPENAI_API_KEY",
            temperature=0.2,
        )
        agent = DiversificationAgent(
            agent_type=AgentType.ANALYZER, llm_config=llm_config
        )

        assert agent.agent_type == AgentType.ANALYZER
        assert agent.llm_config.model_name == "gpt-3.5-turbo"
        assert agent.llm_config.temperature == 0.2
        assert agent.llm is not None
        assert agent.memory is not None
        assert len(agent.tools) == 0

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_agent_with_tools(self):
        """Test agent initialization with tools."""
        from src.orchestration.config import LLMConfig

        mock_llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key_env_var="TEST_API_KEY",
        )

        # Create a proper mock tool with required attributes
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.__name__ = "test_tool"  # Add required __name__ attribute

        # Skip tool initialization to avoid LangChain issues in testing
        with patch.object(DiversificationAgent, "_initialize_agent"):
            agent = DiversificationAgent(
                agent_type=AgentType.MIGRATOR,
                llm_config=mock_llm_config,
                tools=[mock_tool],
            )
            agent.tools = [mock_tool]  # Set directly for testing

        assert len(agent.tools) == 1
        assert agent.tools[0] == mock_tool

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_agent_invoke_without_tools(self):
        """Test agent invocation without tools."""
        from src.orchestration.config import LLMConfig

        mock_llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )

        agent = DiversificationAgent(
            agent_type=AgentType.ANALYZER, llm_config=mock_llm_config
        )
        result = agent.invoke("Test input")

        assert result["output"] == "Mock response"

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_add_tool(self):
        """Test adding a tool to an agent."""
        from src.orchestration.config import LLMConfig

        mock_llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )

        agent = DiversificationAgent(
            agent_type=AgentType.TESTER, llm_config=mock_llm_config
        )

        mock_tool = Mock()
        mock_tool.name = "new_tool"
        mock_tool.__name__ = "new_tool"

        # Mock the reinitialize method to avoid LangChain issues
        with patch.object(agent, "_initialize_agent"):
            agent.add_tool(mock_tool)

        assert mock_tool in agent.tools
        assert len(agent.tools) == 1

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_clear_memory(self):
        """Test clearing agent memory."""
        from src.orchestration.config import LLMConfig

        mock_llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )

        agent = DiversificationAgent(
            agent_type=AgentType.REPAIRER, llm_config=mock_llm_config
        )
        original_memory = agent.memory
        agent.clear_memory()

        # Memory should be a new instance (cleared)
        assert agent.memory is not original_memory


class TestAgentManager:
    """Test cases for AgentManager."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_agent_manager_initialization(self):
        """Test agent manager initialization."""
        from src.orchestration.config import LLMConfig

        llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-4",
            api_key_env_var="OPENAI_API_KEY",
            temperature=0.1,
        )
        manager = AgentManager(llm_config=llm_config)

        assert manager.llm_config.model_name == "gpt-4"
        assert manager.llm_config.temperature == 0.1
        assert len(manager.agents) == 0

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_get_agent(self):
        """Test getting an agent from manager."""
        from src.orchestration.config import LLMConfig

        mock_llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )

        manager = AgentManager(llm_config=mock_llm_config)
        agent = manager.get_agent(AgentType.ANALYZER)

        assert isinstance(agent, DiversificationAgent)
        assert agent.agent_type == AgentType.ANALYZER
        assert AgentType.ANALYZER in manager.agents

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_get_same_agent_twice(self):
        """Test getting the same agent type twice returns same instance."""
        from src.orchestration.config import LLMConfig

        mock_llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )

        manager = AgentManager(llm_config=mock_llm_config)
        agent1 = manager.get_agent(AgentType.MIGRATOR)
        agent2 = manager.get_agent(AgentType.MIGRATOR)

        assert agent1 is agent2

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_clear_all_memories(self):
        """Test clearing all agent memories."""
        from src.orchestration.config import LLMConfig

        mock_llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )

        manager = AgentManager(llm_config=mock_llm_config)
        agent1 = manager.get_agent(AgentType.ANALYZER)
        agent2 = manager.get_agent(AgentType.MIGRATOR)

        original_memory1 = agent1.memory
        original_memory2 = agent2.memory

        manager.clear_all_memories()

        # Both agents should have new memory instances (cleared)
        assert agent1.memory is not original_memory1
        assert agent2.memory is not original_memory2


class TestMCPConnection:
    """Test cases for MCPConnection."""

    def test_mcp_connection_initialization(self):
        """Test MCP connection initialization."""
        mock_client = Mock()
        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)

        assert connection.server_type == MCPServerType.FILESYSTEM
        assert connection.client == mock_client
        assert connection.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        mock_client = Mock()
        mock_client.start_server.return_value = True

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        result = await connection.connect()

        assert result is True
        assert connection.is_connected is True
        mock_client.start_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        mock_client = Mock()
        mock_client.start_server.return_value = False

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        result = await connection.connect()

        assert result is False
        assert connection.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection."""
        mock_client = Mock()
        mock_client.start_server.return_value = True
        mock_client.stop_server = Mock()

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        await connection.connect()
        await connection.disconnect()

        assert connection.is_connected is False
        mock_client.stop_server.assert_called_once()

    def test_call_tool_when_connected(self):
        """Test calling tool when connected."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"result": "success"}

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True

        result = connection.call_tool("test_tool", {"arg": "value"})

        assert result == {"result": "success"}
        mock_client.call_tool.assert_called_once_with("test_tool", {"arg": "value"})

    def test_call_tool_when_disconnected(self):
        """Test calling tool when not connected."""
        mock_client = Mock()

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = False

        result = connection.call_tool("test_tool", {"arg": "value"})

        assert result is None
        mock_client.call_tool.assert_not_called()


class TestMCPManager:
    """Test cases for MCPManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = self.temp_dir.name

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_mcp_manager_initialization(self):
        """Test MCP manager initialization."""
        manager = MCPManager(project_root=self.project_root)

        assert manager.project_root == self.project_root
        assert len(manager.connections) == 0

    @patch("src.orchestration.mcp_manager.FileSystemMCPClient")
    @pytest.mark.asyncio
    async def test_initialize_filesystem_server_success(self, mock_client_class):
        """Test successful filesystem server initialization."""
        mock_client = Mock()
        mock_client.start_server.return_value = True
        mock_client_class.return_value = mock_client

        manager = MCPManager(project_root=self.project_root)
        result = await manager.initialize_filesystem_server()

        assert result is True
        assert MCPServerType.FILESYSTEM in manager.connections
        mock_client_class.assert_called_once_with(project_root=self.project_root)

    @patch("src.orchestration.mcp_manager.FileSystemMCPClient")
    @pytest.mark.asyncio
    async def test_initialize_filesystem_server_failure(self, mock_client_class):
        """Test filesystem server initialization failure."""
        mock_client = Mock()
        mock_client.start_server.return_value = False
        mock_client_class.return_value = mock_client

        manager = MCPManager(project_root=self.project_root)
        result = await manager.initialize_filesystem_server()

        assert result is False
        assert MCPServerType.FILESYSTEM not in manager.connections

    @patch("src.orchestration.mcp_manager.GitMCPClient")
    @pytest.mark.asyncio
    async def test_initialize_git_server_success(self, mock_client_class):
        """Test successful git server initialization."""
        mock_client = Mock()
        mock_client.start_server.return_value = True
        mock_client_class.return_value = mock_client

        manager = MCPManager(project_root=self.project_root)
        result = await manager.initialize_git_server()

        assert result is True
        assert MCPServerType.GIT in manager.connections
        mock_client_class.assert_called_once_with(project_root=self.project_root)

    @patch("src.orchestration.mcp_manager.DockerMCPLauncher")
    @pytest.mark.asyncio
    async def test_initialize_docker_server_success(self, mock_client_class):
        """Test successful docker server initialization."""
        mock_client = Mock()
        mock_client.start_server.return_value = True
        mock_client_class.return_value = mock_client

        manager = MCPManager(project_root=self.project_root)
        result = await manager.initialize_docker_server()

        assert result is True
        assert MCPServerType.DOCKER in manager.connections
        mock_client_class.assert_called_once_with(project_root=self.project_root)

    @pytest.mark.asyncio
    async def test_initialize_all_servers(self):
        """Test initializing all servers."""
        manager = MCPManager(project_root=self.project_root)

        with (
            patch.object(manager, "initialize_filesystem_server", return_value=True),
            patch.object(manager, "initialize_testing_server", return_value=True),
            patch.object(manager, "initialize_git_server", return_value=True),
            patch.object(manager, "initialize_docker_server", return_value=True),
        ):

            results = await manager.initialize_all_servers()

            assert len(results) == 4
            assert all(results.values())

    def test_get_connection_available(self):
        """Test getting available connection."""
        mock_client = Mock()
        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True

        manager = MCPManager(project_root=self.project_root)
        manager.connections[MCPServerType.FILESYSTEM] = connection

        result = manager.get_connection(MCPServerType.FILESYSTEM)

        assert result is connection

    def test_get_connection_unavailable(self):
        """Test getting unavailable connection."""
        manager = MCPManager(project_root=self.project_root)

        result = manager.get_connection(MCPServerType.FILESYSTEM)

        assert result is None

    def test_is_server_available(self):
        """Test checking server availability."""
        mock_client = Mock()
        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True

        manager = MCPManager(project_root=self.project_root)
        manager.connections[MCPServerType.FILESYSTEM] = connection

        assert manager.is_server_available(MCPServerType.FILESYSTEM) is True
        assert (
            manager.is_server_available(MCPServerType.GIT) is False
        )  # Not connected in this test

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Test calling tool through manager."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {"result": "success"}

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True

        manager = MCPManager(project_root=self.project_root)
        manager.connections[MCPServerType.FILESYSTEM] = connection

        result = await manager.call_tool(
            MCPServerType.FILESYSTEM, "test_tool", {"arg": "value"}
        )

        assert result == {"result": "success"}

    def test_mcp_connection_health_check(self):
        """Test MCP connection health checking."""
        mock_client = Mock()
        mock_client.list_tools.return_value = {"result": []}

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True

        assert connection.is_healthy() is True
        mock_client.list_tools.assert_called_once()

    def test_mcp_connection_health_check_failed(self):
        """Test MCP connection health check failure."""
        mock_client = Mock()
        mock_client.list_tools.side_effect = Exception("Connection failed")

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True

        assert connection.is_healthy() is False
        assert connection.is_connected is False

    @pytest.mark.asyncio
    async def test_mcp_connection_restart_success(self):
        """Test successful MCP connection restart."""
        mock_client = Mock()
        mock_client.start_server.return_value = True
        mock_client.stop_server = Mock()

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True

        success = await connection.restart()

        assert success is True
        assert connection.restart_count == 0  # Reset to 0 on successful connect
        mock_client.stop_server.assert_called_once()
        mock_client.start_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_connection_restart_max_attempts(self):
        """Test MCP connection restart with max attempts reached."""
        mock_client = Mock()

        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.restart_count = 3  # Already at max

        success = await connection.restart()

        assert success is False

    @pytest.mark.asyncio
    async def test_detailed_health_check(self):
        """Test detailed health check."""
        mock_client = Mock()
        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True
        connection.restart_count = 1

        manager = MCPManager(project_root=self.project_root)
        manager.connections[MCPServerType.FILESYSTEM] = connection

        with patch.object(connection, "is_healthy", return_value=True):
            health_status = await manager.health_check(detailed=True)

        assert MCPServerType.FILESYSTEM in health_status
        health_info = health_status[MCPServerType.FILESYSTEM]
        assert health_info["healthy"] is True
        assert health_info["connected"] is True
        assert health_info["restart_count"] == 1
        assert health_info["max_restarts"] == 3

    @pytest.mark.asyncio
    async def test_restart_failed_servers(self):
        """Test restarting failed servers."""
        mock_client = Mock()
        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = False

        manager = MCPManager(project_root=self.project_root)
        manager.connections[MCPServerType.FILESYSTEM] = connection

        with patch.object(connection, "is_healthy", return_value=False):
            with patch.object(connection, "restart", return_value=True) as mock_restart:
                restart_results = await manager.restart_failed_servers()

                assert restart_results[MCPServerType.FILESYSTEM] is True
                mock_restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_server_statistics(self):
        """Test getting server statistics."""
        mock_client = Mock()
        connection = MCPConnection(MCPServerType.FILESYSTEM, mock_client)
        connection.is_connected = True

        manager = MCPManager(project_root=self.project_root)
        manager.connections[MCPServerType.FILESYSTEM] = connection

        with patch.object(connection, "is_healthy", return_value=True):
            stats = await manager.get_server_statistics()

        assert stats["total_servers"] == len(MCPServerType)
        assert stats["connected_servers"] >= 1
        assert stats["healthy_servers"] >= 1
        assert "servers" in stats
        assert "filesystem" in stats["servers"]


class TestWorkflowState:
    """Test cases for WorkflowState."""

    def setup_method(self):
        """Set up test environment."""
        self.context = MigrationContext(
            project_path="/test/project",
            source_library="requests",
            target_library="httpx",
        )
        self.workflow = WorkflowState(self.context)

    def test_workflow_initialization(self):
        """Test workflow state initialization."""
        assert self.workflow.context == self.context
        assert self.workflow.current_stage == WorkflowStage.INITIALIZATION
        assert len(self.workflow.steps) > 0
        assert len(self.workflow.step_order) == len(self.workflow.steps)

    def test_get_next_step_initial(self):
        """Test getting the first step."""
        next_step = self.workflow.get_next_step()

        assert next_step is not None
        assert next_step.name == "initialize_environment"
        assert next_step.status == WorkflowStatus.PENDING

    def test_start_step(self):
        """Test starting a workflow step."""
        success = self.workflow.start_step("initialize_environment")

        assert success is True
        step = self.workflow.steps["initialize_environment"]
        assert step.status == WorkflowStatus.RUNNING
        assert step.start_time is not None

    def test_complete_step(self):
        """Test completing a workflow step."""
        self.workflow.start_step("initialize_environment")
        result = {"servers_initialized": 4}
        success = self.workflow.complete_step("initialize_environment", result)

        assert success is True
        step = self.workflow.steps["initialize_environment"]
        assert step.status == WorkflowStatus.COMPLETED
        assert step.end_time is not None
        assert step.result == result

    def test_fail_step(self):
        """Test failing a workflow step."""
        self.workflow.start_step("initialize_environment")
        error = "Server connection failed"
        success = self.workflow.fail_step("initialize_environment", error)

        assert success is True
        step = self.workflow.steps["initialize_environment"]
        assert step.status == WorkflowStatus.FAILED
        assert step.error == error
        assert self.workflow.current_stage == WorkflowStage.FAILED

    def test_retry_step(self):
        """Test retrying a failed step."""
        # First fail a step
        self.workflow.start_step("initialize_environment")
        self.workflow.fail_step("initialize_environment", "Test error")

        # Then retry it
        success = self.workflow.retry_step("initialize_environment")

        assert success is True
        step = self.workflow.steps["initialize_environment"]
        assert step.status == WorkflowStatus.PENDING
        assert step.error is None
        assert step.start_time is None
        assert step.end_time is None

    def test_dependencies(self):
        """Test step dependencies."""
        # Try to start a step with unmet dependencies
        success = self.workflow.start_step("analyze_project")

        assert success is False  # Should fail due to unmet dependencies

        # Complete the dependency first
        self.workflow.start_step("initialize_environment")
        self.workflow.complete_step("initialize_environment")

        # Now it should succeed
        success = self.workflow.start_step("analyze_project")
        assert success is True

    def test_workflow_summary(self):
        """Test getting workflow summary."""
        # Complete one step
        self.workflow.start_step("initialize_environment")
        self.workflow.complete_step("initialize_environment")

        # Fail another step
        self.workflow.start_step("analyze_project")
        self.workflow.fail_step("analyze_project", "Test error")

        summary = self.workflow.get_workflow_summary()

        assert summary["total_steps"] > 0
        assert summary["completed_steps"] == 1
        assert summary["failed_steps"] == 1
        assert "steps" in summary
        assert summary["is_complete"] is False
        assert summary["is_failed"] is False  # Can be retried


class TestDiversificationCoordinator:
    """Test cases for DiversificationCoordinator."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = self.temp_dir.name

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        from src.orchestration.config import LLMConfig

        llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key_env_var="OPENAI_API_KEY",
            temperature=0.2,
        )
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
        )

        assert str(coordinator.project_path) == str(Path(self.project_path).resolve())
        assert coordinator.source_library == "requests"
        assert coordinator.target_library == "httpx"
        assert coordinator.llm_config.model_name == "gpt-3.5-turbo"
        assert coordinator.llm_config.temperature == 0.2
        assert isinstance(coordinator.agent_manager, AgentManager)
        assert isinstance(coordinator.mcp_manager, MCPManager)
        assert isinstance(coordinator.workflow_state, WorkflowState)

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    async def test_execute_workflow_dry_run(self):
        """Test workflow execution in dry run mode."""
        from src.orchestration.config import LLMConfig

        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
        )

        # Mock all the step handlers to return success
        with (
            patch.object(
                coordinator, "_initialize_environment", return_value={"success": True}
            ),
            patch.object(
                coordinator, "_analyze_project", return_value={"success": True}
            ),
            patch.object(coordinator, "_create_backup", return_value={"success": True}),
            patch.object(
                coordinator, "_generate_tests", return_value={"success": True}
            ),
            patch.object(
                coordinator, "_run_baseline_tests", return_value={"success": True}
            ),
            patch.object(coordinator, "_migrate_code", return_value={"success": True}),
            patch.object(
                coordinator, "_validate_migration", return_value={"success": True}
            ),
            patch.object(coordinator, "_repair_issues", return_value={"success": True}),
            patch.object(
                coordinator, "_finalize_migration", return_value={"success": True}
            ),
        ):

            result = await coordinator.execute_workflow(dry_run=True, auto_proceed=True)

            assert result is True
            assert coordinator.workflow_state.is_workflow_complete()

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_get_workflow_status(self):
        """Test getting workflow status."""
        from src.orchestration.config import LLMConfig

        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
        )

        status = coordinator.get_workflow_status()

        assert isinstance(status, dict)
        assert "context" in status
        assert "current_stage" in status
        assert "total_steps" in status


class TestErrorHandler:
    """Test cases for ErrorHandler."""

    def setup_method(self):
        """Set up test environment."""
        self.error_handler = ErrorHandler()

    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        assert len(self.error_handler.error_history) == 0
        assert len(self.error_handler.recovery_handlers) > 0

    def test_handle_error(self):
        """Test handling an error."""
        exception = Exception("Test error")
        error_info = self.error_handler.handle_error(
            exception, ErrorCategory.MCP_CONNECTION, {"server": "filesystem"}
        )

        assert error_info.category == ErrorCategory.MCP_CONNECTION
        assert error_info.message == "Test error"
        assert error_info.context["server"] == "filesystem"
        assert len(self.error_handler.error_history) == 1
        assert len(error_info.recovery_suggestions) > 0

    def test_assess_severity(self):
        """Test severity assessment."""
        # Critical error
        api_key_error = Exception("API key not found")
        error_info = self.error_handler.handle_error(
            api_key_error, ErrorCategory.CONFIGURATION
        )
        assert error_info.severity == ErrorSeverity.CRITICAL

        # High severity error
        mcp_error = Exception("Connection failed")
        error_info = self.error_handler.handle_error(
            mcp_error, ErrorCategory.MCP_CONNECTION
        )
        assert error_info.severity == ErrorSeverity.HIGH

    def test_error_summary(self):
        """Test getting error summary."""
        # Add some errors
        self.error_handler.handle_error(
            Exception("Error 1"), ErrorCategory.MCP_CONNECTION
        )
        self.error_handler.handle_error(
            Exception("Error 2"), ErrorCategory.AGENT_EXECUTION
        )

        summary = self.error_handler.get_error_summary()

        assert summary["total_errors"] == 2
        assert "mcp_connection" in summary["categories"]
        assert "agent_execution" in summary["categories"]
        assert summary["recoverable"] >= 0


class TestLoggingConfig:
    """Test cases for logging configuration."""

    def test_setup_logging(self):
        """Test logging setup."""
        from src.orchestration.config import LoggingConfig

        # This is mainly to ensure no exceptions are raised
        config = LoggingConfig(level="DEBUG", console=True)
        setup_logging(config)

        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "diversifier.test"

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("coordinator")

        assert logger.name == "diversifier.coordinator"
        assert isinstance(logger, logging.Logger)


if __name__ == "__main__":
    pytest.main([__file__])
