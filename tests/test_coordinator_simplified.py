"""Tests for simplified DiversificationCoordinator."""

import pytest
import tempfile
import os
from unittest.mock import patch

from src.orchestration.coordinator import DiversificationCoordinator
from src.orchestration.config import LLMConfig, MigrationConfig


class TestDiversificationCoordinator:
    """Test cases for simplified DiversificationCoordinator."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = self.temp_dir.name

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_coordinator_initialization(self):
        """Test coordinator initialization without agents."""
        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )
        migration_config = MigrationConfig()
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
            migration_config=migration_config,
        )

        assert os.path.realpath(str(coordinator.project_path)) == os.path.realpath(
            self.project_path
        )
        assert coordinator.source_library == "requests"
        assert coordinator.target_library == "httpx"
        assert hasattr(coordinator, "mcp_manager")
        assert not hasattr(coordinator, "agent_manager")

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    @pytest.mark.asyncio
    async def test_placeholder_methods_return_success(self):
        """Test that placeholder methods return success."""
        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )
        migration_config = MigrationConfig()
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
            migration_config=migration_config,
        )

        # Test migrate_code placeholder
        result = await coordinator._migrate_code()
        assert result == {"success": True}

        # Test validate_migration placeholder
        result = await coordinator._validate_migration()
        assert result == {"success": True}

        # Test repair_issues placeholder
        result = await coordinator._repair_issues()
        assert result == {"success": True}

        # Test finalize_migration placeholder
        result = await coordinator._finalize_migration()
        assert result == {"success": True}

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_get_workflow_status(self):
        """Test getting workflow status."""
        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )
        migration_config = MigrationConfig()
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
            migration_config=migration_config,
        )

        status = coordinator.get_workflow_status()

        assert isinstance(status, dict)
        assert "total_steps" in status
        assert "completed_steps" in status
        assert "source_library" in status
        assert "target_library" in status

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    @pytest.mark.asyncio
    async def test_cleanup_without_agents(self):
        """Test cleanup method without agent manager."""
        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="TEST_API_KEY",
        )
        migration_config = MigrationConfig()
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
            migration_config=migration_config,
        )

        # Mock the mcp_manager shutdown method
        with patch.object(
            coordinator.mcp_manager, "shutdown_all_servers"
        ) as mock_shutdown:
            await coordinator._cleanup()
            mock_shutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
