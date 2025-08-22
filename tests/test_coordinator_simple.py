"""Simplified tests for the DiversificationCoordinator using the new simple workflow."""

import pytest
import tempfile
from unittest.mock import patch

from src.orchestration.coordinator import DiversificationCoordinator
from src.orchestration.config import LLMConfig
from src.orchestration.simple_workflow import MigrationWorkflow


class TestDiversificationCoordinator:
    """Test cases for DiversificationCoordinator with simple workflow."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = self.temp_dir.name

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
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

        assert isinstance(coordinator.workflow, MigrationWorkflow)
        assert coordinator.workflow.source_library == "requests"
        assert coordinator.workflow.target_library == "httpx"

    def test_get_workflow_status(self):
        """Test getting workflow status."""
        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="ANTHROPIC_API_KEY",
        )
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
        )

        status = coordinator.get_workflow_status()

        assert isinstance(status, dict)
        assert "total_steps" in status
        assert "completed_steps" in status
        assert "source_library" in status
        assert "target_library" in status
        assert status["source_library"] == "requests"
        assert status["target_library"] == "httpx"
        assert status["total_steps"] == 8

    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test executing the workflow."""
        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-sonnet",
            api_key_env_var="ANTHROPIC_API_KEY",
        )
        coordinator = DiversificationCoordinator(
            project_path=self.project_path,
            source_library="requests",
            target_library="httpx",
            llm_config=llm_config,
        )

        # Mock the workflow execution
        with patch.object(coordinator.workflow, "execute") as mock_execute:
            mock_execute.return_value = True

            result = await coordinator.execute_workflow()

            assert result is True
            mock_execute.assert_called_once()

    def test_coordinator_requires_llm_config(self):
        """Test coordinator requires LLM config."""
        with pytest.raises(ValueError, match="llm_config is required"):
            DiversificationCoordinator(
                project_path=self.project_path,
                source_library="requests",
                target_library="httpx",
                llm_config=None,
            )
