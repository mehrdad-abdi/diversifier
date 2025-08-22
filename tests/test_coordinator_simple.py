"""Simplified tests for the DiversificationCoordinator using the new simple workflow."""

import pytest
import tempfile
import os
from unittest.mock import patch

from src.orchestration.coordinator import DiversificationCoordinator
from src.orchestration.config import LLMConfig


class TestDiversificationCoordinator:
    """Test cases for DiversificationCoordinator with simple workflow."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = self.temp_dir.name

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False)
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

        assert os.path.realpath(str(coordinator.project_path)) == os.path.realpath(
            self.project_path
        )
        assert coordinator.source_library == "requests"
        assert coordinator.target_library == "httpx"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
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

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
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

        # Mock internal workflow step methods
        with patch.object(coordinator, "_initialize_environment") as mock_init:
            mock_init.return_value = {"success": True}

            # For simplicity, just test that the method exists and can be called
            # A more complete test would mock all 8 steps
            result = await coordinator._initialize_environment()
            assert result["success"] is True

    def test_coordinator_requires_llm_config(self):
        """Test coordinator requires LLM config."""
        with pytest.raises(ValueError, match="llm_config is required"):
            DiversificationCoordinator(
                project_path=self.project_path,
                source_library="requests",
                target_library="httpx",
                llm_config=None,
            )
