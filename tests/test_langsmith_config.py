"""Tests for LangSmith tracing configuration."""

import os
from unittest.mock import patch

from src.orchestration.langsmith_config import (
    setup_langsmith_tracing,
    get_langsmith_status,
)


class TestLangSmithConfig:
    """Test LangSmith configuration functionality."""

    def test_setup_langsmith_tracing_disabled_by_default(self):
        """Test that LangSmith tracing is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            result = setup_langsmith_tracing()
            assert result is False

    def test_setup_langsmith_tracing_enabled_without_api_key(self, capsys):
        """Test that LangSmith tracing shows warning when API key is missing."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}, clear=True):
            result = setup_langsmith_tracing()
            assert result is False
            captured = capsys.readouterr()
            assert "LANGSMITH_API_KEY is not set" in captured.out

    def test_setup_langsmith_tracing_enabled_with_api_key(self, capsys):
        """Test that LangSmith tracing is properly configured with API key."""
        env_vars = {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "test-api-key",
            "LANGSMITH_PROJECT": "test-project",
            "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = setup_langsmith_tracing()
            assert result is True

            # Check that LangChain environment variables are set
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGCHAIN_API_KEY") == "test-api-key"
            assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"
            assert (
                os.environ.get("LANGCHAIN_ENDPOINT")
                == "https://api.smith.langchain.com"
            )

            captured = capsys.readouterr()
            assert "LangSmith tracing enabled for project: test-project" in captured.out

    def test_setup_langsmith_tracing_uses_defaults(self):
        """Test that LangSmith uses default values when not specified."""
        env_vars = {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "test-api-key"}

        with patch.dict(os.environ, env_vars, clear=True):
            result = setup_langsmith_tracing()
            assert result is True

            assert os.environ.get("LANGCHAIN_PROJECT") == "diversifier"
            assert (
                os.environ.get("LANGCHAIN_ENDPOINT")
                == "https://api.smith.langchain.com"
            )

    def test_get_langsmith_status(self):
        """Test that status reporting works correctly."""
        env_vars = {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "test-key",
            "LANGSMITH_PROJECT": "test-project",
            "LANGSMITH_ENDPOINT": "https://test.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            status = get_langsmith_status()
            assert status["tracing_enabled"] == "true"
            assert status["project"] == "test-project"
            assert status["endpoint"] == "https://test.com"
            assert status["api_key_set"] == "Yes"

    def test_get_langsmith_status_no_config(self):
        """Test status reporting when no configuration is present."""
        with patch.dict(os.environ, {}, clear=True):
            status = get_langsmith_status()
            assert status["tracing_enabled"] == "false"
            assert status["api_key_set"] == "No"
            assert status["langchain_tracing"] == "false"
