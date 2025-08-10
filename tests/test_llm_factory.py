"""Simplified tests for LLM factory functionality."""

import os
from unittest.mock import patch

from src.orchestration.config import LLMConfig
from src.orchestration.llm_factory import (
    get_default_api_key_env_var,
    get_supported_providers,
    validate_llm_config,
)


class TestLLMFactory:
    """Tests for LLM factory functions."""

    def test_get_supported_providers(self):
        """Test getting list of supported providers."""
        providers = get_supported_providers()
        assert isinstance(providers, list)
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google_genai" in providers  # Updated to match new provider naming

    def test_get_default_api_key_env_var(self):
        """Test getting default API key environment variables."""
        assert get_default_api_key_env_var("anthropic") == "ANTHROPIC_API_KEY"
        assert get_default_api_key_env_var("openai") == "OPENAI_API_KEY"
        assert get_default_api_key_env_var("google") == "GOOGLE_API_KEY"
        assert get_default_api_key_env_var("unknown") == "UNKNOWN_API_KEY"

    def test_validate_llm_config_valid(self):
        """Test validation of valid LLM configuration."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = LLMConfig(
                provider="anthropic",
                model_name="claude-3-sonnet",
                api_key_env_var="ANTHROPIC_API_KEY",
                temperature=0.5,
                max_tokens=1000,
                timeout=60,
                retry_attempts=3,
            )
            issues = validate_llm_config(config)
            assert len(issues) == 0

    def test_validate_llm_config_invalid_provider(self):
        """Test validation with invalid provider."""
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}):
            config = LLMConfig(
                provider="invalid_provider",
                model_name="test-model",
                api_key_env_var="TEST_API_KEY",
            )
            issues = validate_llm_config(config)
            # Now it's just an informational note, not an error - but should still generate an issue
        assert any(
            "not in the list of commonly used providers" in issue for issue in issues
        )

    def test_validate_llm_config_invalid_temperature(self):
        """Test validation with invalid temperature."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = LLMConfig(
                provider="anthropic",
                model_name="claude-3-sonnet",
                api_key_env_var="ANTHROPIC_API_KEY",
                temperature=1.5,  # Too high
            )
            issues = validate_llm_config(config)
            assert any("Temperature must be between" in issue for issue in issues)

            config = LLMConfig(
                provider="anthropic",
                model_name="claude-3-sonnet",
                api_key_env_var="ANTHROPIC_API_KEY",
                temperature=-0.1,  # Too low
            )
            issues = validate_llm_config(config)
            assert any("Temperature must be between" in issue for issue in issues)

    def test_validate_llm_config_missing_api_key(self):
        """Test validation with missing API key."""
        # Ensure API key is not in environment
        with patch.dict(os.environ, {}, clear=True):
            # This should raise a ValueError during LLMConfig creation due to __post_init__
            try:
                config = LLMConfig(
                    provider="anthropic",
                    model_name="claude-3-sonnet",
                    api_key_env_var="ANTHROPIC_API_KEY",
                )
                # If we get here, validation should catch the missing key
                issues = validate_llm_config(config)
                assert any("API key not found" in issue for issue in issues)
            except ValueError as e:
                # This is expected - the __post_init__ validates the API key
                assert "API key environment variable" in str(e)

    def test_validate_llm_config_custom_api_key_env_var(self):
        """Test validation with custom API key environment variable."""
        with patch.dict(os.environ, {"CUSTOM_API_KEY": "test-key"}):
            config = LLMConfig(
                provider="anthropic",
                model_name="claude-3-sonnet",
                api_key_env_var="CUSTOM_API_KEY",
            )
            issues = validate_llm_config(config)
            assert len(issues) == 0
