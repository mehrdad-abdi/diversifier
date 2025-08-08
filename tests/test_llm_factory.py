"""Tests for LLM factory functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.config import LLMConfig
from src.orchestration.llm_factory import (
    create_llm_from_config,
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
        assert "google" in providers

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
                temperature=0.5,
                max_tokens=1000,
                timeout=60,
                retry_attempts=3,
            )
            issues = validate_llm_config(config)
            assert len(issues) == 0

    def test_validate_llm_config_invalid_provider(self):
        """Test validation with invalid provider."""
        config = LLMConfig(provider="invalid_provider")
        issues = validate_llm_config(config)
        assert any("Unsupported provider" in issue for issue in issues)

    def test_validate_llm_config_invalid_temperature(self):
        """Test validation with invalid temperature."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = LLMConfig(temperature=1.5)  # Too high
            issues = validate_llm_config(config)
            assert any("Temperature must be between" in issue for issue in issues)

            config = LLMConfig(temperature=-0.1)  # Too low
            issues = validate_llm_config(config)
            assert any("Temperature must be between" in issue for issue in issues)

    def test_validate_llm_config_invalid_max_tokens(self):
        """Test validation with invalid max_tokens."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = LLMConfig(max_tokens=0)
            issues = validate_llm_config(config)
            assert any("max_tokens must be positive" in issue for issue in issues)

    def test_validate_llm_config_invalid_timeout(self):
        """Test validation with invalid timeout."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = LLMConfig(timeout=0)
            issues = validate_llm_config(config)
            assert any("timeout must be positive" in issue for issue in issues)

    def test_validate_llm_config_invalid_retry_attempts(self):
        """Test validation with invalid retry_attempts."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            config = LLMConfig(retry_attempts=-1)
            issues = validate_llm_config(config)
            assert any(
                "retry_attempts must be non-negative" in issue for issue in issues
            )

    def test_validate_llm_config_missing_api_key(self):
        """Test validation with missing API key."""
        # Ensure API key is not in environment
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig(provider="anthropic")
            issues = validate_llm_config(config)
            assert any("API key not found" in issue for issue in issues)

    def test_validate_llm_config_custom_api_key_env_var(self):
        """Test validation with custom API key environment variable."""
        with patch.dict(os.environ, {"CUSTOM_API_KEY": "test-key"}):
            config = LLMConfig(provider="anthropic", api_key_env_var="CUSTOM_API_KEY")
            issues = validate_llm_config(config)
            assert len(issues) == 0

    def test_create_anthropic_llm(self):
        """Test creating Anthropic LLM instance."""
        mock_instance = MagicMock()
        mock_anthropic = MagicMock(return_value=mock_instance)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                config = LLMConfig(
                    provider="anthropic",
                    model_name="claude-3-sonnet",
                    temperature=0.5,
                    max_tokens=1000,
                )

                result = create_llm_from_config(config)

                assert result == mock_instance
                mock_anthropic.assert_called_once()

                # Check that correct parameters were passed
                call_kwargs = mock_anthropic.call_args[1]
                assert call_kwargs["model"] == "claude-3-sonnet"
                assert call_kwargs["temperature"] == 0.5
                assert call_kwargs["max_tokens"] == 1000
                assert call_kwargs["api_key"] == "test-key"

    def test_create_openai_llm(self):
        """Test creating OpenAI LLM instance."""
        mock_instance = MagicMock()
        mock_openai = MagicMock(return_value=mock_instance)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("langchain_openai.ChatOpenAI", mock_openai):
                config = LLMConfig(
                    provider="openai", model_name="gpt-4", temperature=0.7, max_tokens=2048
                )

                result = create_llm_from_config(config)

                assert result == mock_instance
                mock_openai.assert_called_once()

                # Check that correct parameters were passed
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["model"] == "gpt-4"
                assert call_kwargs["temperature"] == 0.7
                assert call_kwargs["max_tokens"] == 2048
                assert call_kwargs["api_key"] == "test-key"

    def test_create_google_llm(self):
        """Test creating Google LLM instance."""
        mock_instance = MagicMock()
        mock_google = MagicMock(return_value=mock_instance)

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("langchain_google_genai.ChatGoogleGenerativeAI", mock_google):
                config = LLMConfig(
                    provider="google",
                    model_name="gemini-pro",
                    temperature=0.3,
                    max_tokens=1500,
                )

                result = create_llm_from_config(config)

                assert result == mock_instance
                mock_google.assert_called_once()

                # Check that correct parameters were passed
                call_kwargs = mock_google.call_args[1]
                assert call_kwargs["model"] == "gemini-pro"
                assert call_kwargs["temperature"] == 0.3
                assert (
                    call_kwargs["max_output_tokens"] == 1500
                )  # Google uses max_output_tokens
                assert call_kwargs["google_api_key"] == "test-key"

    def test_create_llm_unsupported_provider(self):
        """Test creating LLM with unsupported provider."""
        config = LLMConfig(provider="unsupported")

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_from_config(config)

    def test_create_llm_missing_api_key(self):
        """Test creating LLM with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig(provider="anthropic")

            with pytest.raises(ValueError, match="API key not found"):
                create_llm_from_config(config)

    def test_create_llm_with_custom_api_key_env_var(self):
        """Test creating LLM with custom API key environment variable."""
        mock_instance = MagicMock()
        mock_anthropic = MagicMock(return_value=mock_instance)

        with patch.dict(os.environ, {"CUSTOM_KEY": "test-key"}):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                config = LLMConfig(provider="anthropic", api_key_env_var="CUSTOM_KEY")

                result = create_llm_from_config(config)

                assert result == mock_instance
                call_kwargs = mock_anthropic.call_args[1]
                assert call_kwargs["api_key"] == "test-key"

    def test_create_llm_with_base_url(self):
        """Test creating LLM with custom base URL."""
        mock_instance = MagicMock()
        mock_anthropic = MagicMock(return_value=mock_instance)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                config = LLMConfig(provider="anthropic", base_url="https://custom.api.com")

                result = create_llm_from_config(config)

                assert result == mock_instance
                call_kwargs = mock_anthropic.call_args[1]
                assert call_kwargs["base_url"] == "https://custom.api.com"

    def test_create_llm_with_additional_params(self):
        """Test creating LLM with additional parameters."""
        mock_instance = MagicMock()
        mock_anthropic = MagicMock(return_value=mock_instance)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                config = LLMConfig(
                    provider="anthropic",
                    additional_params={"top_p": 0.9, "stop": ["Human:"]},
                )

                result = create_llm_from_config(config)

                assert result == mock_instance
                call_kwargs = mock_anthropic.call_args[1]
                assert call_kwargs["top_p"] == 0.9
                assert call_kwargs["stop"] == ["Human:"]

    def test_create_llm_from_global_config(self):
        """Test creating LLM using global configuration."""
        mock_instance = MagicMock()
        mock_anthropic = MagicMock(return_value=mock_instance)

        # Mock global config
        mock_config = MagicMock()
        mock_config.llm = LLMConfig(provider="anthropic")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.orchestration.config.get_config", return_value=mock_config):
                with patch("langchain_anthropic.ChatAnthropic", mock_anthropic):
                    result = create_llm_from_config()  # No config parameter

                    assert result == mock_instance

    def test_create_llm_import_error_anthropic(self):
        """Test handling import error for Anthropic."""
        config = LLMConfig(provider="anthropic")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch(
                "langchain_anthropic.ChatAnthropic", side_effect=ImportError
            ):
                with pytest.raises(
                    ImportError, match="langchain_anthropic package is required"
                ):
                    create_llm_from_config(config)

    def test_create_llm_import_error_openai(self):
        """Test handling import error for OpenAI."""
        config = LLMConfig(provider="openai")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch(
                "langchain_openai.ChatOpenAI", side_effect=ImportError
            ):
                with pytest.raises(
                    ImportError, match="langchain_openai package is required"
                ):
                    create_llm_from_config(config)

    def test_create_llm_import_error_google(self):
        """Test handling import error for Google."""
        config = LLMConfig(provider="google")

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch(
                "langchain_google_genai.ChatGoogleGenerativeAI",
                side_effect=ImportError,
            ):
                with pytest.raises(
                    ImportError, match="langchain_google_genai package is required"
                ):
                    create_llm_from_config(config)
