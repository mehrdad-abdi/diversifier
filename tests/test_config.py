"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import src.orchestration.config

from src.orchestration.config import (
    ConfigManager,
    DiversifierConfig,
    LLMConfig,
    LoggingConfig,
    MigrationConfig,
    get_config,
    get_config_manager,
)


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert (
            config.format_string
            == "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
        )

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoggingConfig(
            level="DEBUG",
            format_string="%(name)s - %(levelname)s - %(message)s",
        )
        assert config.level == "DEBUG"
        assert config.format_string == "%(name)s - %(levelname)s - %(message)s"


class TestMigrationConfig:
    """Tests for MigrationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MigrationConfig()
        assert config.test_paths == ["tests/", "test/"]
        assert len(config.common_project_files) > 0


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            api_key_env_var="TEST_API_KEY",
        )
        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-5-sonnet-20241022"
        assert config.temperature == 0.1
        assert config.max_tokens == 200000
        assert config.api_key_env_var == "TEST_API_KEY"
        assert config.additional_params == {}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=False)
    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider="openai",
            model_name="gpt-4",
            api_key_env_var="OPENAI_API_KEY",
            temperature=0.7,
            max_tokens=2048,
            additional_params={"top_p": 0.9, "frequency_penalty": 0.1},
        )
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.api_key_env_var == "OPENAI_API_KEY"
        assert config.additional_params == {"top_p": 0.9, "frequency_penalty": 0.1}

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}, clear=False)
    def test_google_provider_config(self):
        """Test configuration for Google/Gemini provider."""
        config = LLMConfig(
            provider="google",
            model_name="gemini-pro",
            api_key_env_var="GOOGLE_API_KEY",
            temperature=0.5,
        )
        assert config.provider == "google"
        assert config.model_name == "gemini-pro"
        assert config.temperature == 0.5
        assert config.api_key_env_var == "GOOGLE_API_KEY"


class TestDiversifierConfig:
    """Tests for main DiversifierConfig dataclass."""

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_default_values(self):
        """Test default configuration values."""
        llm_config = LLMConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            api_key_env_var="TEST_API_KEY",
        )
        config = DiversifierConfig(llm=llm_config)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.migration, MigrationConfig)
        assert isinstance(config.llm, LLMConfig)


class TestConfigManager:
    """Tests for ConfigManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing global config manager
        src.orchestration.config._config_manager = None

    def test_init_with_required_config_path(self):
        """Test initialization with required config path."""
        manager = ConfigManager("/path/to/config.toml")
        assert manager.config_path == Path("/path/to/config.toml")
        assert manager._config is None

    def test_init_with_path_object(self):
        """Test initialization with Path object."""
        path = Path("/path/to/config.toml")
        manager = ConfigManager(path)
        assert manager.config_path == path
        assert manager._config is None

    def test_load_config_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        manager = ConfigManager("/nonexistent/config.toml")
        with pytest.raises(ValueError, match="Configuration file not found"):
            manager.load_config()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=False)
    def test_load_config_from_file(self):
        """Test loading configuration from TOML file."""
        toml_content = """
[logging]
level = "DEBUG"
format_string = "%(name)s - %(levelname)s - %(message)s"

[migration]
test_paths = ["tests/", "test/"]

[llm]
provider = "openai"
model_name = "gpt-4"
temperature = 0.7
max_tokens = 2048
api_key_env_var = "OPENAI_API_KEY"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            try:
                manager = ConfigManager(f.name)
                config = manager.load_config()

                assert config.logging.level == "DEBUG"
                assert (
                    config.logging.format_string
                    == "%(name)s - %(levelname)s - %(message)s"
                )
                assert config.migration.test_paths == ["tests/", "test/"]
                assert config.llm.provider == "openai"
                assert config.llm.model_name == "gpt-4"
                assert config.llm.temperature == 0.7
                assert config.llm.max_tokens == 2048
                assert config.llm.api_key_env_var == "OPENAI_API_KEY"
            finally:
                os.unlink(f.name)

    def test_load_config_invalid_file(self):
        """Test loading configuration from invalid TOML file."""
        invalid_toml = "invalid toml content ["

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(invalid_toml)
            f.flush()

            try:
                manager = ConfigManager(f.name)
                with pytest.raises(ValueError, match="Failed to load config file"):
                    manager.load_config()
            finally:
                os.unlink(f.name)

    def test_apply_env_overrides(self):
        """Test applying environment variable overrides."""
        # Create a basic TOML config file
        toml_content = """
[llm]
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
api_key_env_var = "TEST_API_KEY"
"""
        env_vars = {
            "TEST_API_KEY": "test-key",
            "DIVERSIFIER_LOG_LEVEL": "ERROR",
            "DIVERSIFIER_LLM_PROVIDER": "google",
            "DIVERSIFIER_LLM_MODEL_NAME": "gemini-pro",
            "DIVERSIFIER_LLM_TEMPERATURE": "0.8",
            "DIVERSIFIER_LLM_MAX_TOKENS": "8192",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False
            ) as f:
                f.write(toml_content)
                f.flush()

                try:
                    manager = ConfigManager(f.name)
                    config = manager.load_config()

                    assert config.logging.level == "ERROR"
                    assert config.llm.provider == "google"
                    assert config.llm.model_name == "gemini-pro"
                    assert config.llm.temperature == 0.8
                    assert config.llm.max_tokens == 8192
                finally:
                    os.unlink(f.name)

    def test_convert_env_value_boolean(self):
        """Test environment value conversion for booleans."""
        manager = ConfigManager("/dummy/path.toml")

        # Boolean handling no longer exists, so test string conversion
        assert manager._convert_env_value("true", "provider") == "true"
        assert manager._convert_env_value("false", "provider") == "false"

    def test_convert_env_value_integer(self):
        """Test environment value conversion for integers."""
        manager = ConfigManager("/dummy/path.toml")

        assert manager._convert_env_value("123", "max_tokens") == 123
        assert manager._convert_env_value("0", "max_tokens") == 0
        assert manager._convert_env_value("4096", "max_tokens") == 4096

    def test_convert_env_value_float(self):
        """Test environment value conversion for floats."""
        manager = ConfigManager("/dummy/path.toml")

        assert manager._convert_env_value("0.7", "temperature") == 0.7
        assert manager._convert_env_value("1.0", "temperature") == 1.0
        assert manager._convert_env_value("0.1", "temperature") == 0.1

    def test_convert_env_value_string(self):
        """Test environment value conversion for strings."""
        manager = ConfigManager("/dummy/path.toml")

        assert manager._convert_env_value("test", "level") == "test"
        assert manager._convert_env_value("path/to/file", "file_path") == "path/to/file"

    def test_get_config_caches_result(self):
        """Test that get_config caches the result."""
        # Create a basic TOML config file
        toml_content = """
[llm]
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
api_key_env_var = "TEST_API_KEY"
"""
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False
            ) as f:
                f.write(toml_content)
                f.flush()

                try:
                    manager = ConfigManager(f.name)
                    config1 = manager.get_config()
                    config2 = manager.get_config()

                    assert config1 is config2
                finally:
                    os.unlink(f.name)

    def test_reload_config(self):
        """Test reloading configuration."""
        # Create a basic TOML config file
        toml_content = """
[llm]
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
api_key_env_var = "TEST_API_KEY"
"""
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False
            ) as f:
                f.write(toml_content)
                f.flush()

                try:
                    manager = ConfigManager(f.name)
                    config1 = manager.load_config()
                    config2 = manager.reload_config()

                    assert config1 is not config2
                    assert isinstance(config2, DiversifierConfig)
                finally:
                    os.unlink(f.name)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
    def test_save_config_template(self):
        """Test saving configuration template."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            template_path = f.name

        try:
            manager = ConfigManager("/dummy/path.toml")
            manager.save_config_template(template_path)

            assert Path(template_path).exists()

            # Test that template can be loaded
            template_manager = ConfigManager(template_path)
            config = template_manager.load_config()
            assert isinstance(config, DiversifierConfig)

        finally:
            if Path(template_path).exists():
                os.unlink(template_path)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
    def test_config_template_max_tokens_update(self):
        """Test that config template can be updated with different max_tokens values.

        This simulates the GitHub workflow scenario where sed is used to update
        max_tokens from the default value to a higher value.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            template_path = f.name

        try:
            # Create config template with default values
            manager = ConfigManager("/dummy/path.toml")
            manager.save_config_template(template_path)

            # Read the template content
            with open(template_path, "r") as f:
                template_content = f.read()

            # Verify template has the new default (200000)
            assert "max_tokens = 200000" in template_content

            # Simulate sed operation to change max_tokens (like GitHub workflow does)
            updated_content = template_content.replace(
                "max_tokens = 200000", "max_tokens = 500000"
            )

            # Write the updated content back
            with open(template_path, "w") as f:
                f.write(updated_content)

            # Load the updated config and verify the change took effect
            updated_manager = ConfigManager(template_path)
            config = updated_manager.load_config()

            assert (
                config.llm.max_tokens == 500000
            ), f"Expected 500000, got {config.llm.max_tokens}"

            # Verify other values remain unchanged
            assert config.llm.provider == "anthropic"
            assert config.llm.model_name == "claude-3-5-sonnet-20241022"
            assert config.llm.temperature == 0.1

        finally:
            if Path(template_path).exists():
                os.unlink(template_path)


class TestGlobalFunctions:
    """Tests for global configuration functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing global config manager
        src.orchestration.config._config_manager = None

    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns singleton instance."""
        manager1 = get_config_manager("/test/path.toml")
        manager2 = get_config_manager("/test/path.toml")

        assert manager1 is manager2
        assert isinstance(manager1, ConfigManager)

    def test_get_config_manager_with_path(self):
        """Test get_config_manager with config path."""
        manager = get_config_manager("/test/path.toml")
        assert manager.config_path == Path("/test/path.toml")

    def test_get_config(self):
        """Test get_config function."""
        # Create a basic TOML config file
        toml_content = """
[llm]
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
api_key_env_var = "TEST_API_KEY"
"""
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False
            ) as f:
                f.write(toml_content)
                f.flush()

                try:
                    config = get_config(f.name)
                    assert isinstance(config, DiversifierConfig)
                finally:
                    os.unlink(f.name)

    @patch.dict(
        os.environ,
        {"DIVERSIFIER_LLM_TEMPERATURE": "0.5", "TEST_API_KEY": "test-key"},
        clear=False,
    )
    def test_get_config_with_env_vars(self):
        """Test get_config with environment variables."""
        # Create a basic TOML config file
        toml_content = """
[llm]
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
api_key_env_var = "TEST_API_KEY"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            try:
                config = get_config(f.name)
                assert config.llm.temperature == 0.5
            finally:
                os.unlink(f.name)

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False)
    def test_load_config_filters_unknown_keys(self):
        """Test that unknown top-level keys are filtered out during config loading."""
        toml_content = """
# Top-level configuration with some unknown keys
performance = "high"  # Unknown key that should be filtered out
unknown_param = "should_be_ignored"  # Another unknown key

[logging]
level = "DEBUG"

[migration]
test_paths = ["tests/", "test/"]

[llm]
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
api_key_env_var = "TEST_API_KEY"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            try:
                manager = ConfigManager(f.name)
                # This should not raise an error about unexpected keyword arguments
                config = manager.load_config()

                # Check that known parameters are loaded correctly
                assert config.logging.level == "DEBUG"
                assert config.migration.test_paths == ["tests/", "test/"]
                assert config.logging.level == "DEBUG"
                assert config.llm.provider == "anthropic"

                # The config should be created successfully despite unknown keys
                assert isinstance(config, DiversifierConfig)
            finally:
                os.unlink(f.name)
