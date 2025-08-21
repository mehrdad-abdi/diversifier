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
    MCPConfig,
    MigrationConfig,
    PerformanceConfig,
    get_config,
    get_config_manager,
)


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.console is True
        assert config.file_path is None
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.backup_count == 5
        assert config.enable_correlation_ids is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoggingConfig(
            level="DEBUG",
            console=False,
            file_path="custom.log",
            max_file_size=5000000,
            backup_count=3,
            enable_correlation_ids=False,
        )
        assert config.level == "DEBUG"
        assert config.console is False
        assert config.file_path == "custom.log"
        assert config.max_file_size == 5000000
        assert config.backup_count == 3
        assert config.enable_correlation_ids is False


class TestMCPConfig:
    """Tests for MCPConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MCPConfig()
        assert config.filesystem_server_path == "src/mcp_servers/filesystem/server.py"
        assert config.testing_server_path == "src/mcp_servers/testing/server.py"
        assert config.git_server_path == "src/mcp_servers/git/server.py"
        assert config.docker_server_path == "src/mcp_servers/docker/server.py"
        assert config.timeout == 30


class TestMigrationConfig:
    """Tests for MigrationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MigrationConfig()
        assert config.max_iterations == 5
        assert config.test_timeout == 300
        assert config.backup_original is True
        assert config.validate_syntax is True
        assert config.require_test_coverage is True
        assert config.min_test_coverage == 0.8
        assert config.test_paths == ["tests/", "test/"]
        assert len(config.allowed_library_pairs) == 4


class TestPerformanceConfig:
    """Tests for PerformanceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PerformanceConfig()
        assert config.enable_metrics is True
        assert config.metrics_file == "performance_metrics.json"
        assert config.log_slow_operations is True
        assert config.slow_operation_threshold == 1.0
        assert config.enable_memory_tracking is False


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
        assert config.max_tokens == 4096
        assert config.timeout == 120
        assert config.api_key_env_var == "TEST_API_KEY"
        assert config.base_url is None
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
            timeout=60,
            base_url="https://api.openai.com/v1",
            additional_params={"top_p": 0.9, "frequency_penalty": 0.1},
        )
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 60
        assert config.api_key_env_var == "OPENAI_API_KEY"
        assert config.base_url == "https://api.openai.com/v1"
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
        assert isinstance(config.mcp, MCPConfig)
        assert isinstance(config.migration, MigrationConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.llm, LLMConfig)
        assert config.project_root == "."
        assert config.temp_dir == "/tmp/diversifier"
        assert config.debug_mode is False


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
# Top-level configuration
project_root = "/test/path"
debug_mode = true

[logging]
level = "DEBUG"
console = false
file_path = "test.log"

[mcp]
timeout = 60

[migration]
max_iterations = 10

[performance]
enable_metrics = false

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
                assert config.logging.console is False
                assert config.logging.file_path == "test.log"
                assert config.mcp.timeout == 60
                assert config.migration.max_iterations == 10
                assert config.performance.enable_metrics is False
                assert config.llm.provider == "openai"
                assert config.llm.model_name == "gpt-4"
                assert config.llm.temperature == 0.7
                assert config.llm.max_tokens == 2048
                assert config.llm.api_key_env_var == "OPENAI_API_KEY"
                assert config.project_root == "/test/path"
                assert config.debug_mode is True
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
            "DIVERSIFIER_LOG_CONSOLE": "false",
            "DIVERSIFIER_MCP_TIMEOUT": "120",
            "DIVERSIFIER_DEBUG": "true",
            "DIVERSIFIER_MIN_COVERAGE": "0.9",
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
                    assert config.logging.console is False
                    assert config.mcp.timeout == 120
                    assert config.debug_mode is True
                    assert config.migration.min_test_coverage == 0.9
                    assert config.llm.provider == "google"
                    assert config.llm.model_name == "gemini-pro"
                    assert config.llm.temperature == 0.8
                    assert config.llm.max_tokens == 8192
                finally:
                    os.unlink(f.name)

    def test_convert_env_value_boolean(self):
        """Test environment value conversion for booleans."""
        manager = ConfigManager("/dummy/path.toml")

        # Test true values
        assert manager._convert_env_value("true", "logging", "console") is True
        assert manager._convert_env_value("1", "logging", "console") is True
        assert manager._convert_env_value("yes", "logging", "console") is True
        assert manager._convert_env_value("on", "logging", "console") is True
        assert manager._convert_env_value("TRUE", "logging", "console") is True

        # Test false values
        assert manager._convert_env_value("false", "logging", "console") is False
        assert manager._convert_env_value("0", "logging", "console") is False
        assert manager._convert_env_value("no", "logging", "console") is False
        assert manager._convert_env_value("off", "logging", "console") is False

    def test_convert_env_value_integer(self):
        """Test environment value conversion for integers."""
        manager = ConfigManager("/dummy/path.toml")

        assert manager._convert_env_value("123", "mcp", "timeout") == 123
        assert manager._convert_env_value("0", "mcp", "timeout") == 0
        assert manager._convert_env_value("-5", "mcp", "timeout") == -5

    def test_convert_env_value_float(self):
        """Test environment value conversion for floats."""
        manager = ConfigManager("/dummy/path.toml")

        assert (
            manager._convert_env_value("1.5", "migration", "min_test_coverage") == 1.5
        )
        assert (
            manager._convert_env_value("0.0", "migration", "min_test_coverage") == 0.0
        )
        assert (
            manager._convert_env_value("99.99", "migration", "min_test_coverage")
            == 99.99
        )
        assert manager._convert_env_value("0.7", "llm", "temperature") == 0.7
        assert manager._convert_env_value("1.0", "llm", "temperature") == 1.0

    def test_convert_env_value_string(self):
        """Test environment value conversion for strings."""
        manager = ConfigManager("/dummy/path.toml")

        assert manager._convert_env_value("test", "logging", "level") == "test"
        assert (
            manager._convert_env_value("path/to/file", "logging", "file_path")
            == "path/to/file"
        )

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
        {"DIVERSIFIER_DEBUG": "true", "TEST_API_KEY": "test-key"},
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
                assert config.debug_mode is True
            finally:
                os.unlink(f.name)
