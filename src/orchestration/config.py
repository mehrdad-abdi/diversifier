"""Configuration management for Diversifier."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str = "INFO"
    console: bool = True
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format_string: str = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    enable_correlation_ids: bool = True


@dataclass
class MCPConfig:
    """MCP server configuration settings."""

    filesystem_server_path: str = "src/mcp_servers/filesystem/server.py"
    testing_server_path: str = "src/mcp_servers/testing/server.py"
    git_server_path: str = "src/mcp_servers/git/server.py"
    docker_server_path: str = "src/mcp_servers/docker/server.py"
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class MigrationConfig:
    """Migration workflow configuration settings."""

    max_iterations: int = 5
    test_timeout: int = 300  # 5 minutes
    backup_original: bool = True
    validate_syntax: bool = True
    require_test_coverage: bool = True
    min_test_coverage: float = 0.8
    allowed_library_pairs: List[tuple] = field(
        default_factory=lambda: [
            ("requests", "httpx"),
            ("urllib", "requests"),
            ("flask", "fastapi"),
            ("sqlite3", "sqlalchemy"),
        ]
    )


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""

    enable_metrics: bool = True
    metrics_file: Optional[str] = "performance_metrics.json"
    log_slow_operations: bool = True
    slow_operation_threshold: float = 1.0  # seconds
    enable_memory_tracking: bool = False


@dataclass
class TaskTemperatureConfig:
    """Task-specific temperature configuration."""

    analyzer: float = 0.1  # Lower for precise analysis
    migrator: float = 0.2  # Slightly higher for code transformation
    tester: float = 0.3  # Higher for creative test generation
    repairer: float = 0.2  # Moderate for problem solving
    doc_analyzer: float = 0.1  # Lower for precise documentation analysis
    source_code_analyzer: float = 0.1  # Lower for precise code analysis
    acceptance_test_generator: float = 0.2  # Moderate for test generation


@dataclass
class LLMConfig:
    """LLM (Large Language Model) configuration settings."""

    provider: str  # anthropic, openai, google_genai, etc. (REQUIRED)
    model_name: str  # Model name (REQUIRED)
    api_key_env_var: str  # Environment variable name for API key (REQUIRED)
    temperature: float = 0.1  # Default temperature
    max_tokens: int = 4096
    timeout: int = 120  # seconds
    retry_attempts: int = 3
    base_url: Optional[str] = None  # Custom API endpoint URL
    task_temperatures: TaskTemperatureConfig = field(
        default_factory=TaskTemperatureConfig
    )
    additional_params: Dict[str, Union[str, int, float, bool]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """Validate LLM configuration after initialization."""
        # Validate that API key environment variable is set
        import os

        api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key environment variable '{self.api_key_env_var}' is not set. "
                f"Please set: export {self.api_key_env_var}=your_api_key_here"
            )


@dataclass
class DiversifierConfig:
    """Complete configuration for the Diversifier tool."""

    llm: LLMConfig  # LLM configuration (REQUIRED)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    project_root: str = "."
    temp_dir: str = "/tmp/diversifier"
    debug_mode: bool = False


class ConfigManager:
    """Manages configuration loading and environment variable overrides."""

    def __init__(self, config_path: Union[str, Path]):
        """Initialize configuration manager.

        Args:
            config_path: Path to TOML configuration file (REQUIRED)
        """
        self.config_path = Path(config_path)
        self._config: Optional[DiversifierConfig] = None

    def load_config(self) -> DiversifierConfig:
        """Load configuration from file and environment variables.

        Returns:
            Complete configuration object
        """
        if self._config is not None:
            return self._config

        # Require config file to exist
        if not self.config_path.exists():
            raise ValueError(
                f"Configuration file not found: {self.config_path}\n"
                f"Create a config file using: diversifier --create-config {self.config_path}"
            )

        # Load from TOML file
        config_data = self._load_toml_config()

        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)

        # Create configuration object
        self._config = self._create_config_from_dict(config_data)

        return self._config

    def _load_toml_config(self) -> Dict:
        """Load configuration from TOML file.

        Returns:
            Configuration dictionary
        """
        try:
            if self.config_path is None:
                raise ValueError("Config path is None")
            with open(self.config_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config file {self.config_path}: {e}")

    def _apply_env_overrides(self, config_data: Dict) -> Dict:
        """Apply environment variable overrides to configuration.

        Args:
            config_data: Base configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        env_mappings = {
            # Logging configuration
            "DIVERSIFIER_LOG_LEVEL": ("logging", "level"),
            "DIVERSIFIER_LOG_FILE": ("logging", "file_path"),
            "DIVERSIFIER_LOG_CONSOLE": ("logging", "console"),
            "DIVERSIFIER_LOG_MAX_SIZE": ("logging", "max_file_size"),
            "DIVERSIFIER_LOG_BACKUP_COUNT": ("logging", "backup_count"),
            "DIVERSIFIER_CORRELATION_IDS": ("logging", "enable_correlation_ids"),
            # MCP configuration
            "DIVERSIFIER_MCP_TIMEOUT": ("mcp", "timeout"),
            "DIVERSIFIER_MCP_RETRIES": ("mcp", "retry_attempts"),
            # Migration configuration
            "DIVERSIFIER_MAX_ITERATIONS": ("migration", "max_iterations"),
            "DIVERSIFIER_TEST_TIMEOUT": ("migration", "test_timeout"),
            "DIVERSIFIER_BACKUP_ORIGINAL": ("migration", "backup_original"),
            "DIVERSIFIER_MIN_COVERAGE": ("migration", "min_test_coverage"),
            # Performance configuration
            "DIVERSIFIER_ENABLE_METRICS": ("performance", "enable_metrics"),
            "DIVERSIFIER_METRICS_FILE": ("performance", "metrics_file"),
            "DIVERSIFIER_SLOW_THRESHOLD": ("performance", "slow_operation_threshold"),
            # LLM configuration
            "DIVERSIFIER_LLM_PROVIDER": ("llm", "provider"),
            "DIVERSIFIER_LLM_MODEL_NAME": ("llm", "model_name"),
            "DIVERSIFIER_LLM_TEMPERATURE": ("llm", "temperature"),
            "DIVERSIFIER_LLM_MAX_TOKENS": ("llm", "max_tokens"),
            "DIVERSIFIER_LLM_TIMEOUT": ("llm", "timeout"),
            "DIVERSIFIER_LLM_RETRY_ATTEMPTS": ("llm", "retry_attempts"),
            "DIVERSIFIER_LLM_API_KEY_ENV_VAR": ("llm", "api_key_env_var"),
            "DIVERSIFIER_LLM_BASE_URL": ("llm", "base_url"),
            # General configuration
            "DIVERSIFIER_PROJECT_ROOT": (None, "project_root"),
            "DIVERSIFIER_TEMP_DIR": (None, "temp_dir"),
            "DIVERSIFIER_DEBUG": (None, "debug_mode"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(value, section, key)

                if section is None:
                    config_data[key] = converted_value
                else:
                    if section not in config_data:
                        config_data[section] = {}
                    config_data[section][key] = converted_value

        return config_data

    def _convert_env_value(
        self, value: str, section: Optional[str], key: str
    ) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type.

        Args:
            value: Environment variable value
            section: Configuration section
            key: Configuration key

        Returns:
            Converted value
        """
        # Boolean values
        if key in [
            "console",
            "backup_original",
            "validate_syntax",
            "require_test_coverage",
            "enable_metrics",
            "log_slow_operations",
            "enable_memory_tracking",
            "enable_correlation_ids",
            "debug_mode",
        ]:
            return value.lower() in ("true", "1", "yes", "on")

        # Integer values
        if key in [
            "max_file_size",
            "backup_count",
            "timeout",
            "retry_attempts",
            "max_iterations",
            "test_timeout",
            "max_tokens",
        ]:
            return int(value)

        # Float values
        if key in ["min_test_coverage", "slow_operation_threshold", "temperature"]:
            return float(value)

        # String values (default)
        return value

    def _create_config_from_dict(self, config_data: Dict) -> DiversifierConfig:
        """Create configuration object from dictionary.

        Args:
            config_data: Configuration dictionary

        Returns:
            Configuration object
        """
        # Extract nested configurations
        logging_data = config_data.get("logging", {})
        mcp_data = config_data.get("mcp", {})
        migration_data = config_data.get("migration", {})
        performance_data = config_data.get("performance", {})
        llm_data = config_data.get("llm", {})

        # Create nested config objects - filter out unknown keys
        logging_config = LoggingConfig(
            **{
                k: v
                for k, v in logging_data.items()
                if k in LoggingConfig.__dataclass_fields__
            }
        )
        mcp_config = MCPConfig(
            **{k: v for k, v in mcp_data.items() if k in MCPConfig.__dataclass_fields__}
        )
        migration_config = MigrationConfig(
            **{
                k: v
                for k, v in migration_data.items()
                if k in MigrationConfig.__dataclass_fields__
            }
        )
        performance_config = PerformanceConfig(
            **{
                k: v
                for k, v in performance_data.items()
                if k in PerformanceConfig.__dataclass_fields__
            }
        )

        # Handle task temperatures separately
        task_temps_data = llm_data.get("task_temperatures", {})
        task_temperatures = TaskTemperatureConfig(
            **{
                k: v
                for k, v in task_temps_data.items()
                if k in TaskTemperatureConfig.__dataclass_fields__
            }
        )

        # Create LLM config with task temperatures
        llm_config_data = {
            k: v
            for k, v in llm_data.items()
            if k != "task_temperatures" and k in LLMConfig.__dataclass_fields__
        }
        llm_config = LLMConfig(task_temperatures=task_temperatures, **llm_config_data)

        # Extract top-level configuration
        top_level_data = {
            k: v
            for k, v in config_data.items()
            if k not in ["logging", "mcp", "migration", "performance", "llm"]
        }

        return DiversifierConfig(
            logging=logging_config,
            mcp=mcp_config,
            migration=migration_config,
            performance=performance_config,
            llm=llm_config,
            **top_level_data,
        )

    def get_config(self) -> DiversifierConfig:
        """Get current configuration, loading if necessary.

        Returns:
            Configuration object
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def reload_config(self) -> DiversifierConfig:
        """Reload configuration from file and environment.

        Returns:
            Reloaded configuration object
        """
        self._config = None
        return self.load_config()

    def save_config_template(self, output_path: Union[str, Path]) -> None:
        """Save a template configuration file.

        Args:
            output_path: Path to save template file
        """
        template_content = """# Diversifier Configuration Template

[logging]
level = "INFO"
console = true
file_path = "diversifier.log"
max_file_size = 10485760  # 10MB
backup_count = 5
enable_correlation_ids = true

[mcp]
filesystem_server_path = "src/mcp_servers/filesystem/server.py"
testing_server_path = "src/mcp_servers/testing/server.py"
git_server_path = "src/mcp_servers/git/server.py"
docker_server_path = "src/mcp_servers/docker/server.py"
timeout = 30
retry_attempts = 3

[migration]
max_iterations = 5
test_timeout = 300
backup_original = true
validate_syntax = true
require_test_coverage = true
min_test_coverage = 0.8

[performance]
enable_metrics = true
metrics_file = "performance_metrics.json"
log_slow_operations = true
slow_operation_threshold = 1.0
enable_memory_tracking = false

[llm]
# LLM Provider: Use correct LangChain provider names  
# See https://python.langchain.com/docs/integrations/chat/ for all supported providers
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
api_key_env_var = "ANTHROPIC_API_KEY"  # REQUIRED: Environment variable name for API key
temperature = 0.1  # Default temperature for all tasks
max_tokens = 4096
timeout = 120
retry_attempts = 3
# Optional: Custom API endpoint URL
# base_url = "https://api.anthropic.com"

# Task-specific temperatures (override default temperature for specific tasks)
[llm.task_temperatures]
analyzer = 0.1  # Lower for precise analysis
migrator = 0.2  # Slightly higher for code transformation
tester = 0.3  # Higher for creative test generation
repairer = 0.2  # Moderate for problem solving
doc_analyzer = 0.1  # Lower for precise documentation analysis
source_code_analyzer = 0.1  # Lower for precise code analysis
acceptance_test_generator = 0.2  # Moderate for test generation

# Example configurations for different providers:
# For OpenAI:
# provider = "openai"
# model_name = "gpt-4"
# api_key_env_var = "OPENAI_API_KEY"

# For Google Gemini (use google_genai for LangChain):
# provider = "google_genai"
# model_name = "gemini-pro"
# api_key_env_var = "GOOGLE_API_KEY"

# For Azure OpenAI:
# provider = "azure_openai"
# model_name = "gpt-4"

# General settings
project_root = "."
temp_dir = "/tmp/diversifier"
debug_mode = false
"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(template_content)


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Union[str, Path]) -> ConfigManager:
    """Get global configuration manager instance.

    Args:
        config_path: Path to configuration file (REQUIRED)

    Returns:
        Configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(config_path: Optional[Union[str, Path]] = None) -> DiversifierConfig:
    """Get current configuration.

    Args:
        config_path: Path to configuration file. If None, uses global config manager.

    Returns:
        Configuration object
        
    Raises:
        ValueError: If no config path provided and no global config manager initialized
    """
    if config_path is not None:
        return get_config_manager(config_path).get_config()
    else:
        # Use global config manager if already initialized
        global _config_manager
        if _config_manager is None:
            raise ValueError(
                "No configuration available. Either provide config_path or initialize "
                "global config first by calling get_config_manager() with a path."
            )
        return _config_manager.get_config()


def get_task_temperature(
    task_name: str, llm_config: Optional[LLMConfig] = None
) -> float:
    """Get temperature for a specific task.

    Args:
        task_name: Name of the task (analyzer, migrator, etc.)
        llm_config: Optional LLM config. If None, uses global config.

    Returns:
        Temperature value for the task
    """
    if llm_config is None:
        llm_config = get_config().llm

    # Try to get task-specific temperature
    task_temp = getattr(llm_config.task_temperatures, task_name, None)
    if task_temp is not None:
        return task_temp

    # Fall back to default temperature
    return llm_config.temperature
