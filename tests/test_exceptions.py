"""Tests for custom exceptions."""

from src.orchestration.error_handling import ErrorCategory, ErrorSeverity
from src.orchestration.exceptions import (
    AgentExecutionError,
    ConfigurationError,
    DiversifierError,
    FileOperationError,
    MCPServerError,
    MigrationError,
    SystemResourceError,
    ValidationError,
)


class TestDiversifierError:
    """Test base DiversifierError class."""

    def test_basic_initialization(self):
        """Test basic error initialization."""
        error = DiversifierError(
            message="Test error",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
        )

        assert str(error) == "Test error"
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.details is None
        assert error.context == {}
        assert error.recovery_suggestions == []

    def test_full_initialization(self):
        """Test error initialization with all parameters."""
        context = {"key": "value"}
        suggestions = ["Try again", "Check config"]

        error = DiversifierError(
            message="Test error",
            category=ErrorCategory.MIGRATION_LOGIC,
            severity=ErrorSeverity.CRITICAL,
            details="Additional details",
            context=context,
            recovery_suggestions=suggestions,
        )

        assert str(error) == "Test error"
        assert error.category == ErrorCategory.MIGRATION_LOGIC
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.details == "Additional details"
        assert error.context == context
        assert error.recovery_suggestions == suggestions


class TestMCPServerError:
    """Test MCP server specific errors."""

    def test_basic_initialization(self):
        """Test MCP server error initialization."""
        error = MCPServerError("Connection failed")

        assert str(error) == "Connection failed"
        assert error.category == ErrorCategory.MCP_CONNECTION
        assert error.severity == ErrorSeverity.HIGH
        assert "server_type" not in error.context
        assert len(error.recovery_suggestions) > 0

    def test_with_server_type(self):
        """Test MCP server error with server type."""
        error = MCPServerError("Connection failed", server_type="filesystem")

        assert error.context["server_type"] == "filesystem"

    def test_default_recovery_suggestions(self):
        """Test default recovery suggestions."""
        error = MCPServerError("Connection failed")

        assert "Retry server connection after brief delay" in error.recovery_suggestions
        assert (
            "Fall back to direct file operations if server unavailable"
            in error.recovery_suggestions
        )


class TestAgentExecutionError:
    """Test agent execution specific errors."""

    def test_basic_initialization(self):
        """Test agent execution error initialization."""
        error = AgentExecutionError("Execution failed")

        assert str(error) == "Execution failed"
        assert error.category == ErrorCategory.AGENT_EXECUTION
        assert error.severity == ErrorSeverity.MEDIUM
        assert len(error.recovery_suggestions) > 0

    def test_with_agent_type(self):
        """Test agent error with agent type."""
        error = AgentExecutionError("Execution failed", agent_type="migrator")

        assert error.context["agent_type"] == "migrator"

    def test_default_recovery_suggestions(self):
        """Test default recovery suggestions."""
        error = AgentExecutionError("Execution failed")

        assert "Clear agent memory and retry" in error.recovery_suggestions
        assert "Check API key configuration" in error.recovery_suggestions


class TestFileOperationError:
    """Test file operation specific errors."""

    def test_basic_initialization(self):
        """Test file operation error initialization."""
        error = FileOperationError("File not found")

        assert str(error) == "File not found"
        assert error.category == ErrorCategory.FILE_OPERATION
        assert error.severity == ErrorSeverity.LOW
        assert len(error.recovery_suggestions) > 0

    def test_with_file_context(self):
        """Test file error with file context."""
        error = FileOperationError(
            "Permission denied", file_path="/test/path", operation="write"
        )

        assert error.context["file_path"] == "/test/path"
        assert error.context["operation"] == "write"

    def test_default_recovery_suggestions(self):
        """Test default recovery suggestions."""
        error = FileOperationError("File error")

        assert "Check file permissions" in error.recovery_suggestions
        assert "Verify disk space availability" in error.recovery_suggestions


class TestMigrationError:
    """Test migration specific errors."""

    def test_basic_initialization(self):
        """Test migration error initialization."""
        error = MigrationError("Migration failed")

        assert str(error) == "Migration failed"
        assert error.category == ErrorCategory.MIGRATION_LOGIC
        assert error.severity == ErrorSeverity.HIGH
        assert len(error.recovery_suggestions) > 0

    def test_with_library_context(self):
        """Test migration error with library context."""
        error = MigrationError(
            "Incompatible libraries", source_lib="requests", target_lib="httpx"
        )

        assert error.context["source_lib"] == "requests"
        assert error.context["target_lib"] == "httpx"

    def test_default_recovery_suggestions(self):
        """Test default recovery suggestions."""
        error = MigrationError("Migration failed")

        assert "Analyze failed migration patterns" in error.recovery_suggestions
        assert "Use conservative migration approach" in error.recovery_suggestions


class TestValidationError:
    """Test validation specific errors."""

    def test_basic_initialization(self):
        """Test validation error initialization."""
        error = ValidationError("Validation failed")

        assert str(error) == "Validation failed"
        assert error.category == ErrorCategory.VALIDATION_FAILURE
        assert error.severity == ErrorSeverity.MEDIUM
        assert len(error.recovery_suggestions) > 0

    def test_with_validation_type(self):
        """Test validation error with validation type."""
        error = ValidationError("Test failed", validation_type="unit_test")

        assert error.context["validation_type"] == "unit_test"

    def test_default_recovery_suggestions(self):
        """Test default recovery suggestions."""
        error = ValidationError("Validation failed")

        assert "Analyze test failure patterns" in error.recovery_suggestions
        assert "Generate additional test coverage" in error.recovery_suggestions


class TestConfigurationError:
    """Test configuration specific errors."""

    def test_basic_initialization(self):
        """Test configuration error initialization."""
        error = ConfigurationError("Config error")

        assert str(error) == "Config error"
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.CRITICAL  # Default for config errors

    def test_non_api_key_error(self):
        """Test non-API key configuration error."""
        error = ConfigurationError("Invalid timeout", config_key="timeout")

        assert error.severity == ErrorSeverity.HIGH  # Not critical for non-API keys

    def test_api_key_error(self):
        """Test API key configuration error."""
        error = ConfigurationError("Missing API key", config_key="openai_api_key")

        assert error.severity == ErrorSeverity.CRITICAL

    def test_default_recovery_suggestions(self):
        """Test default recovery suggestions."""
        error = ConfigurationError("Config error")

        assert "Review configuration file syntax" in error.recovery_suggestions
        assert "Set missing environment variables" in error.recovery_suggestions


class TestSystemResourceError:
    """Test system resource specific errors."""

    def test_basic_initialization(self):
        """Test system resource error initialization."""
        error = SystemResourceError("Resource error")

        assert str(error) == "Resource error"
        assert error.category == ErrorCategory.SYSTEM_RESOURCE
        assert error.severity == ErrorSeverity.CRITICAL
        assert len(error.recovery_suggestions) > 0

    def test_with_resource_type(self):
        """Test system resource error with resource type."""
        error = SystemResourceError("Out of memory", resource_type="memory")

        assert error.context["resource_type"] == "memory"

    def test_default_recovery_suggestions(self):
        """Test default recovery suggestions."""
        error = SystemResourceError("Resource error")

        assert "Clear temporary files" in error.recovery_suggestions
        assert "Process data in smaller batches" in error.recovery_suggestions
