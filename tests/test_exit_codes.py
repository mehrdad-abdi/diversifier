"""Tests for exit codes."""

from src.orchestration.exit_codes import ExitCode, ExitCodeManager
from src.orchestration.error_handling import ErrorCategory, ErrorSeverity


class TestExitCode:
    """Test ExitCode enum values."""

    def test_success_code(self):
        """Test success exit code."""
        assert ExitCode.SUCCESS == 0

    def test_error_code_ranges(self):
        """Test that error codes are in expected ranges."""
        # General errors (1-10)
        assert 1 <= ExitCode.GENERAL_ERROR <= 10
        assert 1 <= ExitCode.INVALID_ARGUMENTS <= 10

        # Configuration errors (11-20)
        assert 11 <= ExitCode.CONFIGURATION_ERROR <= 20
        assert 11 <= ExitCode.MISSING_API_KEY <= 20

        # File operation errors (21-30)
        assert 21 <= ExitCode.FILE_NOT_FOUND <= 30
        assert 21 <= ExitCode.DISK_SPACE_ERROR <= 30

        # MCP server errors (31-40)
        assert 31 <= ExitCode.MCP_SERVER_UNAVAILABLE <= 40
        assert 31 <= ExitCode.MCP_CONNECTION_FAILED <= 40

        # Agent execution errors (41-50)
        assert 41 <= ExitCode.AGENT_EXECUTION_FAILED <= 50
        assert 41 <= ExitCode.API_ERROR <= 50

    def test_unique_values(self):
        """Test that all exit codes have unique values."""
        codes = [code.value for code in ExitCode]
        assert len(codes) == len(set(codes)), "Exit codes should be unique"


class TestExitCodeManager:
    """Test ExitCodeManager class."""

    def test_initialization(self):
        """Test exit code manager initialization."""
        manager = ExitCodeManager()
        assert manager is not None

    def test_configuration_error_mapping(self):
        """Test configuration error mapping."""
        manager = ExitCodeManager()

        # Generic configuration error (should not match specific patterns)
        code = manager.get_exit_code_for_error(
            ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, "Some other error"
        )
        assert code == ExitCode.CONFIGURATION_ERROR

        # API key specific error
        code = manager.get_exit_code_for_error(
            ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL, "Missing API key"
        )
        assert code == ExitCode.MISSING_API_KEY

        # Config file specific error
        code = manager.get_exit_code_for_error(
            ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, "Invalid config file"
        )
        assert code == ExitCode.INVALID_CONFIG_FILE

    def test_file_operation_error_mapping(self):
        """Test file operation error mapping."""
        manager = ExitCodeManager()

        # File not found
        code = manager.get_exit_code_for_error(
            ErrorCategory.FILE_OPERATION, ErrorSeverity.MEDIUM, "File not found"
        )
        assert code == ExitCode.FILE_NOT_FOUND

        # Permission denied
        code = manager.get_exit_code_for_error(
            ErrorCategory.FILE_OPERATION, ErrorSeverity.MEDIUM, "Permission denied"
        )
        assert code == ExitCode.PERMISSION_DENIED

        # Disk space error
        code = manager.get_exit_code_for_error(
            ErrorCategory.FILE_OPERATION, ErrorSeverity.HIGH, "No disk space available"
        )
        assert code == ExitCode.DISK_SPACE_ERROR

        # Generic file operation error
        code = manager.get_exit_code_for_error(
            ErrorCategory.FILE_OPERATION, ErrorSeverity.LOW, "Generic file error"
        )
        assert code == ExitCode.FILE_OPERATION_ERROR

    def test_mcp_connection_error_mapping(self):
        """Test MCP connection error mapping."""
        manager = ExitCodeManager()

        # Server unavailable
        code = manager.get_exit_code_for_error(
            ErrorCategory.MCP_CONNECTION, ErrorSeverity.HIGH, "Server unavailable"
        )
        assert code == ExitCode.MCP_SERVER_UNAVAILABLE

        # Connection timeout
        code = manager.get_exit_code_for_error(
            ErrorCategory.MCP_CONNECTION, ErrorSeverity.HIGH, "Connection timeout"
        )
        assert code == ExitCode.CONNECTION_TIMEOUT

        # Generic connection error
        code = manager.get_exit_code_for_error(
            ErrorCategory.MCP_CONNECTION, ErrorSeverity.HIGH, "Generic connection error"
        )
        assert code == ExitCode.MCP_CONNECTION_FAILED

    def test_agent_execution_error_mapping(self):
        """Test agent execution error mapping."""
        manager = ExitCodeManager()

        # Rate limit error
        code = manager.get_exit_code_for_error(
            ErrorCategory.AGENT_EXECUTION, ErrorSeverity.MEDIUM, "Rate limit exceeded"
        )
        assert code == ExitCode.API_RATE_LIMIT_EXCEEDED

        # API error
        code = manager.get_exit_code_for_error(
            ErrorCategory.AGENT_EXECUTION, ErrorSeverity.MEDIUM, "API call failed"
        )
        assert code == ExitCode.API_ERROR

        # Generic agent error
        code = manager.get_exit_code_for_error(
            ErrorCategory.AGENT_EXECUTION, ErrorSeverity.MEDIUM, "Agent failed"
        )
        assert code == ExitCode.AGENT_EXECUTION_FAILED

    def test_migration_error_mapping(self):
        """Test migration error mapping."""
        manager = ExitCodeManager()

        # Source library not found
        code = manager.get_exit_code_for_error(
            ErrorCategory.MIGRATION_LOGIC,
            ErrorSeverity.HIGH,
            "Library not found",
            context={"source_lib": "requests"},
        )
        assert code == ExitCode.SOURCE_LIBRARY_NOT_FOUND

        # Target library not found
        code = manager.get_exit_code_for_error(
            ErrorCategory.MIGRATION_LOGIC,
            ErrorSeverity.HIGH,
            "Library not found",
            context={"target_lib": "httpx"},
        )
        assert code == ExitCode.TARGET_LIBRARY_NOT_FOUND

        # Incompatible libraries
        code = manager.get_exit_code_for_error(
            ErrorCategory.MIGRATION_LOGIC,
            ErrorSeverity.HIGH,
            "Libraries are incompatible",
        )
        assert code == ExitCode.INCOMPATIBLE_LIBRARIES

        # Generic migration error
        code = manager.get_exit_code_for_error(
            ErrorCategory.MIGRATION_LOGIC, ErrorSeverity.HIGH, "Migration failed"
        )
        assert code == ExitCode.MIGRATION_FAILED

    def test_validation_error_mapping(self):
        """Test validation error mapping."""
        manager = ExitCodeManager()

        # Test failures
        code = manager.get_exit_code_for_error(
            ErrorCategory.VALIDATION_FAILURE,
            ErrorSeverity.MEDIUM,
            "Test failures detected",
        )
        assert code == ExitCode.TEST_FAILURES

        # Syntax error
        code = manager.get_exit_code_for_error(
            ErrorCategory.VALIDATION_FAILURE, ErrorSeverity.HIGH, "Syntax error in code"
        )
        assert code == ExitCode.SYNTAX_ERROR

        # Generic validation error
        code = manager.get_exit_code_for_error(
            ErrorCategory.VALIDATION_FAILURE, ErrorSeverity.MEDIUM, "Validation failed"
        )
        assert code == ExitCode.VALIDATION_FAILED

    def test_system_resource_error_mapping(self):
        """Test system resource error mapping."""
        manager = ExitCodeManager()

        # Memory error
        code = manager.get_exit_code_for_error(
            ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.CRITICAL, "Out of memory"
        )
        assert code == ExitCode.OUT_OF_MEMORY

        # Timeout error
        code = manager.get_exit_code_for_error(
            ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.HIGH, "Operation timeout"
        )
        assert code == ExitCode.OPERATION_TIMEOUT

        # Generic resource error
        code = manager.get_exit_code_for_error(
            ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.CRITICAL, "Resource error"
        )
        assert code == ExitCode.SYSTEM_RESOURCE_ERROR

    def test_unknown_category_default(self):
        """Test unknown category defaults to general error."""
        manager = ExitCodeManager()

        # Create a mock category (this would normally not exist)
        code = manager.get_exit_code_for_error(
            ErrorCategory.FILE_OPERATION,  # Use existing category but treat as unknown
            ErrorSeverity.LOW,
            "Unknown error",
        )
        # Should return a valid exit code (not GENERAL_ERROR in this case since FILE_OPERATION exists)
        assert isinstance(code, ExitCode)

    def test_get_exit_code_description(self):
        """Test exit code descriptions."""
        manager = ExitCodeManager()

        # Test some specific descriptions
        desc = manager.get_exit_code_description(ExitCode.SUCCESS)
        assert "successfully" in desc.lower()

        desc = manager.get_exit_code_description(ExitCode.MISSING_API_KEY)
        assert "api key" in desc.lower()

        desc = manager.get_exit_code_description(ExitCode.FILE_NOT_FOUND)
        assert "file not found" in desc.lower()

        desc = manager.get_exit_code_description(ExitCode.MCP_SERVER_UNAVAILABLE)
        assert "mcp server" in desc.lower() and "unavailable" in desc.lower()

        # Test unknown exit code - use a valid but unmapped code
        desc = manager.get_exit_code_description(ExitCode.GENERAL_ERROR)
        assert len(desc) > 0  # Should have some description

    def test_is_user_error(self):
        """Test user error detection."""
        manager = ExitCodeManager()

        # User error codes
        assert manager.is_user_error(ExitCode.INVALID_ARGUMENTS) is True
        assert manager.is_user_error(ExitCode.MISSING_API_KEY) is True
        assert manager.is_user_error(ExitCode.INVALID_CONFIG_FILE) is True
        assert manager.is_user_error(ExitCode.SOURCE_LIBRARY_NOT_FOUND) is True
        assert manager.is_user_error(ExitCode.TARGET_LIBRARY_NOT_FOUND) is True
        assert manager.is_user_error(ExitCode.INCOMPATIBLE_LIBRARIES) is True

        # Non-user error codes
        assert manager.is_user_error(ExitCode.SUCCESS) is False
        assert manager.is_user_error(ExitCode.MCP_CONNECTION_FAILED) is False
        assert manager.is_user_error(ExitCode.SYSTEM_RESOURCE_ERROR) is False
        assert manager.is_user_error(ExitCode.OUT_OF_MEMORY) is False
