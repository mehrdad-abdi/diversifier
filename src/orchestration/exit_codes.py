"""Exit codes for Diversifier tool."""

from enum import IntEnum
from typing import Dict, Any, Optional
from src.orchestration.error_handling import ErrorCategory, ErrorSeverity


class ExitCode(IntEnum):
    """Exit codes for different failure scenarios."""

    # Success
    SUCCESS = 0

    # General errors (1-10)
    GENERAL_ERROR = 1
    INVALID_ARGUMENTS = 2
    KEYBOARD_INTERRUPT = 3

    # Configuration errors (11-20)
    CONFIGURATION_ERROR = 11
    MISSING_API_KEY = 12
    INVALID_CONFIG_FILE = 13

    # File operation errors (21-30)
    FILE_NOT_FOUND = 21
    PERMISSION_DENIED = 22
    DISK_SPACE_ERROR = 23
    FILE_OPERATION_ERROR = 24

    # MCP server errors (31-40)
    MCP_SERVER_UNAVAILABLE = 31
    MCP_CONNECTION_FAILED = 32
    MCP_SERVER_ERROR = 33

    # Agent execution errors (41-50)
    AGENT_EXECUTION_FAILED = 41
    API_RATE_LIMIT_EXCEEDED = 42
    API_ERROR = 43

    # Migration errors (51-60)
    MIGRATION_FAILED = 51
    SOURCE_LIBRARY_NOT_FOUND = 52
    TARGET_LIBRARY_NOT_FOUND = 53
    INCOMPATIBLE_LIBRARIES = 54

    # Validation errors (61-70)
    VALIDATION_FAILED = 61
    TEST_FAILURES = 62
    SYNTAX_ERROR = 63

    # System resource errors (71-80)
    OUT_OF_MEMORY = 71
    SYSTEM_RESOURCE_ERROR = 72

    # Timeout errors (81-90)
    OPERATION_TIMEOUT = 81
    CONNECTION_TIMEOUT = 82


class ExitCodeManager:
    """Manager for exit codes and termination logic."""

    def __init__(self):
        """Initialize exit code manager."""
        self._error_to_exit_code: Dict[ErrorCategory, ExitCode] = {
            ErrorCategory.CONFIGURATION: ExitCode.CONFIGURATION_ERROR,
            ErrorCategory.FILE_OPERATION: ExitCode.FILE_OPERATION_ERROR,
            ErrorCategory.MCP_CONNECTION: ExitCode.MCP_CONNECTION_FAILED,
            ErrorCategory.AGENT_EXECUTION: ExitCode.AGENT_EXECUTION_FAILED,
            ErrorCategory.MIGRATION_LOGIC: ExitCode.MIGRATION_FAILED,
            ErrorCategory.VALIDATION_FAILURE: ExitCode.VALIDATION_FAILED,
            ErrorCategory.SYSTEM_RESOURCE: ExitCode.SYSTEM_RESOURCE_ERROR,
        }

        self._severity_modifiers: Dict[ErrorSeverity, int] = {
            ErrorSeverity.LOW: 0,
            ErrorSeverity.MEDIUM: 0,
            ErrorSeverity.HIGH: 0,
            ErrorSeverity.CRITICAL: 0,  # Critical errors use base code
        }

    def get_exit_code_for_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        error_message: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> ExitCode:
        """Get appropriate exit code for an error.

        Args:
            category: Error category
            severity: Error severity
            error_message: Error message for specific error detection
            context: Error context information

        Returns:
            Appropriate exit code
        """
        context = context or {}

        # Handle specific error patterns
        if category == ErrorCategory.CONFIGURATION:
            if "api key" in error_message.lower():
                return ExitCode.MISSING_API_KEY
            elif "config" in error_message.lower():
                return ExitCode.INVALID_CONFIG_FILE

        elif category == ErrorCategory.FILE_OPERATION:
            if "not found" in error_message.lower():
                return ExitCode.FILE_NOT_FOUND
            elif "permission denied" in error_message.lower():
                return ExitCode.PERMISSION_DENIED
            elif "disk space" in error_message.lower():
                return ExitCode.DISK_SPACE_ERROR

        elif category == ErrorCategory.MCP_CONNECTION:
            if "unavailable" in error_message.lower():
                return ExitCode.MCP_SERVER_UNAVAILABLE
            elif "timeout" in error_message.lower():
                return ExitCode.CONNECTION_TIMEOUT

        elif category == ErrorCategory.AGENT_EXECUTION:
            if "rate limit" in error_message.lower():
                return ExitCode.API_RATE_LIMIT_EXCEEDED
            elif "api" in error_message.lower():
                return ExitCode.API_ERROR

        elif category == ErrorCategory.MIGRATION_LOGIC:
            source_lib = context.get("source_lib")
            target_lib = context.get("target_lib")
            if source_lib and "not found" in error_message.lower():
                return ExitCode.SOURCE_LIBRARY_NOT_FOUND
            elif target_lib and "not found" in error_message.lower():
                return ExitCode.TARGET_LIBRARY_NOT_FOUND
            elif "incompatible" in error_message.lower():
                return ExitCode.INCOMPATIBLE_LIBRARIES

        elif category == ErrorCategory.VALIDATION_FAILURE:
            if "test" in error_message.lower():
                return ExitCode.TEST_FAILURES
            elif "syntax" in error_message.lower():
                return ExitCode.SYNTAX_ERROR

        elif category == ErrorCategory.SYSTEM_RESOURCE:
            if "memory" in error_message.lower():
                return ExitCode.OUT_OF_MEMORY
            elif "timeout" in error_message.lower():
                return ExitCode.OPERATION_TIMEOUT

        # Default to category-based exit code
        base_code = self._error_to_exit_code.get(category, ExitCode.GENERAL_ERROR)
        modifier = self._severity_modifiers.get(severity, 0)
        return ExitCode(base_code + modifier)

    def get_exit_code_description(self, exit_code: ExitCode) -> str:
        """Get human-readable description for exit code.

        Args:
            exit_code: Exit code

        Returns:
            Description of exit code
        """
        descriptions = {
            ExitCode.SUCCESS: "Operation completed successfully",
            ExitCode.GENERAL_ERROR: "General error occurred",
            ExitCode.INVALID_ARGUMENTS: "Invalid command line arguments",
            ExitCode.KEYBOARD_INTERRUPT: "Operation interrupted by user",
            ExitCode.CONFIGURATION_ERROR: "Configuration error",
            ExitCode.MISSING_API_KEY: "Missing or invalid API key",
            ExitCode.INVALID_CONFIG_FILE: "Invalid configuration file",
            ExitCode.FILE_NOT_FOUND: "Required file not found",
            ExitCode.PERMISSION_DENIED: "Permission denied for file operation",
            ExitCode.DISK_SPACE_ERROR: "Insufficient disk space",
            ExitCode.FILE_OPERATION_ERROR: "File operation failed",
            ExitCode.MCP_SERVER_UNAVAILABLE: "MCP server is unavailable",
            ExitCode.MCP_CONNECTION_FAILED: "Failed to connect to MCP server",
            ExitCode.MCP_SERVER_ERROR: "MCP server error",
            ExitCode.AGENT_EXECUTION_FAILED: "Agent execution failed",
            ExitCode.API_RATE_LIMIT_EXCEEDED: "API rate limit exceeded",
            ExitCode.API_ERROR: "API error occurred",
            ExitCode.MIGRATION_FAILED: "Library migration failed",
            ExitCode.SOURCE_LIBRARY_NOT_FOUND: "Source library not found",
            ExitCode.TARGET_LIBRARY_NOT_FOUND: "Target library not found",
            ExitCode.INCOMPATIBLE_LIBRARIES: "Libraries are incompatible",
            ExitCode.VALIDATION_FAILED: "Migration validation failed",
            ExitCode.TEST_FAILURES: "Test failures detected",
            ExitCode.SYNTAX_ERROR: "Syntax error in generated code",
            ExitCode.OUT_OF_MEMORY: "Out of memory error",
            ExitCode.SYSTEM_RESOURCE_ERROR: "System resource error",
            ExitCode.OPERATION_TIMEOUT: "Operation timed out",
            ExitCode.CONNECTION_TIMEOUT: "Connection timed out",
        }

        return descriptions.get(exit_code, f"Unknown exit code: {exit_code}")

    def should_retry(self, exit_code: ExitCode) -> bool:
        """Determine if operation should be retried based on exit code.

        Args:
            exit_code: Exit code to check

        Returns:
            True if operation should be retried
        """
        # Retryable exit codes (temporary failures)
        retryable_codes = {
            ExitCode.MCP_CONNECTION_FAILED,
            ExitCode.API_RATE_LIMIT_EXCEEDED,
            ExitCode.CONNECTION_TIMEOUT,
            ExitCode.OPERATION_TIMEOUT,
            ExitCode.SYSTEM_RESOURCE_ERROR,
        }

        return exit_code in retryable_codes

    def is_user_error(self, exit_code: ExitCode) -> bool:
        """Determine if exit code represents a user error.

        Args:
            exit_code: Exit code to check

        Returns:
            True if exit code represents user error
        """
        # User error codes (incorrect usage, configuration, etc.)
        user_error_codes = {
            ExitCode.INVALID_ARGUMENTS,
            ExitCode.MISSING_API_KEY,
            ExitCode.INVALID_CONFIG_FILE,
            ExitCode.SOURCE_LIBRARY_NOT_FOUND,
            ExitCode.TARGET_LIBRARY_NOT_FOUND,
            ExitCode.INCOMPATIBLE_LIBRARIES,
        }

        return exit_code in user_error_codes
