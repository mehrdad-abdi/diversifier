"""Custom exceptions for Diversifier tool."""

from typing import Optional, Dict, Any, List
from src.orchestration.error_handling import ErrorCategory, ErrorSeverity


class DiversifierError(Exception):
    """Base exception for all Diversifier errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        """Initialize Diversifier error.

        Args:
            message: Error message
            category: Error category
            severity: Error severity level
            details: Additional error details
            context: Error context information
            recovery_suggestions: Suggestions for error recovery
        """
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.details = details
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []


class MCPServerError(DiversifierError):
    """Exception for MCP server related errors."""

    def __init__(
        self,
        message: str,
        server_type: Optional[str] = None,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        """Initialize MCP server error.

        Args:
            message: Error message
            server_type: Type of MCP server that failed
            details: Additional error details
            context: Error context information
            recovery_suggestions: Suggestions for error recovery
        """
        context = context or {}
        if server_type:
            context["server_type"] = server_type

        super().__init__(
            message=message,
            category=ErrorCategory.MCP_CONNECTION,
            severity=ErrorSeverity.HIGH,
            details=details,
            context=context,
            recovery_suggestions=recovery_suggestions
            or [
                "Retry server connection after brief delay",
                "Check if MCP server process is running",
                "Restart MCP server process",
                "Fall back to direct file operations if server unavailable",
            ],
        )


class AgentExecutionError(DiversifierError):
    """Exception for agent execution errors."""

    def __init__(
        self,
        message: str,
        agent_type: Optional[str] = None,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        """Initialize agent execution error.

        Args:
            message: Error message
            agent_type: Type of agent that failed
            details: Additional error details
            context: Error context information
            recovery_suggestions: Suggestions for error recovery
        """
        context = context or {}
        if agent_type:
            context["agent_type"] = agent_type

        super().__init__(
            message=message,
            category=ErrorCategory.AGENT_EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            context=context,
            recovery_suggestions=recovery_suggestions
            or [
                "Clear agent memory and retry",
                "Simplify task prompt",
                "Use alternative agent approach",
                "Check API key configuration",
                "Retry with exponential backoff",
            ],
        )


class FileOperationError(DiversifierError):
    """Exception for file operation errors."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        """Initialize file operation error.

        Args:
            message: Error message
            file_path: Path to file that caused error
            operation: File operation that failed
            details: Additional error details
            context: Error context information
            recovery_suggestions: Suggestions for error recovery
        """
        context = context or {}
        if file_path:
            context["file_path"] = file_path
        if operation:
            context["operation"] = operation

        super().__init__(
            message=message,
            category=ErrorCategory.FILE_OPERATION,
            severity=ErrorSeverity.LOW,
            details=details,
            context=context,
            recovery_suggestions=recovery_suggestions
            or [
                "Check file permissions",
                "Verify file path exists",
                "Use temporary file for operations",
                "Verify disk space availability",
            ],
        )


class MigrationError(DiversifierError):
    """Exception for migration logic errors."""

    def __init__(
        self,
        message: str,
        source_lib: Optional[str] = None,
        target_lib: Optional[str] = None,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        """Initialize migration error.

        Args:
            message: Error message
            source_lib: Source library being migrated from
            target_lib: Target library being migrated to
            details: Additional error details
            context: Error context information
            recovery_suggestions: Suggestions for error recovery
        """
        context = context or {}
        if source_lib:
            context["source_lib"] = source_lib
        if target_lib:
            context["target_lib"] = target_lib

        super().__init__(
            message=message,
            category=ErrorCategory.MIGRATION_LOGIC,
            severity=ErrorSeverity.HIGH,
            details=details,
            context=context,
            recovery_suggestions=recovery_suggestions
            or [
                "Analyze failed migration patterns",
                "Apply manual fixes for edge cases",
                "Use conservative migration approach",
                "Skip problematic code sections with manual review",
                "Generate additional test cases for validation",
            ],
        )


class ValidationError(DiversifierError):
    """Exception for validation failures."""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            validation_type: Type of validation that failed
            details: Additional error details
            context: Error context information
            recovery_suggestions: Suggestions for error recovery
        """
        context = context or {}
        if validation_type:
            context["validation_type"] = validation_type

        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION_FAILURE,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            context=context,
            recovery_suggestions=recovery_suggestions
            or [
                "Analyze test failure patterns",
                "Apply targeted repairs to failing code",
                "Generate additional test coverage",
                "Review API usage differences",
                "Use incremental migration approach",
            ],
        )


class ConfigurationError(DiversifierError):
    """Exception for configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        """Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that caused error
            details: Additional error details
            context: Error context information
            recovery_suggestions: Suggestions for error recovery
        """
        context = context or {}
        if config_key:
            context["config_key"] = config_key

        # Determine severity based on config type
        severity = ErrorSeverity.CRITICAL
        if config_key and "api_key" not in config_key.lower():
            severity = ErrorSeverity.HIGH

        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=severity,
            details=details,
            context=context,
            recovery_suggestions=recovery_suggestions
            or [
                "Review configuration file syntax",
                "Check required parameters",
                "Use default configuration values",
                "Set missing environment variables",
            ],
        )


class SystemResourceError(DiversifierError):
    """Exception for system resource errors."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ):
        """Initialize system resource error.

        Args:
            message: Error message
            resource_type: Type of system resource (memory, disk, etc.)
            details: Additional error details
            context: Error context information
            recovery_suggestions: Suggestions for error recovery
        """
        context = context or {}
        if resource_type:
            context["resource_type"] = resource_type

        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM_RESOURCE,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            context=context,
            recovery_suggestions=recovery_suggestions
            or [
                "Clear temporary files",
                "Check available system resources",
                "Process data in smaller batches",
                "Restart application if resource leak suspected",
            ],
        )
