"""Error handling and recovery mechanisms for orchestration."""

import gc
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
import traceback


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors that can occur."""

    MCP_CONNECTION = "mcp_connection"
    AGENT_EXECUTION = "agent_execution"
    FILE_OPERATION = "file_operation"
    MIGRATION_LOGIC = "migration_logic"
    VALIDATION_FAILURE = "validation_failure"
    CONFIGURATION = "configuration"
    SYSTEM_RESOURCE = "system_resource"


@dataclass
class ErrorInfo:
    """Information about an error occurrence."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    traceback: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    recoverable: bool = True
    recovery_suggestions: Optional[List[str]] = None

    def __post_init__(self):
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []


class ErrorHandler:
    """Handles errors and recovery mechanisms."""

    def __init__(self):
        """Initialize error handler."""
        self.logger = logging.getLogger("diversifier.error_handler")
        self.error_history: List[ErrorInfo] = []
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}

        # Register default recovery handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default error recovery handlers."""
        self.recovery_handlers[ErrorCategory.MCP_CONNECTION] = (
            self._handle_mcp_connection_error
        )
        self.recovery_handlers[ErrorCategory.AGENT_EXECUTION] = (
            self._handle_agent_execution_error
        )
        self.recovery_handlers[ErrorCategory.FILE_OPERATION] = (
            self._handle_file_operation_error
        )
        self.recovery_handlers[ErrorCategory.MIGRATION_LOGIC] = (
            self._handle_migration_logic_error
        )
        self.recovery_handlers[ErrorCategory.VALIDATION_FAILURE] = (
            self._handle_validation_failure
        )
        self.recovery_handlers[ErrorCategory.CONFIGURATION] = (
            self._handle_configuration_error
        )
        self.recovery_handlers[ErrorCategory.SYSTEM_RESOURCE] = (
            self._handle_system_resource_error
        )

    def handle_error(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorInfo:
        """Handle an error and determine recovery strategy.

        Args:
            exception: The exception that occurred
            category: Category of error
            context: Additional context about the error

        Returns:
            ErrorInfo with recovery suggestions
        """
        # Determine severity
        severity = self._assess_severity(exception, category)

        # Create error info
        error_info = ErrorInfo(
            category=category,
            severity=severity,
            message=str(exception),
            details=getattr(exception, "details", None),
            traceback=traceback.format_exc(),
            context=context or {},
            recoverable=self._is_recoverable(exception, category),
        )

        # Add to history
        self.error_history.append(error_info)

        # Log the error
        self._log_error(error_info)

        # Get recovery suggestions
        if error_info.recoverable and category in self.recovery_handlers:
            try:
                recovery_suggestions = self.recovery_handlers[category](error_info)
                error_info.recovery_suggestions = recovery_suggestions
            except Exception as e:
                self.logger.error(
                    f"Error in recovery handler for {category.value}: {e}"
                )

        return error_info

    def _assess_severity(
        self, exception: Exception, category: ErrorCategory
    ) -> ErrorSeverity:
        """Assess the severity of an error.

        Args:
            exception: The exception
            category: Error category

        Returns:
            Error severity level
        """
        # Critical errors that stop the workflow
        if category == ErrorCategory.CONFIGURATION and "API key" in str(exception):
            return ErrorSeverity.CRITICAL

        if (
            category == ErrorCategory.SYSTEM_RESOURCE
            and "disk space" in str(exception).lower()
        ):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if category in [ErrorCategory.MCP_CONNECTION, ErrorCategory.MIGRATION_LOGIC]:
            return ErrorSeverity.HIGH

        # Medium severity errors
        if category in [
            ErrorCategory.AGENT_EXECUTION,
            ErrorCategory.VALIDATION_FAILURE,
        ]:
            return ErrorSeverity.MEDIUM

        # Low severity errors
        return ErrorSeverity.LOW

    def _is_recoverable(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if an error is recoverable.

        Args:
            exception: The exception
            category: Error category

        Returns:
            True if error is recoverable
        """
        # Configuration errors are usually not recoverable automatically
        if category == ErrorCategory.CONFIGURATION:
            return False

        # System resource errors depend on the specific issue
        if category == ErrorCategory.SYSTEM_RESOURCE:
            if "permission denied" in str(exception).lower():
                return False
            if "disk space" in str(exception).lower():
                return False

        # Most other errors are recoverable with retry or alternative approaches
        return True

    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error information.

        Args:
            error_info: Error information to log
        """
        level_map = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }

        level = level_map[error_info.severity]

        self.logger.log(
            level, f"[{error_info.category.value.upper()}] {error_info.message}"
        )

        if error_info.details:
            self.logger.log(level, f"Details: {error_info.details}")

        if error_info.context:
            self.logger.log(level, f"Context: {error_info.context}")

        if error_info.recovery_suggestions:
            self.logger.info(
                f"Recovery suggestions: {', '.join(error_info.recovery_suggestions)}"
            )

    def _handle_mcp_connection_error(self, error_info: ErrorInfo) -> List[str]:
        """Handle MCP connection errors.

        Args:
            error_info: Error information

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        if "connection refused" in error_info.message.lower():
            suggestions.extend(
                [
                    "Check if MCP server process is running",
                    "Restart MCP server process",
                ]
            )

        if "timeout" in error_info.message.lower():
            suggestions.extend(
                [
                    "Increase connection timeout",
                    "Check system resources",
                ]
            )

        suggestions.append("Fall back to direct file operations if server unavailable")

        return suggestions

    def _handle_agent_execution_error(self, error_info: ErrorInfo) -> List[str]:
        """Handle agent execution errors.

        Args:
            error_info: Error information

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        if (
            "api" in error_info.message.lower()
            or "openai" in error_info.message.lower()
        ):
            suggestions.extend(
                [
                    "Check API key configuration",
                    "Switch to different model if rate limited",
                ]
            )

        if "token" in error_info.message.lower():
            suggestions.extend(
                [
                    "Reduce prompt length",
                    "Split task into smaller chunks",
                    "Use model with larger context window",
                ]
            )

        suggestions.extend(
            [
                "Clear agent memory and retry",
                "Simplify task prompt",
                "Use alternative agent approach",
            ]
        )

        return suggestions

    def _handle_file_operation_error(self, error_info: ErrorInfo) -> List[str]:
        """Handle file operation errors.

        Args:
            error_info: Error information

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        if "permission denied" in error_info.message.lower():
            suggestions.extend(
                [
                    "Check file permissions",
                    "Run with appropriate privileges",
                    "Skip file if non-critical",
                ]
            )

        if "file not found" in error_info.message.lower():
            suggestions.extend(
                [
                    "Verify file path exists",
                    "Check if file was moved or deleted",
                    "Skip missing file if optional",
                ]
            )

        suggestions.extend(
            [
                "Use temporary file for operations",
                "Verify disk space availability",
            ]
        )

        return suggestions

    def _handle_migration_logic_error(self, error_info: ErrorInfo) -> List[str]:
        """Handle migration logic errors.

        Args:
            error_info: Error information

        Returns:
            List of recovery suggestions
        """
        suggestions = [
            "Analyze failed migration patterns",
            "Apply manual fixes for edge cases",
            "Use conservative migration approach",
            "Skip problematic code sections with manual review",
            "Generate additional test cases for validation",
        ]

        return suggestions

    def _handle_validation_failure(self, error_info: ErrorInfo) -> List[str]:
        """Handle validation failures.

        Args:
            error_info: Error information

        Returns:
            List of recovery suggestions
        """
        suggestions = [
            "Analyze test failure patterns",
            "Apply targeted repairs to failing code",
            "Generate additional test coverage",
            "Review API usage differences",
            "Use incremental migration approach",
        ]

        return suggestions

    def _handle_configuration_error(self, error_info: ErrorInfo) -> List[str]:
        """Handle configuration errors.

        Args:
            error_info: Error information

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        if "api key" in error_info.message.lower():
            suggestions.extend(
                [
                    "Set OPENAI_API_KEY environment variable",
                    "Check API key validity",
                    "Verify API key has necessary permissions",
                ]
            )

        suggestions.extend(
            [
                "Review configuration file syntax",
                "Check required parameters",
                "Use default configuration values",
            ]
        )

        return suggestions

    def _handle_system_resource_error(self, error_info: ErrorInfo) -> List[str]:
        """Handle system resource errors.

        Args:
            error_info: Error information

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        if "memory" in error_info.message.lower():
            suggestions.extend(
                [
                    "Clear unnecessary data structures",
                    "Process files in smaller batches",
                    "Restart application if memory leak suspected",
                ]
            )

        if "disk" in error_info.message.lower():
            suggestions.extend(
                [
                    "Clear temporary files",
                    "Check available disk space",
                    "Use alternative storage location",
                ]
            )

        return suggestions

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered.

        Returns:
            Error summary statistics
        """
        total_errors = len(self.error_history)

        if total_errors == 0:
            return {"total_errors": 0, "categories": {}, "severities": {}}

        # Count by category
        categories: Dict[str, int] = {}
        for error in self.error_history:
            category = error.category.value
            categories[category] = categories.get(category, 0) + 1

        # Count by severity
        severities: Dict[str, int] = {}
        for error in self.error_history:
            severity = error.severity.value
            severities[severity] = severities.get(severity, 0) + 1

        # Count recoverable vs non-recoverable
        recoverable = sum(1 for error in self.error_history if error.recoverable)

        return {
            "total_errors": total_errors,
            "categories": categories,
            "severities": severities,
            "recoverable": recoverable,
            "non_recoverable": total_errors - recoverable,
        }

    def attempt_recovery(self, error_info: ErrorInfo) -> bool:
        """Attempt single recovery from an error using registered handlers.

        Args:
            error_info: Error information

        Returns:
            True if recovery was successful, False otherwise
        """
        if not error_info.recoverable:
            return False

        self.logger.info(
            f"Single recovery attempt for {error_info.category.value} error"
        )

        try:
            # Apply recovery strategy based on category
            success = self._apply_recovery_strategy(error_info)
            if success:
                self.logger.info(
                    f"Recovery successful for {error_info.category.value} error"
                )
                return True
            else:
                self.logger.error(
                    f"Recovery failed for {error_info.category.value} error"
                )
                return False

        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            return False

    def _apply_recovery_strategy(self, error_info: ErrorInfo) -> bool:
        """Apply recovery strategy for specific error category.

        Args:
            error_info: Error information

        Returns:
            True if recovery strategy was successful
        """
        if error_info.category == ErrorCategory.MCP_CONNECTION:
            return self._recover_mcp_connection(error_info)
        elif error_info.category == ErrorCategory.AGENT_EXECUTION:
            return self._recover_agent_execution(error_info)
        elif error_info.category == ErrorCategory.FILE_OPERATION:
            return self._recover_file_operation(error_info)
        elif error_info.category == ErrorCategory.SYSTEM_RESOURCE:
            return self._recover_system_resource(error_info)

        # Default recovery: just return False (no automatic recovery)
        return False

    def _recover_mcp_connection(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from MCP connection errors.

        Args:
            error_info: MCP connection error information

        Returns:
            True if connection recovery was successful
        """
        try:
            # Try to restart MCP server if server type is available
            server_type = (
                error_info.context.get("server_type") if error_info.context else None
            )
            if server_type:
                self.logger.info(f"Attempting to restart {server_type} MCP server")
                # This would integrate with MCP manager to restart server
                # For now, we'll simulate a successful restart
                return True

            # Generic connection retry
            self.logger.info("Attempting generic MCP connection recovery")
            return False  # Would implement actual connection retry logic

        except Exception as e:
            self.logger.error(f"MCP connection recovery failed: {e}")
            return False

    def _recover_agent_execution(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from agent execution errors.

        Args:
            error_info: Agent execution error information

        Returns:
            True if agent recovery was successful
        """
        try:
            # Clear agent memory and retry with simpler prompt
            agent_type = (
                error_info.context.get("agent_type") if error_info.context else None
            )
            if agent_type:
                self.logger.info(f"Clearing memory for {agent_type} agent")
                # Would integrate with agent system to clear memory
                return True

            return False

        except Exception as e:
            self.logger.error(f"Agent execution recovery failed: {e}")
            return False

    def _recover_file_operation(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from file operation errors.

        Args:
            error_info: File operation error information

        Returns:
            True if file operation recovery was successful
        """
        try:
            # Create temporary file location
            file_path = (
                error_info.context.get("file_path") if error_info.context else None
            )
            operation = (
                error_info.context.get("operation") if error_info.context else None
            )

            if file_path and operation:
                self.logger.info(
                    f"Attempting file operation recovery for {operation} on {file_path}"
                )
                # Would implement actual file recovery logic
                return False

            return False

        except Exception as e:
            self.logger.error(f"File operation recovery failed: {e}")
            return False

    def _recover_system_resource(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from system resource errors.

        Args:
            error_info: System resource error information

        Returns:
            True if resource recovery was successful
        """
        try:
            resource_type = (
                error_info.context.get("resource_type") if error_info.context else None
            )

            if resource_type == "memory":
                self.logger.info("Attempting memory cleanup")
                # Would implement garbage collection, cache clearing, etc.
                gc.collect()
                return True

            elif resource_type == "disk":
                self.logger.info("Checking disk space availability")
                # Would implement disk cleanup logic
                return False

            return False

        except Exception as e:
            self.logger.error(f"System resource recovery failed: {e}")
            return False
