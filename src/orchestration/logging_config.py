"""Logging configuration for the orchestration system."""

import contextvars
import logging
import logging.handlers
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import LoggingConfig


# Context variable for correlation IDs
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


class DiversifierFormatter(logging.Formatter):
    """Custom formatter for diversifier logs."""

    def __init__(self, use_colors: bool = True, enable_correlation_ids: bool = True):
        """Initialize the formatter.

        Args:
            use_colors: Whether to use color formatting
            enable_correlation_ids: Whether to include correlation IDs
        """
        super().__init__()
        self.use_colors = use_colors
        self.enable_correlation_ids = enable_correlation_ids

        # Color codes for different log levels
        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
            "ENDC": "\033[0m",  # End color
        }

    def format(self, record):
        """Format log record with colors and structure."""
        # Add timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Get correlation ID if enabled
        corr_id = ""
        if self.enable_correlation_ids:
            current_corr_id = correlation_id.get()
            if current_corr_id:
                corr_id = f"[{current_corr_id[:8]}] "

        # Get color for log level
        color = ""
        end_color = ""
        if self.use_colors:
            color = self.colors.get(record.levelname, "")
            end_color = self.colors["ENDC"]

        # Format the message
        formatted = f"{color}[{timestamp}] {corr_id}{record.levelname:8} {record.name:25} | {record.getMessage()}{end_color}"

        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Set up logging configuration for diversifier.

    Args:
        config: Logging configuration object. If None, uses default configuration.
    """
    if config is None:
        from .config import get_config

        config = get_config().logging

    # Convert string level to logging constant
    numeric_level = getattr(logging, config.level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger("diversifier")
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = DiversifierFormatter(
            use_colors=True, enable_correlation_ids=config.enable_correlation_ids
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if config.file_path:
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler for automatic log rotation
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(numeric_level)

        # Use plain formatter for file (no colors) with correlation IDs
        file_formatter = DiversifierFormatter(
            use_colors=False, enable_correlation_ids=config.enable_correlation_ids
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    # Log setup completion
    root_logger.info(
        f"Logging initialized - Level: {config.level}, Console: {config.console}, "
        f"File: {config.file_path}, Correlation IDs: {config.enable_correlation_ids}"
    )


def set_correlation_id(corr_id: Optional[str] = None) -> str:
    """Set correlation ID for current context.

    Args:
        corr_id: Correlation ID to set. If None, generates a new UUID.

    Returns:
        The correlation ID that was set
    """
    if corr_id is None:
        corr_id = str(uuid.uuid4())

    correlation_id.set(corr_id)
    return corr_id


def get_correlation_id() -> str:
    """Get current correlation ID.

    Returns:
        Current correlation ID or empty string if not set
    """
    return correlation_id.get()


def clear_correlation_id() -> None:
    """Clear current correlation ID."""
    correlation_id.set("")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific component.

    Args:
        name: Logger name (will be prefixed with 'diversifier.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"diversifier.{name}")


def log_workflow_step(logger: logging.Logger, step_name: str, stage: str) -> None:
    """Log workflow step execution.

    Args:
        logger: Logger instance
        step_name: Name of the workflow step
        stage: Current stage (start, progress, complete, error)
    """
    if stage == "start":
        logger.info(f"🚀 Starting workflow step: {step_name}")
    elif stage == "progress":
        logger.info(f"⚙️  Processing workflow step: {step_name}")
    elif stage == "complete":
        logger.info(f"✅ Completed workflow step: {step_name}")
    elif stage == "error":
        logger.error(f"❌ Failed workflow step: {step_name}")
    else:
        logger.info(f"📋 Workflow step {step_name}: {stage}")


def log_mcp_operation(
    logger: logging.Logger, server_type: str, operation: str, success: bool
) -> None:
    """Log MCP server operations.

    Args:
        logger: Logger instance
        server_type: Type of MCP server
        operation: Operation being performed
        success: Whether operation was successful
    """
    status = "✅" if success else "❌"
    logger.info(f"{status} MCP {server_type}: {operation}")


def log_agent_interaction(
    logger: logging.Logger, agent_type: str, task: str, duration: Optional[float] = None
) -> None:
    """Log agent interactions.

    Args:
        logger: Logger instance
        agent_type: Type of agent
        task: Task description
        duration: Optional execution duration in seconds
    """
    duration_str = f" ({duration:.2f}s)" if duration else ""
    logger.info(f"🤖 Agent {agent_type}: {task}{duration_str}")


def log_migration_progress(
    logger: logging.Logger,
    files_processed: int,
    total_files: int,
    current_file: Optional[str] = None,
) -> None:
    """Log migration progress.

    Args:
        logger: Logger instance
        files_processed: Number of files processed
        total_files: Total number of files to process
        current_file: Currently processing file
    """
    percentage = (files_processed / total_files * 100) if total_files > 0 else 0
    current_info = f" - Processing: {current_file}" if current_file else ""
    logger.info(
        f"📊 Migration Progress: {files_processed}/{total_files} ({percentage:.1f}%){current_info}"
    )
