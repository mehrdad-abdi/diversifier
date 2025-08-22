"""Logging configuration for the orchestration system."""

import logging
import sys
from typing import Optional

from .config import LoggingConfig, get_config


class DiversifierFormatter(logging.Formatter):
    """Custom formatter for diversifier logs."""

    def __init__(self, format_string: str):
        """Initialize the formatter.

        Args:
            format_string: Format string for log messages
        """
        super().__init__(format_string)


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Set up logging configuration for diversifier.

    Args:
        config: Logging configuration object. If None, uses default configuration.
    """
    if config is None:
        config = get_config().logging

    # Convert string level to logging constant
    numeric_level = getattr(logging, config.level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger("diversifier")
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = DiversifierFormatter(config.format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    # Log setup completion
    root_logger.info(f"Logging initialized - Level: {config.level}")


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
