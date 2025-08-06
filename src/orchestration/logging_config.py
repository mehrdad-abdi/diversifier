"""Logging configuration for the orchestration system."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class DiversifierFormatter(logging.Formatter):
    """Custom formatter for diversifier logs."""

    def __init__(self):
        """Initialize the formatter."""
        super().__init__()

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

        # Get color for log level
        color = self.colors.get(record.levelname, "")
        end_color = self.colors["ENDC"]

        # Format the message
        formatted = f"{color}[{timestamp}] {record.levelname:8} {record.name:25} | {record.getMessage()}{end_color}"

        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, console: bool = True
) -> None:
    """Set up logging configuration for diversifier.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to log to console
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger("diversifier")
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(DiversifierFormatter())
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)

        # Use plain formatter for file (no colors)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    # Log setup completion
    root_logger.info(
        f"Logging initialized - Level: {level}, Console: {console}, File: {log_file}"
    )


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
