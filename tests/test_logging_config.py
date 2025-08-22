"""Tests for logging configuration."""

import logging
from unittest.mock import MagicMock, patch


from src.orchestration.config import LoggingConfig
from src.orchestration.logging_config import (
    DiversifierFormatter,
    get_logger,
    log_agent_interaction,
    log_mcp_operation,
    log_migration_progress,
    log_workflow_step,
    setup_logging,
)


class TestDiversifierFormatter:
    """Tests for DiversifierFormatter class."""

    def test_init_with_format_string(self):
        """Test formatter initialization with format string."""
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        formatter = DiversifierFormatter(format_string)
        assert formatter._style._fmt == format_string

    def test_format_basic_record(self):
        """Test formatting a basic log record."""
        format_string = "%(levelname)s | %(name)s | %(message)s"
        formatter = DiversifierFormatter(format_string)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "INFO" in formatted
        assert "test.logger" in formatted
        assert "Test message" in formatted


class TestSetupLogging:
    """Tests for setup_logging function."""

    def setup_method(self):
        """Clear logging handlers before each test."""
        logger = logging.getLogger("diversifier")
        logger.handlers.clear()
        logger.propagate = True

    def test_setup_logging_default(self):
        """Test setup logging with default configuration."""
        config = LoggingConfig()
        setup_logging(config)

        logger = logging.getLogger("diversifier")
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler only
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.propagate is False

    def test_setup_logging_debug_level(self):
        """Test setup logging with DEBUG level."""
        config = LoggingConfig(level="DEBUG")
        setup_logging(config)

        logger = logging.getLogger("diversifier")
        assert logger.level == logging.DEBUG

    def test_setup_logging_custom_format(self):
        """Test setup logging with custom format string."""
        custom_format = "%(name)s - %(levelname)s - %(message)s"
        config = LoggingConfig(format_string=custom_format)
        setup_logging(config)

        logger = logging.getLogger("diversifier")
        formatter = logger.handlers[0].formatter
        assert isinstance(formatter, DiversifierFormatter)
        assert formatter._style._fmt == custom_format

    @patch("src.orchestration.logging_config.get_config")
    def test_setup_logging_no_config(self, mock_get_config):
        """Test setup logging without providing config."""
        mock_config = MagicMock()
        mock_config.logging = LoggingConfig()
        mock_get_config.return_value = mock_config

        setup_logging()

        mock_get_config.assert_called_once()
        logger = logging.getLogger("diversifier")
        assert len(logger.handlers) >= 1


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test.component")
        assert logger.name == "diversifier.test.component"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_hierarchy(self):
        """Test logger hierarchy."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        assert parent_logger.name == "diversifier.parent"
        assert child_logger.name == "diversifier.parent.child"


class TestLoggingHelpers:
    """Tests for logging helper functions."""

    def setup_method(self):
        """Set up test logger."""
        self.mock_logger = MagicMock()

    def test_log_workflow_step_start(self):
        """Test logging workflow step start."""
        log_workflow_step(self.mock_logger, "test_step", "start")
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Starting workflow step: test_step" in args[0]
        assert "🚀" in args[0]

    def test_log_workflow_step_complete(self):
        """Test logging workflow step completion."""
        log_workflow_step(self.mock_logger, "test_step", "complete")
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Completed workflow step: test_step" in args[0]
        assert "✅" in args[0]

    def test_log_workflow_step_error(self):
        """Test logging workflow step error."""
        log_workflow_step(self.mock_logger, "test_step", "error")
        self.mock_logger.error.assert_called_once()
        args = self.mock_logger.error.call_args[0]
        assert "Failed workflow step: test_step" in args[0]
        assert "❌" in args[0]

    def test_log_workflow_step_progress(self):
        """Test logging workflow step progress."""
        log_workflow_step(self.mock_logger, "test_step", "progress")
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Processing workflow step: test_step" in args[0]
        assert "⚙️" in args[0]

    def test_log_workflow_step_custom(self):
        """Test logging workflow step with custom stage."""
        log_workflow_step(self.mock_logger, "test_step", "custom_stage")
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Workflow step test_step: custom_stage" in args[0]
        assert "📋" in args[0]

    def test_log_mcp_operation_success(self):
        """Test logging successful MCP operation."""
        log_mcp_operation(self.mock_logger, "filesystem", "read_file", True)
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "MCP filesystem: read_file" in args[0]
        assert "✅" in args[0]

    def test_log_mcp_operation_failure(self):
        """Test logging failed MCP operation."""
        log_mcp_operation(self.mock_logger, "filesystem", "read_file", False)
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "MCP filesystem: read_file" in args[0]
        assert "❌" in args[0]

    def test_log_agent_interaction(self):
        """Test logging agent interaction."""
        log_agent_interaction(self.mock_logger, "migrator", "convert imports")
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Agent migrator: convert imports" in args[0]
        assert "🤖" in args[0]

    def test_log_agent_interaction_with_duration(self):
        """Test logging agent interaction with duration."""
        log_agent_interaction(self.mock_logger, "migrator", "convert imports", 2.5)
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Agent migrator: convert imports (2.50s)" in args[0]
        assert "🤖" in args[0]

    def test_log_migration_progress(self):
        """Test logging migration progress."""
        log_migration_progress(self.mock_logger, 5, 10)
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Migration Progress: 5/10 (50.0%)" in args[0]
        assert "📊" in args[0]

    def test_log_migration_progress_with_current_file(self):
        """Test logging migration progress with current file."""
        log_migration_progress(self.mock_logger, 3, 8, "src/main.py")
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Migration Progress: 3/8 (37.5%)" in args[0]
        assert "Processing: src/main.py" in args[0]
        assert "📊" in args[0]

    def test_log_migration_progress_zero_total(self):
        """Test logging migration progress with zero total files."""
        log_migration_progress(self.mock_logger, 0, 0)
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        assert "Migration Progress: 0/0 (0.0%)" in args[0]
