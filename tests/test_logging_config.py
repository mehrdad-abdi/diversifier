"""Tests for logging configuration."""

import contextvars
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


from src.orchestration.config import LoggingConfig
from src.orchestration.logging_config import (
    DiversifierFormatter,
    clear_correlation_id,
    correlation_id,
    get_correlation_id,
    get_logger,
    log_agent_interaction,
    log_mcp_operation,
    log_migration_progress,
    log_workflow_step,
    set_correlation_id,
    setup_logging,
)


class TestDiversifierFormatter:
    """Tests for DiversifierFormatter class."""

    def test_init_default(self):
        """Test formatter initialization with defaults."""
        formatter = DiversifierFormatter()
        assert formatter.use_colors is True
        assert formatter.enable_correlation_ids is True
        assert len(formatter.colors) == 6  # 5 levels + ENDC

    def test_init_custom(self):
        """Test formatter initialization with custom settings."""
        formatter = DiversifierFormatter(use_colors=False, enable_correlation_ids=False)
        assert formatter.use_colors is False
        assert formatter.enable_correlation_ids is False

    def test_format_basic_record(self):
        """Test formatting a basic log record."""
        formatter = DiversifierFormatter(use_colors=False, enable_correlation_ids=False)
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
        assert "[" in formatted  # timestamp bracket

    def test_format_with_colors(self):
        """Test formatting with colors enabled."""
        formatter = DiversifierFormatter(use_colors=True, enable_correlation_ids=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "\033[31m" in formatted  # Red color for ERROR
        assert "\033[0m" in formatted  # End color
        assert "Error message" in formatted

    def test_format_with_correlation_id(self):
        """Test formatting with correlation ID."""
        formatter = DiversifierFormatter(use_colors=False, enable_correlation_ids=True)

        # Set correlation ID
        test_corr_id = "test-correlation-id-12345"
        correlation_id.set(test_corr_id)

        try:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname="",
                lineno=1,
                msg="Test with correlation",
                args=(),
                exc_info=None,
            )

            formatted = formatter.format(record)
            assert test_corr_id[:8] in formatted  # First 8 characters
            assert "Test with correlation" in formatted
        finally:
            correlation_id.set("")

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = DiversifierFormatter(use_colors=False, enable_correlation_ids=False)

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

            formatted = formatter.format(record)
            assert "Error occurred" in formatted
            assert "ValueError: Test exception" in formatted
            assert "Traceback" in formatted


class TestCorrelationIdFunctions:
    """Tests for correlation ID management functions."""

    def setup_method(self):
        """Clear correlation ID before each test."""
        correlation_id.set("")

    def test_set_correlation_id_with_value(self):
        """Test setting a specific correlation ID."""
        test_id = "test-123"
        result = set_correlation_id(test_id)

        assert result == test_id
        assert get_correlation_id() == test_id

    def test_set_correlation_id_auto_generate(self):
        """Test auto-generating correlation ID."""
        result = set_correlation_id()

        assert result != ""
        assert len(result) == 36  # UUID length
        assert get_correlation_id() == result

    def test_get_correlation_id_empty(self):
        """Test getting correlation ID when none is set."""
        result = get_correlation_id()
        assert result == ""

    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

        clear_correlation_id()
        assert get_correlation_id() == ""

    def test_correlation_id_context_isolation(self):
        """Test that correlation IDs are isolated between contexts."""

        def set_and_check_id(corr_id):
            set_correlation_id(corr_id)
            return get_correlation_id()

        # Test in different context vars
        ctx1 = contextvars.copy_context()
        ctx2 = contextvars.copy_context()

        result1 = ctx1.run(set_and_check_id, "context-1")
        result2 = ctx2.run(set_and_check_id, "context-2")

        assert result1 == "context-1"
        assert result2 == "context-2"


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

    def test_setup_logging_file_only(self):
        """Test setup logging with file handler only."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_file = f.name

        try:
            config = LoggingConfig(console=False, file_path=log_file)
            setup_logging(config)

            logger = logging.getLogger("diversifier")
            assert len(logger.handlers) == 1
            assert hasattr(logger.handlers[0], "stream")  # RotatingFileHandler
            assert Path(log_file).exists()
        finally:
            if Path(log_file).exists():
                Path(log_file).unlink()

    def test_setup_logging_both_handlers(self):
        """Test setup logging with both console and file handlers."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_file = f.name

        try:
            config = LoggingConfig(console=True, file_path=log_file)
            setup_logging(config)

            logger = logging.getLogger("diversifier")
            assert len(logger.handlers) == 2
        finally:
            if Path(log_file).exists():
                Path(log_file).unlink()

    def test_setup_logging_debug_level(self):
        """Test setup logging with DEBUG level."""
        config = LoggingConfig(level="DEBUG")
        setup_logging(config)

        logger = logging.getLogger("diversifier")
        assert logger.level == logging.DEBUG

    def test_setup_logging_no_correlation_ids(self):
        """Test setup logging with correlation IDs disabled."""
        config = LoggingConfig(enable_correlation_ids=False)
        setup_logging(config)

        logger = logging.getLogger("diversifier")
        formatter = logger.handlers[0].formatter
        assert isinstance(formatter, DiversifierFormatter)
        assert formatter.enable_correlation_ids is False

    @patch("src.orchestration.config.get_config")
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
