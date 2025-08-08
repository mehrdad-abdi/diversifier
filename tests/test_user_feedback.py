"""Tests for user feedback system."""

import io
from unittest.mock import patch

from src.orchestration.user_feedback import (
    UserFeedback,
    FeedbackLevel,
    UserMessage,
    ProgressIndicator,
)
from src.orchestration.error_handling import ErrorInfo, ErrorCategory, ErrorSeverity
from src.orchestration.exceptions import DiversifierError


class TestUserMessage:
    """Test UserMessage dataclass."""

    def test_default_initialization(self):
        """Test default message initialization."""
        message = UserMessage("Test message")

        assert message.message == "Test message"
        assert message.level == FeedbackLevel.INFO
        assert message.show_timestamp is False
        assert message.prefix is None

    def test_full_initialization(self):
        """Test full message initialization."""
        message = UserMessage(
            message="Test message",
            level=FeedbackLevel.ERROR,
            show_timestamp=True,
            prefix="TEST",
        )

        assert message.message == "Test message"
        assert message.level == FeedbackLevel.ERROR
        assert message.show_timestamp is True
        assert message.prefix == "TEST"


class TestProgressIndicator:
    """Test ProgressIndicator class."""

    def test_initialization(self):
        """Test progress indicator initialization."""
        indicator = ProgressIndicator("Processing", show_spinner=True)

        assert indicator.message == "Processing"
        assert indicator.show_spinner is True
        assert indicator._stop_event is not None

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_start_without_spinner(self, mock_stdout):
        """Test starting progress indicator without spinner."""
        indicator = ProgressIndicator("Processing", show_spinner=False)
        indicator.start()

        output = mock_stdout.getvalue()
        assert "⏳ Processing..." in output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_stop_with_success_message(self, mock_stdout):
        """Test stopping progress indicator with success message."""
        indicator = ProgressIndicator("Processing", show_spinner=False)
        indicator.start()
        indicator.stop("Success!")

        output = mock_stdout.getvalue()
        assert "✅ Success!" in output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_stop_without_success_message(self, mock_stdout):
        """Test stopping progress indicator without success message."""
        indicator = ProgressIndicator("Processing", show_spinner=False)
        indicator.start()
        indicator.stop()

        output = mock_stdout.getvalue()
        assert "✅ Done" in output


class TestUserFeedback:
    """Test UserFeedback class."""

    def test_initialization(self):
        """Test user feedback initialization."""
        feedback = UserFeedback(verbose=True, quiet=False)

        assert feedback.verbose is True
        assert feedback.quiet is False

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_info_message(self, mock_stdout):
        """Test info message display."""
        feedback = UserFeedback()
        feedback.info("Test info message")

        output = mock_stdout.getvalue()
        assert "Test info message" in output
        assert "ℹ️" in output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_success_message(self, mock_stdout):
        """Test success message display."""
        feedback = UserFeedback()
        feedback.success("Test success message")

        output = mock_stdout.getvalue()
        assert "Test success message" in output
        assert "✅" in output

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_error_message(self, mock_stderr):
        """Test error message display."""
        feedback = UserFeedback()
        feedback.error("Test error message")

        output = mock_stderr.getvalue()
        assert "Test error message" in output
        assert "❌" in output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_warning_message(self, mock_stdout):
        """Test warning message display."""
        feedback = UserFeedback()
        feedback.warning("Test warning message")

        output = mock_stdout.getvalue()
        assert "Test warning message" in output
        assert "⚠️" in output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_debug_message_verbose(self, mock_stdout):
        """Test debug message display in verbose mode."""
        feedback = UserFeedback(verbose=True)
        feedback.debug("Test debug message")

        output = mock_stdout.getvalue()
        assert "Test debug message" in output
        assert "🔍" in output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_debug_message_not_verbose(self, mock_stdout):
        """Test debug message not displayed when not verbose."""
        feedback = UserFeedback(verbose=False)
        feedback.debug("Test debug message")

        output = mock_stdout.getvalue()
        assert "Test debug message" not in output

    def test_quiet_mode_suppression(self):
        """Test that quiet mode suppresses non-essential messages."""
        feedback = UserFeedback(quiet=True)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            feedback.info("Test info message")
            assert mock_stdout.getvalue() == ""

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            feedback.debug("Test debug message")
            assert mock_stdout.getvalue() == ""

        # Error messages should still be displayed
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            feedback.error("Test error message")
            assert "Test error message" in mock_stderr.getvalue()

    def test_display_error_info(self):
        """Test display of ErrorInfo."""
        feedback = UserFeedback()
        error_info = ErrorInfo(
            category=ErrorCategory.MCP_CONNECTION,
            severity=ErrorSeverity.HIGH,
            message="Connection failed",
            details="Server unreachable",
            recovery_suggestions=["Retry connection", "Check network"],
        )

        with (
            patch("sys.stderr", new_callable=io.StringIO) as mock_stderr,
            patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
        ):
            feedback.display_error_info(error_info)

            stderr_output = mock_stderr.getvalue()
            stdout_output = mock_stdout.getvalue()
            full_output = stderr_output + stdout_output

            assert "Connection failed" in full_output
            assert "Suggested actions:" in full_output
            assert "Retry connection" in full_output

    def test_display_diversifier_exception(self):
        """Test display of DiversifierError."""
        feedback = UserFeedback()
        exception = DiversifierError(
            message="Test error",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestions=["Check config"],
        )

        with (
            patch("sys.stderr", new_callable=io.StringIO) as mock_stderr,
            patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
        ):
            feedback.display_exception(exception)

            stderr_output = mock_stderr.getvalue()
            stdout_output = mock_stdout.getvalue()
            full_output = stderr_output + stdout_output

            assert "Test error" in full_output
            assert "Check config" in full_output

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_display_generic_exception(self, mock_stderr):
        """Test display of generic exception."""
        feedback = UserFeedback()
        exception = ValueError("Generic error")

        feedback.display_exception(exception)

        output = mock_stderr.getvalue()
        assert "Unexpected error: Generic error" in output

    def test_progress_context_manager(self):
        """Test progress context manager."""
        feedback = UserFeedback()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            with feedback.progress("Processing"):
                pass

            output = mock_stdout.getvalue()
            # Should contain progress indicators
            assert len(output) > 0

    def test_progress_context_manager_quiet(self):
        """Test progress context manager in quiet mode."""
        feedback = UserFeedback(quiet=True)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            with feedback.progress("Processing"):
                pass

            output = mock_stdout.getvalue()
            assert output == ""

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_step_progress(self, mock_stdout):
        """Test step progress display."""
        feedback = UserFeedback()
        feedback.step(2, 5, "Processing files")

        output = mock_stdout.getvalue()
        assert "Step 2/5" in output
        assert "Processing files" in output
        assert "[" in output  # Progress bar

    def test_step_progress_quiet(self):
        """Test step progress not displayed in quiet mode."""
        feedback = UserFeedback(quiet=True)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            feedback.step(1, 3, "Processing")
            assert mock_stdout.getvalue() == ""

    def test_create_progress_bar(self):
        """Test progress bar creation."""
        feedback = UserFeedback()

        # Test various progress values
        bar = feedback._create_progress_bar(0, 10, 10)
        assert "0%" in bar
        assert "░" in bar

        bar = feedback._create_progress_bar(5, 10, 10)
        assert "50%" in bar
        assert "█" in bar
        assert "░" in bar

        bar = feedback._create_progress_bar(10, 10, 10)
        assert "100%" in bar

        # Test edge case: total is 0
        bar = feedback._create_progress_bar(0, 0, 10)
        assert "0%" in bar

    @patch("builtins.input", return_value="y")
    def test_confirm_yes(self, mock_input):
        """Test confirmation with yes response."""
        feedback = UserFeedback()
        result = feedback.confirm("Continue?")

        assert result is True
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="n")
    def test_confirm_no(self, mock_input):
        """Test confirmation with no response."""
        feedback = UserFeedback()
        result = feedback.confirm("Continue?")

        assert result is False

    @patch("builtins.input", return_value="")
    def test_confirm_default_true(self, mock_input):
        """Test confirmation with empty response and default True."""
        feedback = UserFeedback()
        result = feedback.confirm("Continue?", default=True)

        assert result is True

    @patch("builtins.input", return_value="")
    def test_confirm_default_false(self, mock_input):
        """Test confirmation with empty response and default False."""
        feedback = UserFeedback()
        result = feedback.confirm("Continue?", default=False)

        assert result is False

    def test_confirm_quiet_mode(self):
        """Test confirmation in quiet mode returns default."""
        feedback = UserFeedback(quiet=True)

        result = feedback.confirm("Continue?", default=True)
        assert result is True

        result = feedback.confirm("Continue?", default=False)
        assert result is False

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_show_summary(self, mock_stdout):
        """Test summary display."""
        feedback = UserFeedback()
        items = ["Item 1", "Item 2", "Item 3"]
        feedback.show_summary("Test Summary", items)

        output = mock_stdout.getvalue()
        assert "📋 Test Summary" in output
        assert "• Item 1" in output
        assert "• Item 2" in output
        assert "• Item 3" in output

    def test_show_summary_quiet(self):
        """Test summary not displayed in quiet mode."""
        feedback = UserFeedback(quiet=True)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            feedback.show_summary("Test", ["Item 1"])
            assert mock_stdout.getvalue() == ""
