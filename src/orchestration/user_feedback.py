"""User feedback system for Diversifier tool."""

import sys
import time
import traceback
from typing import Optional, List
from contextlib import contextmanager
from threading import Thread, Event
from dataclasses import dataclass
from enum import Enum

from src.orchestration.exceptions import DiversifierError
from src.orchestration.error_handling import ErrorInfo, ErrorSeverity


class FeedbackLevel(Enum):
    """Feedback level for user messages."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class UserMessage:
    """User message with formatting information."""

    message: str
    level: FeedbackLevel = FeedbackLevel.INFO
    show_timestamp: bool = False
    prefix: Optional[str] = None


class ProgressIndicator:
    """Progress indicator for long-running operations."""

    def __init__(self, message: str, show_spinner: bool = True):
        """Initialize progress indicator.

        Args:
            message: Progress message to display
            show_spinner: Whether to show animated spinner
        """
        self.message = message
        self.show_spinner = show_spinner
        self._stop_event = Event()
        self._spinner_thread: Optional[Thread] = None
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._current_char = 0

    def start(self) -> None:
        """Start the progress indicator."""
        if self.show_spinner:
            self._stop_event.clear()
            self._spinner_thread = Thread(target=self._animate_spinner)
            self._spinner_thread.daemon = True
            self._spinner_thread.start()
        else:
            print(f"⏳ {self.message}...", flush=True)

    def stop(self, success_message: Optional[str] = None) -> None:
        """Stop the progress indicator.

        Args:
            success_message: Optional success message to display
        """
        if self.show_spinner and self._spinner_thread:
            self._stop_event.set()
            self._spinner_thread.join(timeout=0.1)
            # Clear the spinner line
            print("\r" + " " * (len(self.message) + 10) + "\r", end="", flush=True)

        if success_message:
            print(f"✅ {success_message}")
        elif not self.show_spinner:
            print("✅ Done")

    def _animate_spinner(self) -> None:
        """Animate the spinner."""
        while not self._stop_event.is_set():
            char = self._spinner_chars[self._current_char]
            print(f"\r{char} {self.message}...", end="", flush=True)
            self._current_char = (self._current_char + 1) % len(self._spinner_chars)
            time.sleep(0.1)


class UserFeedback:
    """User feedback system with error handling and progress indicators."""

    def __init__(self, verbose: bool = False, quiet: bool = False):
        """Initialize user feedback system.

        Args:
            verbose: Enable verbose output
            quiet: Suppress non-critical messages
        """
        self.verbose = verbose
        self.quiet = quiet
        self._color_codes = {
            FeedbackLevel.DEBUG: "\033[90m",  # Dark gray
            FeedbackLevel.INFO: "\033[94m",  # Blue
            FeedbackLevel.WARNING: "\033[93m",  # Yellow
            FeedbackLevel.ERROR: "\033[91m",  # Red
            FeedbackLevel.SUCCESS: "\033[92m",  # Green
        }
        self._reset_code = "\033[0m"
        self._icons = {
            FeedbackLevel.DEBUG: "🔍",
            FeedbackLevel.INFO: "ℹ️",
            FeedbackLevel.WARNING: "⚠️",
            FeedbackLevel.ERROR: "❌",
            FeedbackLevel.SUCCESS: "✅",
        }

    def info(self, message: str, prefix: Optional[str] = None) -> None:
        """Display info message.

        Args:
            message: Message to display
            prefix: Optional prefix for the message
        """
        self._display_message(
            UserMessage(message=message, level=FeedbackLevel.INFO, prefix=prefix)
        )

    def success(self, message: str, prefix: Optional[str] = None) -> None:
        """Display success message.

        Args:
            message: Message to display
            prefix: Optional prefix for the message
        """
        self._display_message(
            UserMessage(message=message, level=FeedbackLevel.SUCCESS, prefix=prefix)
        )

    def warning(self, message: str, prefix: Optional[str] = None) -> None:
        """Display warning message.

        Args:
            message: Message to display
            prefix: Optional prefix for the message
        """
        self._display_message(
            UserMessage(message=message, level=FeedbackLevel.WARNING, prefix=prefix)
        )

    def error(self, message: str, prefix: Optional[str] = None) -> None:
        """Display error message.

        Args:
            message: Message to display
            prefix: Optional prefix for the message
        """
        self._display_message(
            UserMessage(message=message, level=FeedbackLevel.ERROR, prefix=prefix)
        )

    def debug(self, message: str, prefix: Optional[str] = None) -> None:
        """Display debug message (only if verbose mode enabled).

        Args:
            message: Message to display
            prefix: Optional prefix for the message
        """
        if self.verbose:
            self._display_message(
                UserMessage(message=message, level=FeedbackLevel.DEBUG, prefix=prefix)
            )

    def display_error_info(self, error_info: ErrorInfo) -> None:
        """Display comprehensive error information.

        Args:
            error_info: Error information to display
        """
        # Display main error message
        severity_icon = {
            ErrorSeverity.LOW: "⚠️",
            ErrorSeverity.MEDIUM: "❌",
            ErrorSeverity.HIGH: "🚨",
            ErrorSeverity.CRITICAL: "💥",
        }

        icon = severity_icon.get(error_info.severity, "❌")
        category = error_info.category.value.replace("_", " ").title()

        self.error(f"{icon} {category}: {error_info.message}")

        # Display details if available
        if error_info.details and self.verbose:
            self.info(f"Details: {error_info.details}")

        # Display context if available and verbose
        if error_info.context and self.verbose:
            self.debug(f"Context: {error_info.context}")

        # Display recovery suggestions
        if error_info.recovery_suggestions:
            self.info("Suggested actions:")
            for i, suggestion in enumerate(error_info.recovery_suggestions, 1):
                self.info(f"  {i}. {suggestion}")

    def display_exception(self, exception: Exception) -> None:
        """Display exception information with user-friendly formatting.

        Args:
            exception: Exception to display
        """
        if isinstance(exception, DiversifierError):
            # Create ErrorInfo from DiversifierError for consistent display
            error_info = ErrorInfo(
                category=exception.category,
                severity=exception.severity,
                message=str(exception),
                details=exception.details,
                context=exception.context,
                recovery_suggestions=exception.recovery_suggestions,
            )
            self.display_error_info(error_info)
        else:
            # Display generic exception
            self.error(f"Unexpected error: {str(exception)}")
            if self.verbose:
                self.debug(f"Traceback:\n{traceback.format_exc()}")

    @contextmanager
    def progress(self, message: str, success_message: Optional[str] = None):
        """Context manager for progress indication.

        Args:
            message: Progress message
            success_message: Optional success message

        Usage:
            with feedback.progress("Processing files"):
                # Long running operation
                process_files()
        """
        if self.quiet:
            yield
            return

        indicator = ProgressIndicator(message, show_spinner=not self.quiet)
        indicator.start()
        try:
            yield
            indicator.stop(success_message)
        except Exception:
            indicator.stop()
            raise

    def step(self, step_num: int, total_steps: int, message: str) -> None:
        """Display step progress.

        Args:
            step_num: Current step number (1-based)
            total_steps: Total number of steps
            message: Step description
        """
        if self.quiet:
            return

        progress_bar = self._create_progress_bar(step_num, total_steps)
        self.info(f"Step {step_num}/{total_steps}: {message} {progress_bar}")

    def _create_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Create a text progress bar.

        Args:
            current: Current progress value
            total: Total progress value
            width: Width of progress bar in characters

        Returns:
            Progress bar string
        """
        if total == 0:
            return "[" + " " * width + "] 0%"

        percentage = min(100, int((current / total) * 100))
        filled = int((current / total) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage}%"

    def _display_message(self, message: UserMessage) -> None:
        """Display a formatted message.

        Args:
            message: Message to display
        """
        # Skip non-essential messages in quiet mode
        if self.quiet and message.level in [
            FeedbackLevel.DEBUG,
            FeedbackLevel.INFO,
        ]:
            return

        # Format message
        color = self._color_codes.get(message.level, "")
        icon = self._icons.get(message.level, "")
        reset = self._reset_code

        prefix = f"{message.prefix}: " if message.prefix else ""
        timestamp = f"[{time.strftime('%H:%M:%S')}] " if message.show_timestamp else ""

        formatted_message = f"{color}{timestamp}{icon} {prefix}{message.message}{reset}"

        # Output to appropriate stream
        output_stream = (
            sys.stderr if message.level == FeedbackLevel.ERROR else sys.stdout
        )
        print(formatted_message, file=output_stream, flush=True)

    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask user for confirmation.

        Args:
            message: Confirmation message
            default: Default response if user just presses enter

        Returns:
            True if user confirms, False otherwise
        """
        if self.quiet:
            return default

        default_text = " [Y/n]" if default else " [y/N]"
        response = input(f"❓ {message}{default_text}: ").strip().lower()

        if not response:
            return default

        return response in ["y", "yes", "true", "1"]

    def show_summary(self, title: str, items: List[str]) -> None:
        """Show a summary with title and bullet points.

        Args:
            title: Summary title
            items: List of summary items
        """
        if self.quiet:
            return

        self.info(f"\n📋 {title}")
        for item in items:
            self.info(f"  • {item}")
        self.info("")  # Empty line for spacing
