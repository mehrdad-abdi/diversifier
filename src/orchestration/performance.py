"""Performance monitoring and metrics collection for Diversifier."""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PerformanceConfig, get_config
from .logging_config import get_logger, get_correlation_id


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""

    operation_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000


@dataclass
class WorkflowMetrics:
    """Aggregated metrics for a complete workflow."""

    workflow_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    operations: List[OperationMetrics] = field(default_factory=list)
    success: bool = True
    files_processed: int = 0
    lines_modified: int = 0
    tests_run: int = 0
    tests_passed: int = 0

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Total duration in milliseconds."""
        return self.total_duration * 1000 if self.total_duration else None

    @property
    def success_rate(self) -> float:
        """Success rate of operations."""
        if not self.operations:
            return 1.0
        successful = sum(1 for op in self.operations if op.success)
        return successful / len(self.operations)

    @property
    def average_operation_duration(self) -> float:
        """Average operation duration in seconds."""
        if not self.operations:
            return 0.0
        return sum(op.duration for op in self.operations) / len(self.operations)


class PerformanceMonitor:
    """Monitors and collects performance metrics."""

    def __init__(self, config: PerformanceConfig):
        """Initialize performance monitor.

        Args:
            config: Performance configuration
        """
        self.config = config
        self.logger = get_logger("performance")
        self.current_workflow: Optional[WorkflowMetrics] = None
        self._operation_stack: List[str] = []

    def start_workflow(self, workflow_id: str) -> None:
        """Start monitoring a workflow.

        Args:
            workflow_id: Unique identifier for the workflow
        """
        if self.current_workflow is not None:
            self.logger.warning(
                f"Starting new workflow {workflow_id} while {self.current_workflow.workflow_id} is active"
            )

        self.current_workflow = WorkflowMetrics(
            workflow_id=workflow_id, start_time=time.time()
        )

        if self.config.enable_metrics:
            self.logger.info(f"🚀 Started workflow: {workflow_id}")

    def end_workflow(self, success: bool = True) -> Optional[WorkflowMetrics]:
        """End current workflow monitoring.

        Args:
            success: Whether the workflow completed successfully

        Returns:
            Completed workflow metrics or None if no workflow was active
        """
        if self.current_workflow is None:
            self.logger.warning("Attempted to end workflow but no workflow is active")
            return None

        self.current_workflow.end_time = time.time()
        self.current_workflow.total_duration = (
            self.current_workflow.end_time - self.current_workflow.start_time
        )
        self.current_workflow.success = success

        if self.config.enable_metrics:
            status = "✅ Completed" if success else "❌ Failed"
            duration = self.current_workflow.total_duration_ms
            self.logger.info(
                f"{status} workflow: {self.current_workflow.workflow_id} ({duration:.1f}ms)"
            )

            # Log performance summary
            self._log_workflow_summary(self.current_workflow)

        # Save metrics to file if configured
        if self.config.metrics_file:
            self._save_metrics_to_file(self.current_workflow)

        completed_workflow = self.current_workflow
        self.current_workflow = None
        return completed_workflow

    @contextmanager
    def monitor_operation(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager to monitor an operation.

        Args:
            operation_name: Name of the operation
            metadata: Optional metadata about the operation

        Yields:
            Operation context
        """
        start_time = time.time()
        correlation_id = get_correlation_id()
        error_message = None
        success = True

        if metadata is None:
            metadata = {}

        self._operation_stack.append(operation_name)

        try:
            if self.config.enable_metrics:
                self.logger.debug(f"⚙️  Starting operation: {operation_name}")
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            if self.config.enable_metrics:
                self.logger.error(
                    f"❌ Operation failed: {operation_name} - {error_message}"
                )
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # Create operation metrics
            op_metrics = OperationMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                error_message=error_message,
                metadata=metadata,
                correlation_id=correlation_id,
            )

            # Add to current workflow if active
            if self.current_workflow:
                self.current_workflow.operations.append(op_metrics)

            # Log slow operations
            if (
                self.config.log_slow_operations
                and duration > self.config.slow_operation_threshold
            ):
                self.logger.warning(
                    f"🐌 Slow operation: {operation_name} took {duration:.2f}s"
                )

            # Log completion
            if self.config.enable_metrics and success:
                self.logger.debug(
                    f"✅ Completed operation: {operation_name} ({duration*1000:.1f}ms)"
                )

            self._operation_stack.pop()

    def record_file_processed(self, file_path: str, lines_modified: int = 0) -> None:
        """Record that a file was processed.

        Args:
            file_path: Path to the processed file
            lines_modified: Number of lines modified
        """
        if self.current_workflow:
            self.current_workflow.files_processed += 1
            self.current_workflow.lines_modified += lines_modified

        if self.config.enable_metrics:
            self.logger.debug(
                f"📄 Processed file: {file_path} ({lines_modified} lines modified)"
            )

    def record_test_results(self, total_tests: int, passed_tests: int) -> None:
        """Record test execution results.

        Args:
            total_tests: Total number of tests run
            passed_tests: Number of tests that passed
        """
        if self.current_workflow:
            self.current_workflow.tests_run += total_tests
            self.current_workflow.tests_passed += passed_tests

        if self.config.enable_metrics:
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            self.logger.info(
                f"🧪 Test results: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)"
            )

    def get_current_workflow_metrics(self) -> Optional[WorkflowMetrics]:
        """Get current workflow metrics.

        Returns:
            Current workflow metrics or None if no workflow is active
        """
        return self.current_workflow

    def _log_workflow_summary(self, workflow: WorkflowMetrics) -> None:
        """Log workflow performance summary.

        Args:
            workflow: Completed workflow metrics
        """
        summary_lines = [
            f"📊 Workflow Summary: {workflow.workflow_id}",
            f"   Duration: {workflow.total_duration_ms:.1f}ms",
            f"   Operations: {len(workflow.operations)}",
            f"   Success Rate: {workflow.success_rate:.1%}",
            f"   Files Processed: {workflow.files_processed}",
            f"   Lines Modified: {workflow.lines_modified}",
            f"   Tests: {workflow.tests_passed}/{workflow.tests_run}",
        ]

        if workflow.operations:
            summary_lines.append(
                f"   Avg Operation Time: {workflow.average_operation_duration*1000:.1f}ms"
            )

            # Find slowest operation
            slowest = max(workflow.operations, key=lambda op: op.duration)
            summary_lines.append(
                f"   Slowest Operation: {slowest.operation_name} ({slowest.duration*1000:.1f}ms)"
            )

        for line in summary_lines:
            self.logger.info(line)

    def _save_metrics_to_file(self, workflow: WorkflowMetrics) -> None:
        """Save workflow metrics to file.

        Args:
            workflow: Workflow metrics to save
        """
        try:
            if self.config.metrics_file is None:
                return
            metrics_path = Path(self.config.metrics_file)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing metrics
            metrics_data = []
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        metrics_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self.logger.warning(
                        f"Could not load existing metrics from {metrics_path}"
                    )

            # Add new workflow metrics
            workflow_dict = asdict(workflow)
            metrics_data.append(workflow_dict)

            # Save updated metrics
            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=2, default=str)

            self.logger.debug(f"💾 Saved metrics to {metrics_path}")

        except Exception as e:
            self.logger.error(f"Failed to save metrics to file: {e}")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(
    config: Optional[PerformanceConfig] = None,
) -> PerformanceMonitor:
    """Get global performance monitor instance.

    Args:
        config: Optional performance configuration

    Returns:
        Performance monitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        if config is None:
            config = get_config().performance
        _performance_monitor = PerformanceMonitor(config)
    return _performance_monitor


def monitor_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator/context manager to monitor an operation.

    Args:
        operation_name: Name of the operation
        metadata: Optional metadata about the operation

    Returns:
        Context manager for monitoring the operation
    """
    return get_performance_monitor().monitor_operation(operation_name, metadata)


def start_workflow(workflow_id: str) -> None:
    """Start monitoring a workflow.

    Args:
        workflow_id: Unique identifier for the workflow
    """
    get_performance_monitor().start_workflow(workflow_id)


def end_workflow(success: bool = True) -> Optional[WorkflowMetrics]:
    """End current workflow monitoring.

    Args:
        success: Whether the workflow completed successfully

    Returns:
        Completed workflow metrics or None if no workflow was active
    """
    return get_performance_monitor().end_workflow(success)


def record_file_processed(file_path: str, lines_modified: int = 0) -> None:
    """Record that a file was processed.

    Args:
        file_path: Path to the processed file
        lines_modified: Number of lines modified
    """
    get_performance_monitor().record_file_processed(file_path, lines_modified)


def record_test_results(total_tests: int, passed_tests: int) -> None:
    """Record test execution results.

    Args:
        total_tests: Total number of tests run
        passed_tests: Number of tests that passed
    """
    get_performance_monitor().record_test_results(total_tests, passed_tests)
