"""Tests for performance monitoring."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.orchestration.performance
from src.orchestration.config import PerformanceConfig
from src.orchestration.performance import (
    OperationMetrics,
    PerformanceMonitor,
    WorkflowMetrics,
    end_workflow,
    get_performance_monitor,
    monitor_operation,
    record_file_processed,
    record_test_results,
    start_workflow,
)


class TestOperationMetrics:
    """Tests for OperationMetrics dataclass."""

    def test_operation_metrics_creation(self):
        """Test creating operation metrics."""
        start_time = time.time()
        end_time = start_time + 1.5
        duration = end_time - start_time

        metrics = OperationMetrics(
            operation_name="test_operation",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=True,
            error_message=None,
            metadata={"key": "value"},
            correlation_id="test-123",
        )

        assert metrics.operation_name == "test_operation"
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.duration == duration
        assert metrics.success is True
        assert metrics.error_message is None
        assert metrics.metadata == {"key": "value"}
        assert metrics.correlation_id == "test-123"
        assert abs(metrics.duration_ms - 1500) < 10  # Allow small timing variance

    def test_operation_metrics_failure(self):
        """Test creating operation metrics for failure."""
        metrics = OperationMetrics(
            operation_name="failed_operation",
            start_time=1000.0,
            end_time=1001.0,
            duration=1.0,
            success=False,
            error_message="Test error",
        )

        assert metrics.success is False
        assert metrics.error_message == "Test error"


class TestWorkflowMetrics:
    """Tests for WorkflowMetrics dataclass."""

    def test_workflow_metrics_creation(self):
        """Test creating workflow metrics."""
        metrics = WorkflowMetrics(workflow_id="test-workflow", start_time=1000.0)

        assert metrics.workflow_id == "test-workflow"
        assert metrics.start_time == 1000.0
        assert metrics.end_time is None
        assert metrics.total_duration is None
        assert len(metrics.operations) == 0
        assert metrics.success is True
        assert metrics.files_processed == 0
        assert metrics.lines_modified == 0
        assert metrics.tests_run == 0
        assert metrics.tests_passed == 0

    def test_workflow_metrics_completion(self):
        """Test workflow metrics after completion."""
        metrics = WorkflowMetrics(
            workflow_id="test-workflow",
            start_time=1000.0,
            end_time=1005.0,
            total_duration=5.0,
            files_processed=10,
            lines_modified=50,
            tests_run=25,
            tests_passed=23,
        )

        assert metrics.total_duration_ms == 5000.0
        assert (
            abs(metrics.success_rate - 1.0) < 0.001
        )  # No operations, so success rate is 1.0

    def test_workflow_metrics_with_operations(self):
        """Test workflow metrics with operations."""
        op1 = OperationMetrics("op1", 1000.0, 1001.0, 1.0, True)
        op2 = OperationMetrics("op2", 1001.0, 1003.0, 2.0, True)
        op3 = OperationMetrics("op3", 1003.0, 1004.0, 1.0, False, "Error")

        metrics = WorkflowMetrics(
            workflow_id="test-workflow", start_time=1000.0, operations=[op1, op2, op3]
        )

        assert abs(metrics.success_rate - 0.667) < 0.01  # 2/3 success
        assert abs(metrics.average_operation_duration - 1.333) < 0.01  # (1+2+1)/3


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PerformanceConfig(enable_metrics=True, log_slow_operations=True)
        self.monitor = PerformanceMonitor(self.config)

    def test_monitor_creation(self):
        """Test creating performance monitor."""
        assert self.monitor.config == self.config
        assert self.monitor.current_workflow is None
        assert len(self.monitor._operation_stack) == 0

    def test_start_workflow(self):
        """Test starting workflow monitoring."""
        self.monitor.start_workflow("test-workflow")

        assert self.monitor.current_workflow is not None
        assert self.monitor.current_workflow.workflow_id == "test-workflow"
        assert isinstance(self.monitor.current_workflow.start_time, float)
        assert self.monitor.current_workflow.end_time is None

    def test_end_workflow(self):
        """Test ending workflow monitoring."""
        self.monitor.start_workflow("test-workflow")

        # Add some operations
        with self.monitor.monitor_operation("test_op"):
            time.sleep(0.01)  # Small sleep to ensure duration > 0

        completed = self.monitor.end_workflow(success=True)

        assert completed is not None
        assert completed.workflow_id == "test-workflow"
        assert completed.success is True
        assert completed.end_time is not None
        assert completed.total_duration is not None
        assert completed.total_duration > 0
        assert len(completed.operations) == 1
        assert self.monitor.current_workflow is None

    def test_end_workflow_no_active(self):
        """Test ending workflow when none is active."""
        result = self.monitor.end_workflow()
        assert result is None

    def test_monitor_operation_success(self):
        """Test monitoring successful operation."""
        self.monitor.start_workflow("test-workflow")

        with self.monitor.monitor_operation("test_operation", {"key": "value"}):
            time.sleep(0.01)

        workflow = self.monitor.current_workflow
        assert len(workflow.operations) == 1

        operation = workflow.operations[0]
        assert operation.operation_name == "test_operation"
        assert operation.success is True
        assert operation.error_message is None
        assert operation.metadata == {"key": "value"}
        assert operation.duration > 0

    def test_monitor_operation_failure(self):
        """Test monitoring failed operation."""
        self.monitor.start_workflow("test-workflow")

        with pytest.raises(ValueError):
            with self.monitor.monitor_operation("failing_operation"):
                raise ValueError("Test error")

        workflow = self.monitor.current_workflow
        assert len(workflow.operations) == 1

        operation = workflow.operations[0]
        assert operation.operation_name == "failing_operation"
        assert operation.success is False
        assert operation.error_message == "Test error"

    def test_monitor_operation_without_workflow(self):
        """Test monitoring operation without active workflow."""
        with self.monitor.monitor_operation("standalone_operation"):
            time.sleep(0.01)

        # Operation should complete without errors even without active workflow
        assert self.monitor.current_workflow is None

    @patch("src.orchestration.performance.get_logger")
    def test_slow_operation_logging(self, mock_get_logger):
        """Test logging of slow operations."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Set very low threshold to trigger slow operation warning
        config = PerformanceConfig(
            enable_metrics=True,
            log_slow_operations=True,
            slow_operation_threshold=0.001,  # 1ms
        )
        monitor = PerformanceMonitor(config)

        with monitor.monitor_operation("slow_operation"):
            time.sleep(0.01)  # 10ms, should be > threshold

        # Check that warning was logged
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Slow operation: slow_operation" in warning_call

    def test_record_file_processed(self):
        """Test recording file processing."""
        self.monitor.start_workflow("test-workflow")

        self.monitor.record_file_processed("test.py", 25)

        workflow = self.monitor.current_workflow
        assert workflow.files_processed == 1
        assert workflow.lines_modified == 25

        self.monitor.record_file_processed("another.py", 10)
        assert workflow.files_processed == 2
        assert workflow.lines_modified == 35

    def test_record_test_results(self):
        """Test recording test results."""
        self.monitor.start_workflow("test-workflow")

        self.monitor.record_test_results(10, 8)

        workflow = self.monitor.current_workflow
        assert workflow.tests_run == 10
        assert workflow.tests_passed == 8

        self.monitor.record_test_results(5, 5)
        assert workflow.tests_run == 15
        assert workflow.tests_passed == 13

    def test_get_current_workflow_metrics(self):
        """Test getting current workflow metrics."""
        assert self.monitor.get_current_workflow_metrics() is None

        self.monitor.start_workflow("test-workflow")
        current = self.monitor.get_current_workflow_metrics()

        assert current is not None
        assert current.workflow_id == "test-workflow"

    def test_save_metrics_to_file(self):
        """Test saving metrics to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            metrics_file = f.name

        try:
            config = PerformanceConfig(enable_metrics=True, metrics_file=metrics_file)
            monitor = PerformanceMonitor(config)

            monitor.start_workflow("test-workflow")
            with monitor.monitor_operation("test_op"):
                time.sleep(0.01)
            monitor.end_workflow()

            # Check that file was created and contains data
            assert Path(metrics_file).exists()

            with open(metrics_file) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["workflow_id"] == "test-workflow"
            assert len(data[0]["operations"]) == 1

        finally:
            if Path(metrics_file).exists():
                Path(metrics_file).unlink()

    def test_save_metrics_append_to_existing(self):
        """Test appending metrics to existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            metrics_file = f.name
            # Write existing data
            json.dump([{"workflow_id": "existing"}], f)

        try:
            config = PerformanceConfig(enable_metrics=True, metrics_file=metrics_file)
            monitor = PerformanceMonitor(config)

            monitor.start_workflow("new-workflow")
            monitor.end_workflow()

            # Check that new data was appended
            with open(metrics_file) as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["workflow_id"] == "existing"
            assert data[1]["workflow_id"] == "new-workflow"

        finally:
            if Path(metrics_file).exists():
                Path(metrics_file).unlink()


class TestGlobalFunctions:
    """Tests for global performance monitoring functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear global monitor
        src.orchestration.performance._performance_monitor = None

    def test_get_performance_monitor_singleton(self):
        """Test that get_performance_monitor returns singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        assert monitor1 is monitor2
        assert isinstance(monitor1, PerformanceMonitor)

    def test_get_performance_monitor_with_config(self):
        """Test get_performance_monitor with custom config."""
        config = PerformanceConfig(enable_metrics=False)
        monitor = get_performance_monitor(config)

        assert monitor.config.enable_metrics is False

    @patch("src.orchestration.performance.get_performance_monitor")
    def test_monitor_operation_decorator(self, mock_get_monitor):
        """Test monitor_operation as decorator/context manager."""
        mock_monitor = MagicMock()
        mock_get_monitor.return_value = mock_monitor

        with monitor_operation("test_op", {"key": "value"}):
            pass

        mock_monitor.monitor_operation.assert_called_once_with(
            "test_op", {"key": "value"}
        )

    @patch("src.orchestration.performance.get_performance_monitor")
    def test_start_workflow_global(self, mock_get_monitor):
        """Test global start_workflow function."""
        mock_monitor = MagicMock()
        mock_get_monitor.return_value = mock_monitor

        start_workflow("test-workflow")

        mock_monitor.start_workflow.assert_called_once_with("test-workflow")

    @patch("src.orchestration.performance.get_performance_monitor")
    def test_end_workflow_global(self, mock_get_monitor):
        """Test global end_workflow function."""
        mock_monitor = MagicMock()
        mock_get_monitor.return_value = mock_monitor
        mock_monitor.end_workflow.return_value = "result"

        result = end_workflow(False)

        mock_monitor.end_workflow.assert_called_once_with(False)
        assert result == "result"

    @patch("src.orchestration.performance.get_performance_monitor")
    def test_record_file_processed_global(self, mock_get_monitor):
        """Test global record_file_processed function."""
        mock_monitor = MagicMock()
        mock_get_monitor.return_value = mock_monitor

        record_file_processed("test.py", 10)

        mock_monitor.record_file_processed.assert_called_once_with("test.py", 10)

    @patch("src.orchestration.performance.get_performance_monitor")
    def test_record_test_results_global(self, mock_get_monitor):
        """Test global record_test_results function."""
        mock_monitor = MagicMock()
        mock_get_monitor.return_value = mock_monitor

        record_test_results(10, 8)

        mock_monitor.record_test_results.assert_called_once_with(10, 8)
