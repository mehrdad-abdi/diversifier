"""Tests for coordinator test generation workflow integration."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.orchestration.coordinator import DiversificationCoordinator
from src.orchestration.acceptance_test_generator import (
    WorkflowExecutionResult,
    AcceptanceTestGenerationResult,
    AcceptanceTestSuite,
)


class TestCoordinatorTestGeneration:
    """Test coordinator's Docker-based test generation integration."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a coordinator with mocked dependencies."""
        with (
            patch("src.orchestration.coordinator.AgentManager"),
            patch("src.orchestration.coordinator.MCPManager"),
            patch("src.orchestration.coordinator.AcceptanceTestGenerator"),
            patch("src.orchestration.coordinator.DocumentationAnalyzer"),
            patch("src.orchestration.coordinator.SourceCodeAnalyzer"),
        ):
            coordinator = DiversificationCoordinator(
                project_path="/test/project",
                source_library="requests",
                target_library="httpx",
            )
            return coordinator

    @pytest.fixture
    def sample_workflow_result(self):
        """Create sample workflow execution result."""
        generation_result = AcceptanceTestGenerationResult(
            test_suites=[
                AcceptanceTestSuite(
                    name="http_api_tests",
                    description="HTTP API tests",
                    test_file_content="def test_api(): pass",
                    docker_compose_content="version: '3.8'",
                )
            ],
            test_scenarios=[],
            docker_configuration={},
            test_dependencies=["pytest", "requests"],
            coverage_analysis={"total": 1},
            generation_confidence=0.8,
        )

        return WorkflowExecutionResult(
            workflow_id="test-workflow-123",
            success=True,
            generation_result=generation_result,
            docker_compose_path="/test/project/acceptance_tests/docker-compose.test.yml",
            test_image_id="sha256:abc123",
            execution_logs=["Test generation completed"],
            error_messages=[],
            execution_time_seconds=30.0,
            timestamp="2024-01-01T12:00:00Z",
        )

    @pytest.mark.asyncio
    async def test_generate_tests_success(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test successful Docker-based test generation."""
        # Mock analyzers with correct method names
        mock_coordinator.doc_analyzer.analyze_project_documentation = AsyncMock(
            return_value=Mock()
        )
        mock_coordinator.source_analyzer.analyze_project_source_code = AsyncMock(
            return_value=Mock()
        )

        # Mock test generator
        mock_coordinator.acceptance_test_generator.run_complete_workflow = AsyncMock(
            return_value=sample_workflow_result
        )

        result = await mock_coordinator._generate_tests()

        assert result["success"] is True
        assert result["test_suites"] == 1
        assert result["docker_compose_available"] is True
        assert result["generation_confidence"] == 0.8
        assert "acceptance_tests" in result["output_directory"]

    @pytest.mark.asyncio
    async def test_generate_tests_with_analysis_fallback(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test test generation with analysis fallback."""
        # Mock analyzer failures with correct method names
        mock_coordinator.doc_analyzer.analyze_project_documentation = AsyncMock(
            return_value=None
        )
        mock_coordinator.source_analyzer.analyze_project_source_code = AsyncMock(
            return_value=None
        )

        # Mock test generator success
        mock_coordinator.acceptance_test_generator.run_complete_workflow = AsyncMock(
            return_value=sample_workflow_result
        )

        result = await mock_coordinator._generate_tests()

        # Should succeed with fallback analysis
        assert result["success"] is True

        # Verify analyzers were called with correct method names
        mock_coordinator.doc_analyzer.analyze_project_documentation.assert_called_once()
        mock_coordinator.source_analyzer.analyze_project_source_code.assert_called_once()

        # Verify workflow was called with minimal analysis
        workflow_call_args = (
            mock_coordinator.acceptance_test_generator.run_complete_workflow.call_args
        )
        assert workflow_call_args[1]["doc_analysis"] is not None
        assert workflow_call_args[1]["source_analysis"] is not None

    @pytest.mark.asyncio
    async def test_generate_tests_workflow_failure(self, mock_coordinator):
        """Test handling of workflow failure."""
        # Mock analyzers with correct method names
        mock_coordinator.doc_analyzer.analyze_project_documentation = AsyncMock(
            return_value=Mock()
        )
        mock_coordinator.source_analyzer.analyze_project_source_code = AsyncMock(
            return_value=Mock()
        )

        # Mock workflow failure
        failed_result = WorkflowExecutionResult(
            workflow_id="failed-workflow",
            success=False,
            generation_result=None,
            docker_compose_path=None,
            test_image_id=None,
            execution_logs=[],
            error_messages=["Test generation failed", "Docker error"],
            execution_time_seconds=10.0,
            timestamp="2024-01-01T12:00:00Z",
        )

        mock_coordinator.acceptance_test_generator.run_complete_workflow = AsyncMock(
            return_value=failed_result
        )

        result = await mock_coordinator._generate_tests()

        assert result["success"] is False
        assert "Test generation workflow failed" in result["error"]

    @pytest.mark.asyncio
    async def test_run_baseline_tests_success(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test successful baseline test execution."""
        # Setup workflow state with test generation results
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "workflow_result": sample_workflow_result
        }

        # Mock successful test execution
        execution_results = {
            "test_execution_success": True,
            "test_output": "===== 10 passed in 5.2s =====",
            "test_exit_code": 0,
        }

        mock_coordinator.acceptance_test_generator.execute_test_workflow = AsyncMock(
            return_value=execution_results
        )

        result = await mock_coordinator._run_baseline_tests()

        assert result["success"] is True
        assert result["baseline_established"] is True
        assert result["test_results"]["passed"] == 10
        assert result["test_results"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_run_baseline_tests_no_docker_compose(self, mock_coordinator):
        """Test baseline tests when no Docker Compose is available."""
        # Mock workflow result without Docker Compose
        workflow_result_no_docker = WorkflowExecutionResult(
            workflow_id="test-workflow-no-docker",
            success=True,
            generation_result=Mock(),
            docker_compose_path=None,  # No Docker Compose
            test_image_id=None,
            execution_logs=[],
            error_messages=[],
            execution_time_seconds=10.0,
            timestamp="2024-01-01T12:00:00Z",
        )

        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "workflow_result": workflow_result_no_docker
        }

        result = await mock_coordinator._run_baseline_tests()

        assert result["success"] is True
        assert result["baseline_established"] is False
        assert "Docker tests not executed" in result["test_results"]["note"]

    @pytest.mark.asyncio
    async def test_validate_migration_success(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test successful migration validation."""
        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "workflow_result": sample_workflow_result
        }

        # Setup baseline results
        mock_coordinator.workflow_state.steps["run_baseline_tests"] = Mock()
        mock_coordinator.workflow_state.steps["run_baseline_tests"].result = {
            "test_results": {"passed": 10, "failed": 0}
        }

        # Mock successful validation
        execution_results = {
            "test_execution_success": True,
            "test_output": "===== 10 passed in 6.1s =====",
            "test_exit_code": 0,
        }

        mock_coordinator.acceptance_test_generator.execute_test_workflow = AsyncMock(
            return_value=execution_results
        )

        result = await mock_coordinator._validate_migration()

        assert result["success"] is True
        assert result["validation_complete"] is True
        assert result["test_results"]["passed"] == 10
        assert result["test_results"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_validate_migration_with_failures(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test migration validation with test failures."""
        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "workflow_result": sample_workflow_result
        }

        # Setup baseline results
        mock_coordinator.workflow_state.steps["run_baseline_tests"] = Mock()
        mock_coordinator.workflow_state.steps["run_baseline_tests"].result = {
            "test_results": {"passed": 10, "failed": 0}
        }

        # Mock validation with some failures (this triggers logic branch where failed > 0)
        execution_results = {
            "test_execution_success": True,
            "test_output": "===== 7 passed, 3 failed in 8.3s =====",
            "test_exit_code": 0,  # pytest completed but some tests failed
        }

        mock_coordinator.acceptance_test_generator.execute_test_workflow = AsyncMock(
            return_value=execution_results
        )

        result = await mock_coordinator._validate_migration()

        assert result["success"] is False  # Too many failures compared to baseline
        assert result["validation_complete"] is True
        assert result["test_results"]["passed"] == 7
        assert result["test_results"]["failed"] == 3
        assert len(result["test_results"]["failures"]) == 3

    @pytest.mark.asyncio
    async def test_validate_migration_baseline_comparison(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test validation with baseline comparison."""
        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "workflow_result": sample_workflow_result
        }

        # Setup baseline with 10 passed tests
        mock_coordinator.workflow_state.steps["run_baseline_tests"] = Mock()
        mock_coordinator.workflow_state.steps["run_baseline_tests"].result = {
            "test_results": {"passed": 10, "failed": 0}
        }

        # Mock validation with 9 passed tests (within 90% threshold)
        execution_results = {
            "test_execution_success": True,
            "test_output": "===== 9 passed, 1 failed in 7.2s =====",
            "test_exit_code": 0,
        }

        mock_coordinator.acceptance_test_generator.execute_test_workflow = AsyncMock(
            return_value=execution_results
        )

        result = await mock_coordinator._validate_migration()

        # Should fail because we have failures
        assert result["success"] is False
        baseline_comparison = result["test_results"]["baseline_comparison"]
        assert baseline_comparison["baseline_passed"] == 10
        assert baseline_comparison["current_passed"] == 7  # Simulated failed tests
        assert baseline_comparison["acceptable_threshold"] == 9

    @pytest.mark.asyncio
    async def test_validate_migration_execution_error(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test validation with Docker execution error."""
        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "workflow_result": sample_workflow_result
        }

        # Mock execution error
        mock_coordinator.acceptance_test_generator.execute_test_workflow = AsyncMock(
            side_effect=RuntimeError("Docker container failed to start")
        )

        result = await mock_coordinator._validate_migration()

        assert result["success"] is False
        assert "Docker validation execution failed" in result["error"]
        assert result["validation_complete"] is False
