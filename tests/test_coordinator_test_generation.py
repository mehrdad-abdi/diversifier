"""Tests for coordinator test generation workflow integration."""

import os
import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.orchestration.config import LLMConfig
from src.orchestration.coordinator import DiversificationCoordinator
from src.orchestration.test_generation import (
    EfficientTestGenerationResult,
    LibraryUsageSummary,
    TestDiscoveryResult,
    TestGenerationResult,
)

# Keep these for legacy test fixtures
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class AcceptanceTestSuite:
    name: str
    description: str
    test_file_content: str
    docker_compose_content: Optional[str] = None


@dataclass
class AcceptanceTestGenerationResult:
    test_suites: List[AcceptanceTestSuite]
    test_scenarios: List[Any]
    docker_configuration: Dict[str, Any]
    test_dependencies: List[str]
    coverage_analysis: Dict[str, Any]
    generation_confidence: float


@dataclass
class WorkflowExecutionResult:
    workflow_id: str
    success: bool
    generation_result: Optional[AcceptanceTestGenerationResult]
    docker_compose_path: Optional[str]
    test_image_id: Optional[str]
    execution_logs: List[str]
    error_messages: List[str]
    execution_time_seconds: float
    timestamp: str


class TestCoordinatorTestGeneration:
    """Test coordinator's Docker-based test generation integration."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a coordinator with mocked dependencies."""
        with (
            patch("src.orchestration.coordinator.AgentManager"),
            patch("src.orchestration.coordinator.MCPManager"),
            patch("src.orchestration.test_generation.EfficientTestGenerator"),
            patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=False),
        ):
            mock_llm_config = LLMConfig(
                provider="anthropic",
                model_name="claude-3-5-sonnet-20241022",
                api_key_env_var="TEST_API_KEY",
            )
            coordinator = DiversificationCoordinator(
                project_path="/test/project",
                source_library="requests",
                target_library="httpx",
                llm_config=mock_llm_config,
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
        """Test successful efficient test generation."""
        # Mock efficient test generation result
        mock_generation_result = EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary("requests", 5),
            test_discovery_result=TestDiscoveryResult(10),
            focused_test_result=TestGenerationResult([], "requests", 5, 3, 0.6),
            total_execution_time=2.5,
            pipeline_success=True,
            output_directory="/test/project/generated_library_tests",
        )

        # Mock efficient test generator
        mock_coordinator.efficient_test_generator.generate_efficient_tests = AsyncMock(
            return_value=mock_generation_result
        )

        # Mock get_generation_summary
        mock_summary = {
            "success": True,
            "target_library": "requests",
            "execution_time": 2.5,
            "library_usage": {"total_usages": 5, "affected_files": 3},
            "generated_tests": {"tests_generated": 3, "success_rate": 0.6},
        }
        mock_coordinator.efficient_test_generator.get_generation_summary = Mock(
            return_value=mock_summary
        )

        result = await mock_coordinator._generate_tests()

        assert result["success"] is True
        assert result["summary"]["generated_tests"]["tests_generated"] == 3
        assert result["summary"]["library_usage"]["total_usages"] == 5
        assert "generated_library_tests" in result["output_directory"]

    @pytest.mark.asyncio
    async def test_generate_tests_pipeline_failure(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test test generation with pipeline failure."""
        # Mock pipeline failure
        mock_generation_result = EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary("requests", 0),
            test_discovery_result=TestDiscoveryResult(0),
            focused_test_result=TestGenerationResult([], "requests", 0, 0, 0.0),
            total_execution_time=1.0,
            pipeline_success=False,  # Pipeline failed
            output_directory="/test/project/generated_library_tests",
        )

        # Mock efficient test generator
        mock_coordinator.efficient_test_generator.generate_efficient_tests = AsyncMock(
            return_value=mock_generation_result
        )

        result = await mock_coordinator._generate_tests()

        # Should fail due to pipeline failure
        assert result["success"] is False
        assert result["error"] == "Efficient test generation pipeline failed"

    @pytest.mark.asyncio
    async def test_generate_tests_exception(self, mock_coordinator):
        """Test handling of workflow exception."""
        # Mock exception during test generation
        mock_coordinator.efficient_test_generator.generate_efficient_tests = AsyncMock(
            side_effect=Exception("Test generation error")
        )

        result = await mock_coordinator._generate_tests()

        assert result["success"] is False
        assert "Test generation error" in result["error"]

    @pytest.mark.asyncio
    async def test_run_baseline_tests_success(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test successful baseline test execution."""
        # Mock generation result with tests
        mock_generation_result = EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary("requests", 5),
            test_discovery_result=TestDiscoveryResult(10),
            focused_test_result=TestGenerationResult([], "requests", 5, 3, 0.6),
            total_execution_time=2.5,
            pipeline_success=True,
            output_directory="/test/project/generated_library_tests",
        )

        # Setup workflow state with test generation results
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "generation_result": mock_generation_result
        }

        # Mock MCP manager to be available
        mock_coordinator.mcp_manager.is_server_available = Mock(return_value=True)

        # Mock successful test execution
        test_results = {
            "success": True,
            "passed": 10,
            "failed": 0,
            "exit_code": 0,
            "output": "===== 10 passed in 5.2s =====",
        }

        mock_coordinator.mcp_manager.call_tool = AsyncMock(return_value=test_results)

        result = await mock_coordinator._run_baseline_tests()

        assert result["success"] is True
        assert result["baseline_established"] is True
        assert result["test_results"]["passed"] == 10
        assert result["test_results"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_run_baseline_tests_no_tests_generated(self, mock_coordinator):
        """Test baseline tests when no tests were generated."""
        # Mock generation result with no tests
        mock_generation_result = EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary("requests", 0),
            test_discovery_result=TestDiscoveryResult(0),
            focused_test_result=TestGenerationResult(
                [], "requests", 0, 0, 0.0
            ),  # No tests generated
            total_execution_time=1.0,
            pipeline_success=True,
            output_directory="/test/project/generated_library_tests",
        )

        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "generation_result": mock_generation_result
        }

        result = await mock_coordinator._run_baseline_tests()

        assert result["success"] is True
        assert result["baseline_established"] is False
        assert "No tests available for execution" in result["test_results"]["note"]

    @pytest.mark.asyncio
    async def test_validate_migration_success(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test successful migration validation."""
        # Mock generation result with tests
        mock_generation_result = EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary("requests", 5),
            test_discovery_result=TestDiscoveryResult(10),
            focused_test_result=TestGenerationResult([], "requests", 5, 3, 0.6),
            total_execution_time=2.5,
            pipeline_success=True,
            output_directory="/test/project/generated_library_tests",
        )

        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "generation_result": mock_generation_result
        }

        # Setup baseline results
        mock_coordinator.workflow_state.steps["run_baseline_tests"] = Mock()
        mock_coordinator.workflow_state.steps["run_baseline_tests"].result = {
            "test_results": {"passed": 10, "failed": 0}
        }

        # Mock MCP manager and successful validation
        mock_coordinator.mcp_manager.is_server_available = Mock(return_value=True)

        test_results = {
            "success": True,
            "passed": 10,
            "failed": 0,
            "exit_code": 0,
            "output": "===== 10 passed in 6.1s =====",
        }

        mock_coordinator.mcp_manager.call_tool = AsyncMock(return_value=test_results)

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
        # Mock generation result with tests
        mock_generation_result = EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary("requests", 5),
            test_discovery_result=TestDiscoveryResult(10),
            focused_test_result=TestGenerationResult([], "requests", 5, 3, 0.6),
            total_execution_time=2.5,
            pipeline_success=True,
            output_directory="/test/project/generated_library_tests",
        )

        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "generation_result": mock_generation_result
        }

        # Setup baseline results
        mock_coordinator.workflow_state.steps["run_baseline_tests"] = Mock()
        mock_coordinator.workflow_state.steps["run_baseline_tests"].result = {
            "test_results": {"passed": 10, "failed": 0}
        }

        # Mock MCP manager and validation with failures
        mock_coordinator.mcp_manager.is_server_available = Mock(return_value=True)

        test_results = {
            "success": True,
            "passed": 7,
            "failed": 3,
            "exit_code": 0,
            "output": "===== 7 passed, 3 failed in 8.3s =====",
        }

        mock_coordinator.mcp_manager.call_tool = AsyncMock(return_value=test_results)

        result = await mock_coordinator._validate_migration()

        assert result["success"] is False  # Too many failures compared to baseline
        assert result["validation_complete"] is True
        assert result["test_results"]["passed"] == 7
        assert result["test_results"]["failed"] == 3
        assert len(result["test_results"]["failures"]) == 1  # Placeholder failure

    @pytest.mark.asyncio
    async def test_validate_migration_baseline_comparison(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test validation with baseline comparison."""
        # Mock generation result with tests
        mock_generation_result = EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary("requests", 5),
            test_discovery_result=TestDiscoveryResult(10),
            focused_test_result=TestGenerationResult([], "requests", 5, 3, 0.6),
            total_execution_time=2.5,
            pipeline_success=True,
            output_directory="/test/project/generated_library_tests",
        )

        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "generation_result": mock_generation_result
        }

        # Setup baseline with 10 passed tests
        mock_coordinator.workflow_state.steps["run_baseline_tests"] = Mock()
        mock_coordinator.workflow_state.steps["run_baseline_tests"].result = {
            "test_results": {"passed": 10, "failed": 0}
        }

        # Mock MCP manager and validation with some failures
        mock_coordinator.mcp_manager.is_server_available = Mock(return_value=True)

        test_results = {
            "success": True,
            "passed": 9,
            "failed": 1,
            "exit_code": 0,
            "output": "===== 9 passed, 1 failed in 7.2s =====",
        }

        mock_coordinator.mcp_manager.call_tool = AsyncMock(return_value=test_results)

        result = await mock_coordinator._validate_migration()

        # Should fail because we have failures
        assert result["success"] is False
        baseline_comparison = result["test_results"]["baseline_comparison"]
        assert baseline_comparison["baseline_passed"] == 10
        assert baseline_comparison["current_passed"] == 9
        assert baseline_comparison["acceptable_threshold"] == 9

    @pytest.mark.asyncio
    async def test_validate_migration_execution_error(
        self, mock_coordinator, sample_workflow_result
    ):
        """Test validation with pytest MCP execution error."""
        # Mock generation result with tests
        mock_generation_result = EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary("requests", 5),
            test_discovery_result=TestDiscoveryResult(10),
            focused_test_result=TestGenerationResult([], "requests", 5, 3, 0.6),
            total_execution_time=2.5,
            pipeline_success=True,
            output_directory="/test/project/generated_library_tests",
        )

        # Setup workflow state
        mock_coordinator.workflow_state.steps["generate_tests"] = Mock()
        mock_coordinator.workflow_state.steps["generate_tests"].result = {
            "generation_result": mock_generation_result
        }

        # Mock MCP manager to be available but throw error
        mock_coordinator.mcp_manager.is_server_available = Mock(return_value=True)
        mock_coordinator.mcp_manager.call_tool = AsyncMock(
            side_effect=RuntimeError("pytest execution failed")
        )

        result = await mock_coordinator._validate_migration()

        assert result["success"] is False
        assert "Pytest MCP validation execution failed" in result["error"]
        assert result["validation_complete"] is False
