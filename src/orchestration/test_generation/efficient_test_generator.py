"""Efficient test generation pipeline focusing on library usage points."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

from ..mcp_manager import MCPManager, MCPServerType
from .library_usage_analyzer import LibraryUsageAnalyzer, LibraryUsageSummary
from .test_discovery import TestDiscoveryAnalyzer, TestDiscoveryResult
from .focused_test_generator import FocusedTestGenerator, TestGenerationResult
from ..config import LLMConfig


@dataclass
class EfficientTestGenerationResult:
    """Results of the efficient test generation pipeline."""

    library_usage_summary: LibraryUsageSummary
    test_discovery_result: TestDiscoveryResult
    focused_test_result: TestGenerationResult
    total_execution_time: float
    pipeline_success: bool
    output_directory: str


class EfficientTestGenerator:
    """Main pipeline for efficient test generation based on library usage analysis."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize the efficient test generator pipeline.

        Args:
            project_root: Root directory of the project
            mcp_manager: MCP manager for operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.efficient_test_generator")

        # Initialize components
        self.usage_analyzer = LibraryUsageAnalyzer(project_root, mcp_manager)
        self.test_discovery = TestDiscoveryAnalyzer(project_root, mcp_manager)
        self.focused_generator = FocusedTestGenerator(project_root, mcp_manager)

    async def generate_efficient_tests(
        self,
        target_library: str,
        llm_config: LLMConfig,
        output_dir: Optional[str] = None,
    ) -> EfficientTestGenerationResult:
        """Run the complete efficient test generation pipeline.

        Args:
            target_library: Library to analyze and generate tests for
            llm_config: LLM configuration for test generation
            output_dir: Optional output directory for generated tests

        Returns:
            Complete pipeline results
        """
        start_time = time.time()

        self.logger.info(
            f"Starting efficient test generation pipeline for {target_library}"
        )

        try:
            # Step 1: Analyze library usage across the project
            self.logger.info("Step 1: Analyzing library usage with Python AST...")
            library_usage = await self.usage_analyzer.analyze_library_usage(
                target_library
            )

            if library_usage.total_usages == 0:
                self.logger.warning(
                    f"No usage of {target_library} found in the project"
                )
                return self._create_empty_result(
                    target_library, time.time() - start_time
                )

            self.logger.info(
                f"Found {library_usage.total_usages} usages in {len(library_usage.affected_files)} files"
            )

            # Step 2: Discover existing tests that cover library usage points
            self.logger.info("Step 2: Discovering existing test coverage...")
            test_discovery = await self.test_discovery.discover_test_coverage(
                library_usage, target_library
            )

            self.logger.info(
                f"Test coverage: {len(test_discovery.usage_coverage)}/{library_usage.total_usages} "
                f"usages covered ({test_discovery.coverage_percentage:.1f}%)"
            )

            # Step 3: Generate focused tests for uncovered usage points
            self.logger.info("Step 3: Generating focused unit tests...")
            focused_tests = await self.focused_generator.generate_focused_tests(
                library_usage, test_discovery, target_library, llm_config
            )

            self.logger.info(
                f"Generated {focused_tests.tests_generated} new focused tests"
            )

            # Step 4: Export generated tests
            if output_dir is None:
                output_dir = str(self.project_root / "generated_library_tests")

            self.logger.info("Step 4: Exporting generated tests...")
            final_output_dir = await self.focused_generator.export_generated_tests(
                focused_tests, output_dir
            )

            # Step 5: Run tests to validate they work (optional)
            if focused_tests.tests_generated > 0:
                await self._validate_generated_tests(final_output_dir)

            execution_time = time.time() - start_time

            result = EfficientTestGenerationResult(
                library_usage_summary=library_usage,
                test_discovery_result=test_discovery,
                focused_test_result=focused_tests,
                total_execution_time=execution_time,
                pipeline_success=True,
                output_directory=final_output_dir,
            )

            self.logger.info(
                f"Efficient test generation completed successfully in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Efficient test generation pipeline failed: {e}")

            return EfficientTestGenerationResult(
                library_usage_summary=LibraryUsageSummary(target_library, 0),
                test_discovery_result=TestDiscoveryResult(0),
                focused_test_result=TestGenerationResult([], target_library, 0, 0, 0.0),
                total_execution_time=execution_time,
                pipeline_success=False,
                output_directory=output_dir
                or str(self.project_root / "generated_library_tests"),
            )

    def _create_empty_result(
        self, target_library: str, execution_time: float
    ) -> EfficientTestGenerationResult:
        """Create an empty result when no library usage is found."""
        return EfficientTestGenerationResult(
            library_usage_summary=LibraryUsageSummary(target_library, 0),
            test_discovery_result=TestDiscoveryResult(0),
            focused_test_result=TestGenerationResult([], target_library, 0, 0, 0.0),
            total_execution_time=execution_time,
            pipeline_success=True,
            output_directory=str(self.project_root / "generated_library_tests"),
        )

    async def _validate_generated_tests(self, test_dir: str) -> bool:
        """Validate that generated tests can be executed with pytest.

        Args:
            test_dir: Directory containing generated tests

        Returns:
            True if tests pass validation, False otherwise
        """
        if not self.mcp_manager.is_server_available(MCPServerType.TESTING):
            self.logger.info(
                "Testing MCP server not available, skipping test validation"
            )
            return True

        try:
            self.logger.info("Validating generated tests with pytest...")

            result = await self.mcp_manager.call_tool(
                MCPServerType.TESTING,
                "run_tests",
                {
                    "test_path": test_dir,
                    "collect_only": True,  # Only check if tests can be collected
                    "verbose": True,
                },
            )

            if result and result.get("success"):
                self.logger.info("Generated tests passed validation")
                return True
            else:
                self.logger.warning(f"Generated tests failed validation: {result}")
                return False

        except Exception as e:
            self.logger.warning(f"Error validating generated tests: {e}")
            return False

    def get_generation_summary(
        self, result: EfficientTestGenerationResult
    ) -> Dict[str, Any]:
        """Get a summary of the generation results.

        Args:
            result: Pipeline results

        Returns:
            Summary dictionary
        """
        if not result.pipeline_success:
            return {
                "success": False,
                "error": "Pipeline failed",
                "execution_time": result.total_execution_time,
            }

        return {
            "success": True,
            "target_library": result.library_usage_summary.target_library,
            "execution_time": result.total_execution_time,
            "library_usage": {
                "total_usages": result.library_usage_summary.total_usages,
                "affected_files": len(result.library_usage_summary.affected_files),
                "imported_modules": len(result.library_usage_summary.imported_modules),
                "used_functions": len(result.library_usage_summary.used_functions),
                "used_classes": len(result.library_usage_summary.used_classes),
            },
            "existing_test_coverage": {
                "total_tests_found": result.test_discovery_result.total_tests_found,
                "relevant_tests": len(result.test_discovery_result.relevant_tests),
                "coverage_percentage": result.test_discovery_result.coverage_percentage,
                "covered_usages": len(result.test_discovery_result.usage_coverage),
                "uncovered_usages": len(result.test_discovery_result.uncovered_usages),
            },
            "generated_tests": {
                "tests_generated": result.focused_test_result.tests_generated,
                "success_rate": result.focused_test_result.generation_success_rate,
                "total_usage_points_targeted": result.focused_test_result.total_usage_points,
            },
            "output_directory": result.output_directory,
        }
