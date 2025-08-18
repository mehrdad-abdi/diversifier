"""Efficient test discovery pipeline focusing on library usage points."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from ..mcp_manager import MCPManager
from .library_usage_analyzer import LibraryUsageAnalyzer, LibraryUsageSummary
from .call_graph_test_discovery import (
    CallGraphTestDiscoveryAnalyzer,
    CallGraphTestDiscoveryResult,
)


@dataclass
class EfficientTestDiscoveryResult:
    """Results of the efficient test discovery pipeline."""

    library_usage_summary: LibraryUsageSummary
    test_discovery_result: CallGraphTestDiscoveryResult
    total_execution_time: float
    pipeline_success: bool


class EfficientTestGenerator:
    """Main pipeline for efficient test discovery based on library usage analysis."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize the efficient test discovery pipeline.

        Args:
            project_root: Root directory of the project
            mcp_manager: MCP manager for operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.efficient_test_generator")

        # Initialize components
        self.usage_analyzer = LibraryUsageAnalyzer(project_root, mcp_manager)
        self.test_discovery = CallGraphTestDiscoveryAnalyzer(project_root, mcp_manager)

    async def discover_test_coverage(
        self,
        target_library: str,
    ) -> EfficientTestDiscoveryResult:
        """Run the efficient test discovery pipeline.

        Args:
            target_library: Library to analyze test coverage for

        Returns:
            Test discovery results
        """
        start_time = time.time()

        self.logger.info(
            f"Starting efficient test discovery pipeline for {target_library}"
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
            self.logger.info(
                "Step 2: Discovering existing test coverage using call graph analysis..."
            )
            test_discovery = await self.test_discovery.discover_test_coverage(
                library_usage
            )

            self.logger.info(
                f"Test coverage: {len(test_discovery.coverage_paths)}/{library_usage.total_usages} "
                f"usages covered ({test_discovery.coverage_percentage:.1f}%)"
            )

            execution_time = time.time() - start_time

            result = EfficientTestDiscoveryResult(
                library_usage_summary=library_usage,
                test_discovery_result=test_discovery,
                total_execution_time=execution_time,
                pipeline_success=True,
            )

            self.logger.info(
                f"Efficient test discovery completed successfully in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Efficient test discovery pipeline failed: {e}")

            return EfficientTestDiscoveryResult(
                library_usage_summary=LibraryUsageSummary(target_library, 0),
                test_discovery_result=CallGraphTestDiscoveryResult(
                    total_nodes=0,
                    test_nodes=0,
                    library_usage_nodes=0,
                    coverage_paths=[],
                    uncovered_usages=[],
                    coverage_percentage=0.0,
                ),
                total_execution_time=execution_time,
                pipeline_success=False,
            )

    def _create_empty_result(
        self, target_library: str, execution_time: float
    ) -> EfficientTestDiscoveryResult:
        """Create an empty result when no library usage is found."""
        return EfficientTestDiscoveryResult(
            library_usage_summary=LibraryUsageSummary(target_library, 0),
            test_discovery_result=CallGraphTestDiscoveryResult(
                total_nodes=0,
                test_nodes=0,
                library_usage_nodes=0,
                coverage_paths=[],
                uncovered_usages=[],
                coverage_percentage=0.0,
            ),
            total_execution_time=execution_time,
            pipeline_success=True,
        )

    def get_discovery_summary(
        self, result: EfficientTestDiscoveryResult
    ) -> Dict[str, Any]:
        """Get a summary of the discovery results.

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
            "test_coverage": {
                "total_tests_found": result.test_discovery_result.test_nodes,
                "total_nodes": result.test_discovery_result.total_nodes,
                "coverage_percentage": result.test_discovery_result.coverage_percentage,
                "covered_usages": len(result.test_discovery_result.coverage_paths),
                "uncovered_usages": len(result.test_discovery_result.uncovered_usages),
                "coverage_paths": [
                    {
                        "test_function": f"{path.test_node.file_path}::{path.test_node.function_name}",
                        "usage_location": f"{path.library_usage.file_path}::{path.library_usage.function_name}",
                        "call_chain_depth": path.depth,
                    }
                    for path in result.test_discovery_result.coverage_paths
                ],
            },
        }
