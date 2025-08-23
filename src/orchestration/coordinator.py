"""High-level workflow orchestration coordinator."""

import asyncio
import logging
import re
from typing import Dict, Any
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import LangChainException

from .mcp_manager import MCPManager, MCPServerType
from .test_generation import TestCoverageSelector
from .config import LLMConfig, MigrationConfig
from .test_running.llm_test_runner import LLMTestRunner, UnrecoverableTestRunnerError


class RateLimitError(Exception):
    """Exception raised when API rate limits are hit."""

    pass


class DiversificationCoordinator:
    """Main coordinator for the diversification workflow."""

    def __init__(
        self,
        project_path: str,
        source_library: str,
        target_library: str,
        llm_config: LLMConfig,
        migration_config: "MigrationConfig",
    ):
        """Initialize the diversification coordinator.

        Args:
            project_path: Path to the project to diversify
            source_library: Library to migrate from
            target_library: Library to migrate to
            llm_config: LLM configuration to use
            migration_config: Migration configuration including test_path
        """
        self.project_path = Path(project_path).resolve()
        self.source_library = source_library
        self.target_library = target_library
        self.llm_config = llm_config
        self.migration_config = migration_config

        # Initialize components
        self.mcp_manager = MCPManager(project_root=str(self.project_path))

        # Initialize test coverage selector
        self.test_coverage_selector = TestCoverageSelector(
            str(self.project_path),
            self.mcp_manager,
            (
                self.migration_config.test_paths
                if self.migration_config.test_paths
                else ["tests/"]
            ),
        )

        self.logger = logging.getLogger("diversifier.coordinator")

        # Store results from each step
        self.step_results: Dict[str, Dict[str, Any]] = {}

    async def execute_workflow(self) -> bool:
        """Execute the complete diversification workflow.

        Returns:
            True if workflow completed successfully
        """
        self.logger.info(
            f"Starting diversification workflow: {self.source_library} -> {self.target_library}"
        )
        self.logger.info(f"Project path: {self.project_path}")

        try:
            # Execute all steps in sequence
            steps = [
                ("initialize_environment", self._initialize_environment),
                ("select_tests", self._select_tests),
                ("run_baseline_tests", self._run_baseline_tests),
                ("migrate_code", self._migrate_code),
                ("validate_migration", self._validate_migration),
                ("repair_issues", self._repair_issues),
                ("finalize_migration", self._finalize_migration),
            ]

            for step_name, step_func in steps:
                self.logger.info(f"Executing step: {step_name}")

                try:
                    result = await self._execute_step_with_retry(step_func)

                    if not result.get("success", False):
                        error = result.get(
                            "error", "Step failed without specific error"
                        )
                        self.logger.error(f"Step {step_name} failed: {error}")
                        return False

                    self.step_results[step_name] = result
                    self.logger.info(f"Step {step_name} completed successfully")

                except Exception as e:
                    self.logger.error(f"Step {step_name} raised exception: {e}")
                    return False

            self.logger.info("Diversification workflow completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return False
        finally:
            await self._cleanup()

    async def _execute_step_with_retry(self, step_func) -> Dict[str, Any]:
        """Execute a step with retry only for rate limit errors.

        Args:
            step_func: Step function to execute

        Returns:
            Step result dictionary
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                return await step_func()
            except (RateLimitError, LangChainException) as e:
                if "rate" in str(e).lower() or "limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        self.logger.warning(
                            f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                raise
            except Exception:
                # Fail immediately for non-rate-limit errors
                raise

        # This should never be reached due to the raise statements above
        return {"success": False, "error": "Max retries exceeded"}

    async def _initialize_environment(self) -> Dict[str, Any]:
        """Initialize MCP servers and agents."""
        try:
            # Validate LLM configuration first
            if not self._validate_llm_config():
                return {"success": False, "error": "Invalid LLM configuration"}

            # Initialize MCP servers
            server_results = await self.mcp_manager.initialize_all_servers()

            # Check if filesystem server is available (required)
            if not server_results.get(MCPServerType.FILESYSTEM, False):
                return {
                    "success": False,
                    "error": "Filesystem MCP server failed to initialize",
                }

            # Log server status
            available_servers = [
                server.value for server, success in server_results.items() if success
            ]
            self.logger.info(f"Available MCP servers: {available_servers}")

            return {
                "success": True,
                "mcp_servers": server_results,
                "available_servers": available_servers,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _validate_llm_config(self) -> bool:
        """Validate LLM configuration by attempting to initialize a chat model."""
        try:
            model_id = (
                f"{self.llm_config.provider.lower()}:{self.llm_config.model_name}"
            )

            kwargs: Dict[str, Any] = {
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
            }
            kwargs.update(self.llm_config.additional_params)

            init_chat_model(model=model_id, **kwargs)

            self.logger.info(
                f"✅ LLM configuration validated for {self.llm_config.provider}:{self.llm_config.model_name}"
            )
            return True

        except Exception as e:
            self.logger.error("❌ Error: Invalid LLM configuration")
            self.logger.error(f"Provider: {self.llm_config.provider}")
            self.logger.error(f"Model: {self.llm_config.model_name}")
            self.logger.error(f"Error: {e}")
            return False

    async def _select_tests(self) -> Dict[str, Any]:
        """Select existing tests that cover library usage based on call graph analysis."""
        try:
            self.logger.info("Starting test coverage selection workflow")

            # Use the test coverage selection pipeline
            selection_result = await self.test_coverage_selector.select_test_coverage(
                target_library=self.source_library,
            )

            if not selection_result.pipeline_success:
                return {
                    "success": False,
                    "error": "Test coverage selection pipeline failed",
                }

            # Get selection summary
            summary = self.test_coverage_selector.get_selection_summary(
                selection_result
            )

            self.logger.info(
                f"Selected {summary['test_coverage']['covered_usages']} covered "
                f"and {summary['test_coverage']['uncovered_usages']} uncovered "
                f"library usages ({summary['test_coverage']['coverage_percentage']:.1f}% coverage)"
            )

            return {
                "success": True,
                "selection_result": selection_result,
                "summary": summary,
                "execution_time": selection_result.total_execution_time,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_baseline_tests(self) -> Dict[str, Any]:
        """Run baseline focused unit tests before migration."""
        try:
            self.logger.info("Running baseline focused unit tests")

            # Get test selection results from previous step
            if "select_tests" not in self.step_results:
                return {
                    "success": False,
                    "error": "No test selection results available",
                }

            selection_result = self.step_results["select_tests"].get("selection_result")
            if not selection_result:
                return {"success": False, "error": "No selection results available"}

            # Extract test functions that cover library usage
            coverage_paths = selection_result.test_discovery_result.coverage_paths
            if not coverage_paths:
                self.logger.info("No test coverage found for the selected library")
                return {
                    "success": True,
                    "test_results": {
                        "tests_executed": 0,
                        "passed": 0,
                        "failed": 0,
                        "note": "No tests cover the selected library usage",
                    },
                }

            # Collect unique test functions to execute
            test_functions = set()
            for path in coverage_paths:
                test_node = path.test_node
                test_spec = f"{test_node.file_path}::{test_node.function_name}"
                test_functions.add(test_spec)

            self.logger.info(
                f"Found {len(test_functions)} unique tests covering library usage"
            )

            # Use LLM-powered test running for intelligent test execution
            return await self._run_tests_with_llm(test_functions)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_tests_with_llm(self, test_functions: set) -> Dict[str, Any]:
        """Run tests using LLM-powered intelligent test runner."""
        try:
            self.logger.info(
                f"Running {len(test_functions)} tests using LLM-powered test runner"
            )

            # Initialize LLM test runner for the target project
            runner = LLMTestRunner(
                project_path=str(self.project_path),
                llm_config=self.llm_config,
                migration_config=self.migration_config,
                mcp_manager=self.mcp_manager,
            )

            # Analyze target project structure
            project_structure = runner.analyze_project_structure()
            self.logger.info(
                f"Analyzed target project: found {len(project_structure['test_directories'])} test directories"
            )

            # Detect target project's development requirements with specific test functions
            dev_requirements = await runner.detect_dev_requirements(
                project_structure, test_functions=list(test_functions)
            )
            self.logger.info(
                f"Detected target project requirements: {dev_requirements['testing_framework']}"
            )

            # Set up target project's test environment
            setup_results = await runner.setup_test_environment(dev_requirements)
            if not setup_results["success"]:
                return {
                    "success": False,
                    "error": f"Failed to setup target project test environment: {setup_results['errors']}",
                    "test_results": {
                        "tests_executed": 0,
                        "passed": 0,
                        "failed": len(test_functions),
                        "selected_tests": list(test_functions),
                        "llm_powered": True,
                    },
                }

            # Execute the tests in target project's environment
            test_results = await runner.run_tests(dev_requirements)

            self.logger.info(
                f"LLM test execution completed: {test_results['summary']['successful_commands']}/{test_results['summary']['total_commands']} commands successful"
            )

            # Convert to expected format
            if test_results["overall_success"]:
                output = (
                    test_results["test_commands_executed"][0]["stdout"]
                    if test_results["test_commands_executed"]
                    else ""
                )
                passed = failed = 0

                # Simple parsing of pytest output
                passed_match = re.search(r"(\d+) passed", output)
                failed_match = re.search(r"(\d+) failed", output)

                if passed_match:
                    passed = int(passed_match.group(1))
                if failed_match:
                    failed = int(failed_match.group(1))

                return {
                    "success": True,
                    "test_results": {
                        "tests_executed": passed + failed,
                        "passed": passed,
                        "failed": failed,
                        "skipped": 0,
                        "duration": 0.0,
                        "selected_tests": list(test_functions),
                        "output": output,
                        "stderr": (
                            test_results["test_commands_executed"][0].get("stderr", "")
                            if test_results["test_commands_executed"]
                            else ""
                        ),
                        "llm_powered": True,
                    },
                }
            else:
                return {
                    "success": False,
                    "error": "LLM test execution failed",
                    "test_results": {
                        "tests_executed": 0,
                        "passed": 0,
                        "failed": len(test_functions),
                        "selected_tests": list(test_functions),
                        "llm_powered": True,
                    },
                }

        except UnrecoverableTestRunnerError as e:
            self.logger.error(f"FATAL: Unrecoverable test runner error: {e.message}")
            raise RuntimeError(
                f"Test runner encountered unrecoverable {e.error_type} error: {e.message}"
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM test execution failed: {e}",
                "test_results": {
                    "tests_executed": 0,
                    "passed": 0,
                    "failed": 0,
                    "selected_tests": list(test_functions),
                    "llm_powered": True,
                },
            }

    async def _migrate_code(self) -> Dict[str, Any]:
        """Migrate from source to target library."""
        self.logger.info("Step migrate_code will come here")
        return {"success": True}

    async def _validate_migration(self) -> Dict[str, Any]:
        """Run focused unit tests to validate migration."""
        self.logger.info("Step validate_migration will come here")
        return {"success": True}

    async def _repair_issues(self) -> Dict[str, Any]:
        """Repair any issues found during validation."""
        self.logger.info("Step repair_issues will come here")
        return {"success": True}

    async def _finalize_migration(self) -> Dict[str, Any]:
        """Clean up and finalize migration."""
        self.logger.info("Step finalize_migration will come here")
        return {"success": True}

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self.mcp_manager.shutdown_all_servers()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status.

        Returns:
            Workflow status summary
        """
        completed_steps = len(
            [r for r in self.step_results.values() if r.get("success")]
        )
        failed_steps = len(
            [r for r in self.step_results.values() if not r.get("success")]
        )

        return {
            "project_path": str(self.project_path),
            "source_library": self.source_library,
            "target_library": self.target_library,
            "total_steps": 8,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "is_complete": completed_steps == 8,
            "step_results": self.step_results,
        }
