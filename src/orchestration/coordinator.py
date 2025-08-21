"""High-level workflow orchestration coordinator."""

import logging
import re
from typing import Dict, Any, Optional
from pathlib import Path

from langchain.chat_models import init_chat_model

from .agent import AgentManager, AgentType
from .mcp_manager import MCPManager, MCPServerType
from .workflow import WorkflowState, MigrationContext
from .test_generation import TestCoverageSelector
from .config import LLMConfig, MigrationConfig
from .test_running.llm_test_runner import LLMTestRunner, UnrecoverableTestRunnerError


class DiversificationCoordinator:
    """Main coordinator for the diversification workflow."""

    def __init__(
        self,
        project_path: str,
        source_library: str,
        target_library: str,
        llm_config: Optional[LLMConfig] = None,
        migration_config: Optional["MigrationConfig"] = None,
    ):
        """Initialize the diversification coordinator.

        Args:
            project_path: Path to the project to diversify
            source_library: Library to migrate from
            target_library: Library to migrate to
            llm_config: LLM configuration to use. If None, uses global config.
            migration_config: Migration configuration including test_path. If None, uses defaults.
        """
        self.project_path = Path(project_path).resolve()
        self.source_library = source_library
        self.target_library = target_library
        if llm_config is None:
            raise ValueError(
                "llm_config is required - no default configuration available"
            )
        self.llm_config = llm_config

        # Use provided migration config or create default
        if migration_config is None:
            migration_config = MigrationConfig()
        self.migration_config = migration_config

        # Initialize components
        self.agent_manager = AgentManager(llm_config=self.llm_config)
        self.mcp_manager = MCPManager(project_root=str(self.project_path))

        # Initialize workflow state
        context = MigrationContext(
            project_path=str(self.project_path),
            source_library=source_library,
            target_library=target_library,
        )
        self.workflow_state = WorkflowState(context)

        # Initialize test coverage selection components
        self.test_coverage_selector = TestCoverageSelector(
            str(self.project_path),
            self.mcp_manager,
            (
                self.migration_config.test_paths[0]
                if self.migration_config.test_paths
                else "tests/"
            ),
        )

        self.logger = logging.getLogger("diversifier.coordinator")

        # Configuration
        self.auto_proceed = False

    def _validate_api_key(self) -> bool:
        """Validate LLM configuration by attempting to initialize a chat model.

        Returns:
            True if LLM config is valid, False otherwise
        """
        try:
            # Create the model identifier for init_chat_model
            model_id = (
                f"{self.llm_config.provider.lower()}:{self.llm_config.model_name}"
            )

            # Prepare initialization arguments
            kwargs: Dict[str, Any] = {
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
            }
            # Add additional params
            for key, value in self.llm_config.additional_params.items():
                kwargs[key] = value

            # Try to initialize the LLM - this validates the config without API calls
            init_chat_model(model=model_id, **kwargs)

            self.logger.info(
                f"✅ LLM configuration validated for {self.llm_config.provider}:{self.llm_config.model_name}"
            )
            return True

        except Exception as e:
            print("❌ Error: Invalid LLM configuration")
            print(f"Provider: {self.llm_config.provider}")
            print(f"Model: {self.llm_config.model_name}")
            print("")
            print(f"Error: {e}")

            return False

    async def execute_workflow(self, auto_proceed: bool = False) -> bool:
        """Execute the complete diversification workflow.

        Args:
            auto_proceed: If True, don't wait for user confirmation

        Returns:
            True if workflow completed successfully
        """
        self.auto_proceed = auto_proceed

        self.logger.info(
            f"Starting diversification workflow: {self.source_library} -> {self.target_library}"
        )
        self.logger.info(f"Project path: {self.project_path}")

        # Validate API key before starting workflow
        if not self._validate_api_key():
            return False

        try:
            # Execute workflow steps in order
            while (
                not self.workflow_state.is_workflow_complete()
                and not self.workflow_state.is_workflow_failed()
            ):
                next_step = self.workflow_state.get_next_step()
                if not next_step:
                    break

                success = await self._execute_step(next_step.name)
                if not success:
                    self.logger.error(f"Step {next_step.name} failed")
                    self.logger.error("Workflow failed - no retry mechanism")
                    break

            # Check final result
            if self.workflow_state.is_workflow_complete():
                self.logger.info("Diversification workflow completed successfully")
                return True
            else:
                self.logger.error("Diversification workflow failed")
                return False

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return False
        finally:
            await self.cleanup()

    async def _execute_step(self, step_name: str) -> bool:
        """Execute a specific workflow step.

        Args:
            step_name: Name of step to execute

        Returns:
            True if step completed successfully
        """
        # breakpoint()
        if not self.workflow_state.start_step(step_name):
            return False

        try:
            self.logger.info(f"Executing step: {step_name}")

            # Route to appropriate step handler
            if step_name == "initialize_environment":
                result = await self._initialize_environment()
            elif step_name == "create_backup":
                result = await self._create_backup()
            elif step_name == "select_tests":
                result = await self._select_tests()
            elif step_name == "run_baseline_tests":
                result = await self._run_baseline_tests()
            elif step_name == "migrate_code":
                result = await self._migrate_code()
            elif step_name == "validate_migration":
                result = await self._validate_migration()
            elif step_name == "repair_issues":
                result = await self._repair_issues()
            elif step_name == "finalize_migration":
                result = await self._finalize_migration()
            else:
                raise ValueError(f"Unknown step: {step_name}")

            if result.get("success", False):
                self.workflow_state.complete_step(step_name, result)
                return True
            else:
                error = result.get("error", "Step failed without specific error")
                self.workflow_state.fail_step(step_name, error)
                return False

        except Exception as e:
            self.workflow_state.fail_step(step_name, str(e))
            self.logger.error(f"Step {step_name} raised exception: {e}")
            return False

    async def _initialize_environment(self) -> Dict[str, Any]:
        """Initialize MCP servers and agents."""
        try:
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

    async def _create_backup(self) -> Dict[str, Any]:
        """Create backup of project before migration."""
        try:

            # Use git MCP server to create backup branch
            if self.mcp_manager.is_server_available(MCPServerType.GIT):
                self.logger.info("Creating git-based backup branch")

                # Get current status
                status_result = await self.mcp_manager.call_tool(
                    MCPServerType.GIT, "get_status", {"repo_path": "."}
                )

                if not status_result:
                    self.logger.error(
                        "Could not get git status, backup creation failed"
                    )
                    return {"success": False, "error": "Git status check failed"}

                # Create backup branch
                backup_result = await self.mcp_manager.call_tool(
                    MCPServerType.GIT,
                    "create_temp_branch",
                    {
                        "repo_path": ".",
                        "base_branch": status_result.get("branch", "main"),
                        "prefix": "diversifier-backup",
                    },
                )

                if backup_result and backup_result.get("status") == "success":
                    backup_branch = backup_result.get("temp_branch")
                    self.logger.info(f"Created backup branch: {backup_branch}")

                    return {
                        "success": True,
                        "backup_path": backup_branch,
                        "backup_method": "git_branch",
                        "base_branch": backup_result.get("base_branch"),
                    }
                else:
                    self.logger.error("Git backup failed")
                    return {"success": False, "error": "Git backup creation failed"}
            else:
                self.logger.error(
                    "Git MCP server not available, backup creation failed"
                )
                return {"success": False, "error": "Git MCP server not available"}

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _select_tests(self) -> Dict[str, Any]:
        """Select existing tests that cover library usage based on call graph analysis."""
        try:
            self.logger.info("Starting test coverage selection workflow")

            # Use the test coverage selection pipeline
            selection_result = await self.test_coverage_selector.select_test_coverage(
                target_library=self.source_library,  # Analyze usage of the source library
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
            self.logger.error(f"Test coverage selection failed: {e}")
            return {"success": False, "error": str(e)}

    async def _run_baseline_tests(self) -> Dict[str, Any]:
        """Run baseline focused unit tests before migration."""
        try:
            self.logger.info("Running baseline focused unit tests")

            # Get test selection results from previous step
            generate_step = self.workflow_state.steps.get("select_tests")
            if not generate_step or not generate_step.result:
                return {
                    "success": False,
                    "error": "No test selection results available",
                }

            selection_result = generate_step.result.get("selection_result")
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
                # Format as pytest can understand: file_path::function_name
                test_spec = f"{test_node.file_path}::{test_node.function_name}"
                test_functions.add(test_spec)

            self.logger.info(
                f"Found {len(test_functions)} unique tests covering library usage"
            )

            # Use LLM-powered test running for intelligent test execution
            self.logger.info("Using LLM-powered test runner to execute tests")
            return await self._run_tests_with_llm(test_functions)

        except Exception as e:
            self.logger.error(f"Baseline test execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _run_tests_with_llm(self, test_functions: set) -> Dict[str, Any]:
        """Run tests using LLM-powered intelligent test runner.

        Args:
            test_functions: Set of test specifications to run

        Returns:
            Test results dictionary
        """
        try:
            self.logger.info(
                f"Running {len(test_functions)} tests using LLM-powered test runner"
            )

            # Initialize LLM test runner for the target project
            runner = LLMTestRunner(str(self.project_path), self.migration_config)

            # Analyze target project structure to understand its environment
            project_structure = await runner.analyze_project_structure()
            self.logger.info(
                f"Analyzed target project: found {len(project_structure['test_files'])} test files"
            )

            # Use LLM to detect target project's development requirements
            dev_requirements = await runner.detect_dev_requirements(project_structure)
            self.logger.info(
                f"Detected target project requirements: {dev_requirements['testing_framework']}"
            )

            # Override test commands to run specific test functions instead of full test suite
            focused_requirements = dev_requirements.copy()
            focused_requirements["test_commands"] = [
                f"python -m pytest -v {' '.join(test_functions)}"
            ]
            focused_requirements["analysis"] = (
                "Focused test execution for baseline tests"
            )

            # Set up target project's test environment (install dev dependencies)
            setup_results = await runner.setup_test_environment(focused_requirements)
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
            test_results = await runner.run_tests(focused_requirements)

            self.logger.info(
                f"LLM test execution completed: {test_results['summary']['successful_commands']}/{test_results['summary']['total_commands']} commands successful"
            )

            # Convert to expected format
            if test_results["overall_success"]:
                # Parse the output to extract test counts
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
                        "duration": 0.0,  # Not tracked in simple runner
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
            # Fatal LLM test runner errors - do not retry, exit workflow
            self.logger.error(f"FATAL: Unrecoverable test runner error: {e.message}")
            self.logger.error(f"Error type: {e.error_type}")
            if e.original_error:
                self.logger.error(f"Original error: {e.original_error}")

            # Signal to workflow that this is unrecoverable
            raise RuntimeError(
                f"Test runner encountered unrecoverable {e.error_type} error: {e.message}"
            )

        except Exception as e:
            # Other recoverable exceptions
            self.logger.error(f"LLM test execution failed: {e}")
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
        try:
            # Get migrator agent
            migrator = self.agent_manager.get_agent(AgentType.MIGRATOR)

            migration_prompt = f"""
            Please migrate the Python code from {self.source_library} to {self.target_library}.
            
            Migration requirements:
            1. Update all import statements
            2. Convert API calls to the new library
            3. Handle parameter mapping and structural changes
            4. Maintain functional equivalence
            5. Follow best practices for the target library
            
            Perform the migration while preserving all functionality.
            """

            result = migrator.invoke(migration_prompt)

            return {
                "success": True,
                "migration_result": result.get("output", ""),
                "files_modified": [],  # Will be populated by actual migration tools
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _validate_migration(self) -> Dict[str, Any]:
        """Run focused unit tests to validate migration."""
        try:
            self.logger.info("Validating migration with focused unit tests")

            # Get test selection results from previous step
            generate_step = self.workflow_state.steps.get("select_tests")
            if not generate_step or not generate_step.result:
                return {
                    "success": False,
                    "error": "No test selection results available for validation",
                }

            selection_result = generate_step.result.get("selection_result")
            if not selection_result:
                return {
                    "success": False,
                    "error": "No selection results available for validation",
                }

            # Since we only do test selection now, skip validation
            self.logger.info(
                "Test coverage selection completed - no test generation for validation"
            )
            return {
                "success": True,
                "test_results": {
                    "note": "Test coverage selection only - no tests generated to validate"
                },
            }

        except Exception as e:
            self.logger.error(f"Migration validation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _repair_issues(self) -> Dict[str, Any]:
        """Repair any issues found during validation."""
        try:
            # Check if repair is needed based on validation results
            validation_step = self.workflow_state.steps.get("validate_migration")
            if validation_step and validation_step.result:
                test_results = validation_step.result.get("test_results", {})
                if test_results.get("failed", 0) == 0:
                    return {"success": True, "repairs_needed": False}

            # Get repairer agent
            repairer = self.agent_manager.get_agent(AgentType.REPAIRER)

            repair_prompt = """
            Please analyze and repair the issues found during migration validation.
            
            Issues to address:
            1. Failed tests from validation step
            2. API compatibility problems
            3. Behavioral differences
            
            Apply targeted fixes to resolve the issues while maintaining functional equivalence.
            """

            result = repairer.invoke(repair_prompt)

            return {
                "success": True,
                "repair_result": result.get("output", ""),
                "repairs_applied": 1,
                "attempt": self.workflow_state.context.repair_attempts + 1,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _finalize_migration(self) -> Dict[str, Any]:
        """Clean up and finalize migration."""
        try:
            self.logger.info("Finalizing migration")

            # Use git MCP server for cleanup when available
            if self.mcp_manager.is_server_available(MCPServerType.GIT):
                self.logger.info("Using git MCP server for finalization")

                # Get migration results to see what files were modified
                migration_step = self.workflow_state.steps.get("migrate_code")
                files_modified = []
                if migration_step and migration_step.result:
                    files_modified = migration_step.result.get("files_modified", [])

                # Stage modified files
                if files_modified:
                    stage_result = await self.mcp_manager.call_tool(
                        MCPServerType.GIT,
                        "add_files",
                        {"repo_path": ".", "file_patterns": files_modified},
                    )

                    if stage_result and stage_result.get("status") == "success":
                        self.logger.info(f"Staged {len(files_modified)} modified files")

                        # Commit changes
                        commit_message = f"diversifier: Migrate from {self.source_library} to {self.target_library}"
                        commit_result = await self.mcp_manager.call_tool(
                            MCPServerType.GIT,
                            "commit_changes",
                            {"repo_path": ".", "message": commit_message},
                        )

                        if commit_result and commit_result.get("status") == "success":
                            commit_hash = commit_result.get("commit_hash") or ""
                            self.logger.info(
                                f"Migration committed: {commit_hash[:8] if commit_hash else 'unknown'}"
                            )

                            return {
                                "success": True,
                                "migration_finalized": True,
                                "cleanup_complete": True,
                                "commit_hash": commit_hash or "unknown",
                                "files_committed": len(files_modified),
                            }
                        else:
                            self.logger.warning("Failed to commit changes")
                    else:
                        self.logger.warning("Failed to stage files for commit")
                else:
                    self.logger.info("No modified files to commit")

            else:
                self.logger.info("Git MCP server not available, basic cleanup only")

            return {
                "success": True,
                "migration_finalized": True,
                "cleanup_complete": True,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self.mcp_manager.shutdown_all_servers()
            self.agent_manager.clear_all_memories()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status.

        Returns:
            Workflow status summary
        """
        return self.workflow_state.get_workflow_summary()

    def save_workflow_state(self, file_path: str) -> bool:
        """Save current workflow state to file.

        Args:
            file_path: Path to save state file

        Returns:
            True if saved successfully
        """
        return self.workflow_state.save_state(file_path)
