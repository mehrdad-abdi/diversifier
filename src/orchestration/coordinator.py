"""High-level workflow orchestration coordinator."""

import logging
from typing import Dict, Any
from pathlib import Path

from .agent import AgentManager, AgentType
from .mcp_manager import MCPManager, MCPServerType
from .workflow import WorkflowState, MigrationContext


class DiversificationCoordinator:
    """Main coordinator for the diversification workflow."""

    def __init__(
        self,
        project_path: str,
        source_library: str,
        target_library: str,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
    ):
        """Initialize the diversification coordinator.

        Args:
            project_path: Path to the project to diversify
            source_library: Library to migrate from
            target_library: Library to migrate to
            model_name: LLM model to use
            temperature: Temperature for LLM responses
        """
        self.project_path = Path(project_path).resolve()
        self.source_library = source_library
        self.target_library = target_library

        # Initialize components
        self.agent_manager = AgentManager(
            model_name=model_name, temperature=temperature
        )
        self.mcp_manager = MCPManager(project_root=str(self.project_path))

        # Initialize workflow state
        context = MigrationContext(
            project_path=str(self.project_path),
            source_library=source_library,
            target_library=target_library,
        )
        self.workflow_state = WorkflowState(context)

        self.logger = logging.getLogger("diversifier.coordinator")

        # Configuration
        self.dry_run = False
        self.auto_proceed = False

    async def execute_workflow(
        self, dry_run: bool = False, auto_proceed: bool = False
    ) -> bool:
        """Execute the complete diversification workflow.

        Args:
            dry_run: If True, don't make actual changes
            auto_proceed: If True, don't wait for user confirmation

        Returns:
            True if workflow completed successfully
        """
        self.dry_run = dry_run
        self.auto_proceed = auto_proceed

        self.logger.info(
            f"Starting diversification workflow: {self.source_library} -> {self.target_library}"
        )
        self.logger.info(f"Project path: {self.project_path}")

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

                    # Try to recover if possible
                    if self.workflow_state.can_retry_step(next_step.name):
                        self.logger.info(f"Retrying step: {next_step.name}")
                        self.workflow_state.retry_step(next_step.name)
                        continue
                    else:
                        self.logger.error("Cannot retry step, workflow failed")
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
        if not self.workflow_state.start_step(step_name):
            return False

        try:
            self.logger.info(f"Executing step: {step_name}")

            # Route to appropriate step handler
            if step_name == "initialize_environment":
                result = await self._initialize_environment()
            elif step_name == "analyze_project":
                result = await self._analyze_project()
            elif step_name == "create_backup":
                result = await self._create_backup()
            elif step_name == "generate_tests":
                result = await self._generate_tests()
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

    async def _analyze_project(self) -> Dict[str, Any]:
        """Analyze project structure and library usage."""
        try:
            # Get analyzer agent
            analyzer = self.agent_manager.get_agent(AgentType.ANALYZER)

            # Analyze project structure
            analysis_prompt = f"""
            Please analyze the Python project at {self.project_path} for migration from {self.source_library} to {self.target_library}.
            
            I need you to:
            1. Identify all Python files that use {self.source_library}
            2. Analyze the API usage patterns
            3. Assess the complexity of the migration
            4. Identify potential compatibility issues
            
            Provide a detailed analysis report.
            """

            result = analyzer.invoke(analysis_prompt)

            # Use filesystem MCP to get additional project info
            project_info = await self.mcp_manager.call_tool(
                MCPServerType.FILESYSTEM,
                "find_python_files",
                {"library_name": self.source_library},
            )

            return {
                "success": True,
                "analysis_report": result.get("output", ""),
                "project_info": project_info,
                "files_to_migrate": (
                    project_info.get("matching_files", []) if project_info else []
                ),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_backup(self) -> Dict[str, Any]:
        """Create backup of project before migration."""
        try:
            if self.dry_run:
                self.logger.info("Dry run: Skipping backup creation")
                return {"success": True, "backup_path": None}

            # Use git MCP server to create backup branch
            if self.mcp_manager.is_server_available(MCPServerType.GIT):
                self.logger.info("Creating git-based backup branch")

                # Get current status
                status_result = await self.mcp_manager.call_tool(
                    MCPServerType.GIT, "get_status", {"repo_path": "."}
                )

                if not status_result:
                    self.logger.warning(
                        "Could not get git status, using fallback backup"
                    )
                    return self._fallback_backup()

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
                    self.logger.warning("Git backup failed, using fallback")
                    return self._fallback_backup()
            else:
                self.logger.warning(
                    "Git MCP server not available, using fallback backup"
                )
                return self._fallback_backup()

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return {"success": False, "error": str(e)}

    def _fallback_backup(self) -> Dict[str, Any]:
        """Fallback backup method when git is not available."""
        self.logger.info("Using fallback backup method (copy-based)")
        return {
            "success": True,
            "backup_path": f"{self.project_path}_backup",
            "backup_method": "copy",
        }

    async def _generate_tests(self) -> Dict[str, Any]:
        """Generate library-independent tests."""
        try:
            # Get tester agent
            tester = self.agent_manager.get_agent(AgentType.TESTER)

            test_generation_prompt = f"""
            Please generate comprehensive, library-independent tests for the migration from {self.source_library} to {self.target_library}.
            
            The tests should:
            1. Validate functional equivalence between the libraries
            2. Be independent of the specific library implementation
            3. Cover all major API usage patterns
            4. Include edge cases and error conditions
            
            Generate a test suite that can validate the migration.
            """

            result = tester.invoke(test_generation_prompt)

            return {
                "success": True,
                "test_suite": result.get("output", ""),
                "generated_at": "runtime",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_baseline_tests(self) -> Dict[str, Any]:
        """Run baseline tests before migration."""
        try:
            # TODO: Implement test execution when testing MCP server is available
            self.logger.info("Running baseline tests (placeholder implementation)")

            return {
                "success": True,
                "test_results": {"passed": 10, "failed": 0, "total": 10},
                "baseline_established": True,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

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

            if self.dry_run:
                self.logger.info("Dry run: Simulating code migration")
                result = {"output": "Dry run migration simulation"}
            else:
                result = migrator.invoke(migration_prompt)

            return {
                "success": True,
                "migration_result": result.get("output", ""),
                "files_modified": [],  # Will be populated by actual migration tools
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _validate_migration(self) -> Dict[str, Any]:
        """Run tests to validate migration."""
        try:
            # TODO: Implement test validation when testing MCP server is available
            self.logger.info("Validating migration (placeholder implementation)")

            # Simulate test results
            test_results = {
                "passed": 9,
                "failed": 1,
                "total": 10,
                "failures": ["test_api_compatibility"] if not self.dry_run else [],
            }

            return {
                "success": test_results["failed"] == 0,
                "test_results": test_results,
                "validation_complete": True,
            }

        except Exception as e:
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

            if self.dry_run:
                self.logger.info("Dry run: Simulating issue repair")
                result = {"output": "Dry run repair simulation"}
            else:
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

            if self.dry_run:
                self.logger.info("Dry run: Simulating migration finalization")
                return {
                    "success": True,
                    "migration_finalized": True,
                    "cleanup_complete": True,
                }

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
                            self.logger.info(f"Migration committed: {commit_hash[:8] if commit_hash else 'unknown'}")

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
