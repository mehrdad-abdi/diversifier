"""Workflow management for coordinating multiple test generation workflows."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .acceptance_test_generator import (
    AcceptanceTestGenerator,
    WorkflowExecutionResult,
)
from .mcp_manager import MCPManager
from .doc_analyzer import DocumentationAnalysisResult
from .source_code_analyzer import SourceCodeAnalysisResult


@dataclass
class WorkflowConfiguration:
    """Configuration for test generation workflow."""

    project_root: str
    output_directory: str
    docker_registry: Optional[str] = None
    test_image_name: str = "diversifier-tests"
    test_image_tag: str = "latest"
    execute_tests: bool = False
    model_name: str = "gpt-4"
    cleanup_containers: bool = True


class WorkflowManager:
    """Manager for coordinating multiple test generation workflows."""

    def __init__(self, base_config: WorkflowConfiguration):
        """Initialize workflow manager.

        Args:
            base_config: Base configuration for workflows
        """
        self.base_config = base_config
        self.active_workflows: Dict[str, AcceptanceTestGenerator] = {}
        self.completed_workflows: Dict[str, WorkflowExecutionResult] = {}
        self.logger = logging.getLogger("diversifier.workflow_manager")

    async def create_workflow(
        self,
        project_root: str,
        output_directory: str,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new test generation workflow.

        Args:
            project_root: Root directory of the project to test
            output_directory: Directory for test output
            custom_config: Custom configuration overrides

        Returns:
            Workflow ID
        """
        # Create workflow configuration
        config = WorkflowConfiguration(
            project_root=project_root,
            output_directory=output_directory,
            docker_registry=self.base_config.docker_registry,
            test_image_name=self.base_config.test_image_name,
            test_image_tag=self.base_config.test_image_tag,
            execute_tests=self.base_config.execute_tests,
            model_name=self.base_config.model_name,
            cleanup_containers=self.base_config.cleanup_containers,
        )

        # Apply custom configuration
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create MCP manager and test generator
        mcp_manager = MCPManager(project_root)
        generator = AcceptanceTestGenerator(project_root, mcp_manager)
        workflow_id = generator.workflow_id

        self.active_workflows[workflow_id] = generator
        self.logger.info(f"Created workflow {workflow_id}")

        return workflow_id

    async def run_workflow(
        self,
        workflow_id: str,
        doc_analysis: DocumentationAnalysisResult,
        source_analysis: SourceCodeAnalysisResult,
        output_dir: Optional[str] = None,
        execute_tests: Optional[bool] = None,
    ) -> WorkflowExecutionResult:
        """Run a specific workflow.

        Args:
            workflow_id: ID of the workflow to run
            doc_analysis: Documentation analysis results
            source_analysis: Source code analysis results
            output_dir: Optional output directory override
            execute_tests: Optional execute tests override

        Returns:
            Workflow execution results
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        generator = self.active_workflows[workflow_id]

        # Use base config values if not overridden
        if execute_tests is None:
            execute_tests = self.base_config.execute_tests

        result = await generator.run_complete_workflow(
            doc_analysis=doc_analysis,
            source_analysis=source_analysis,
            output_dir=output_dir,
            model_name=self.base_config.model_name,
            execute_tests=execute_tests,
        )

        # Move to completed workflows
        self.completed_workflows[workflow_id] = result
        del self.active_workflows[workflow_id]

        return result

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Workflow status information
        """
        if workflow_id in self.active_workflows:
            generator = self.active_workflows[workflow_id]
            return {
                "status": "active",
                "workflow_id": workflow_id,
                "logs": generator.execution_logs[-10:],  # Last 10 logs
                "errors": generator.error_messages,
            }
        elif workflow_id in self.completed_workflows:
            result = self.completed_workflows[workflow_id]
            return {
                "status": "completed",
                "success": result.success,
                "workflow_id": workflow_id,
                "execution_time": result.execution_time_seconds,
                "timestamp": result.timestamp,
            }
        else:
            return {"status": "not_found", "workflow_id": workflow_id}

    def list_workflows(self) -> Dict[str, List[str]]:
        """List all workflows by status.

        Returns:
            Dictionary mapping status to workflow IDs
        """
        return {
            "active": list(self.active_workflows.keys()),
            "completed": list(self.completed_workflows.keys()),
        }

    async def get_workflow_generator(
        self, workflow_id: str
    ) -> Optional[AcceptanceTestGenerator]:
        """Get the AcceptanceTestGenerator for a specific workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            AcceptanceTestGenerator instance or None if not found
        """
        return self.active_workflows.get(workflow_id)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow.

        Args:
            workflow_id: ID of the workflow to cancel

        Returns:
            True if cancelled successfully, False if not found
        """
        if workflow_id in self.active_workflows:
            generator = self.active_workflows[workflow_id]
            generator._log("Workflow cancelled by user")

            # Move to completed with cancelled status
            result = WorkflowExecutionResult(
                workflow_id=workflow_id,
                success=False,
                generation_result=None,
                docker_compose_path=None,
                test_image_id=None,
                execution_logs=generator.execution_logs.copy(),
                error_messages=generator.error_messages + ["Workflow cancelled"],
                execution_time_seconds=0.0,
                timestamp="cancelled",
            )

            self.completed_workflows[workflow_id] = result
            del self.active_workflows[workflow_id]

            self.logger.info(f"Cancelled workflow {workflow_id}")
            return True

        return False
