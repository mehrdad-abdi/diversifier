"""Workflow state management and orchestration logic."""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import json


class WorkflowStage(Enum):
    """Stages in the diversification workflow."""

    INITIALIZATION = "initialization"
    PROJECT_ANALYSIS = "project_analysis"
    TEST_GENERATION = "test_generation"
    CODE_MIGRATION = "code_migration"
    VALIDATION = "validation"
    REPAIR = "repair"
    COMPLETION = "completion"
    FAILED = "failed"


class WorkflowStatus(Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Represents a single step in the workflow."""

    name: str
    stage: WorkflowStage
    description: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Get step execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def start(self) -> None:
        """Mark step as started."""
        self.status = WorkflowStatus.RUNNING
        self.start_time = datetime.now()

    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as completed."""
        self.status = WorkflowStatus.COMPLETED
        self.end_time = datetime.now()
        if result:
            self.result = result

    def fail(self, error: str) -> None:
        """Mark step as failed."""
        self.status = WorkflowStatus.FAILED
        self.end_time = datetime.now()
        self.error = error


@dataclass
class MigrationContext:
    """Context information for the migration workflow."""

    project_path: str
    source_library: str
    target_library: str
    start_time: datetime = field(default_factory=datetime.now)
    backup_created: bool = False
    git_branch: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    migration_files: List[str] = field(default_factory=list)
    repair_attempts: int = 0
    max_repair_attempts: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "project_path": self.project_path,
            "source_library": self.source_library,
            "target_library": self.target_library,
            "start_time": self.start_time.isoformat(),
            "backup_created": self.backup_created,
            "git_branch": self.git_branch,
            "test_results": self.test_results,
            "migration_files": self.migration_files,
            "repair_attempts": self.repair_attempts,
            "max_repair_attempts": self.max_repair_attempts,
        }


class WorkflowState:
    """Manages the state of the diversification workflow."""

    def __init__(self, context: MigrationContext):
        """Initialize workflow state.

        Args:
            context: Migration context information
        """
        self.context = context
        self.current_stage = WorkflowStage.INITIALIZATION
        self.steps: Dict[str, WorkflowStep] = {}
        self.step_order: List[str] = []

        self.logger = logging.getLogger("diversifier.workflow")

        # Initialize workflow steps
        self._initialize_steps()

    def _initialize_steps(self) -> None:
        """Initialize all workflow steps."""
        steps = [
            WorkflowStep(
                name="initialize_environment",
                stage=WorkflowStage.INITIALIZATION,
                description="Initialize MCP servers and agents",
            ),
            WorkflowStep(
                name="create_backup",
                stage=WorkflowStage.PROJECT_ANALYSIS,
                description="Create backup of project before migration",
                dependencies=["initialize_environment"],
            ),
            WorkflowStep(
                name="select_tests",
                stage=WorkflowStage.TEST_GENERATION,
                description="Select existing tests that cover library usage",
                dependencies=["create_backup"],
            ),
            WorkflowStep(
                name="run_baseline_tests",
                stage=WorkflowStage.TEST_GENERATION,
                description="Run baseline tests before migration",
                dependencies=["select_tests"],
            ),
            WorkflowStep(
                name="migrate_code",
                stage=WorkflowStage.CODE_MIGRATION,
                description="Migrate from source to target library",
                dependencies=["run_baseline_tests"],
            ),
            WorkflowStep(
                name="validate_migration",
                stage=WorkflowStage.VALIDATION,
                description="Run tests to validate migration",
                dependencies=["migrate_code"],
            ),
            WorkflowStep(
                name="repair_issues",
                stage=WorkflowStage.REPAIR,
                description="Repair any issues found during validation",
                dependencies=["validate_migration"],
            ),
            WorkflowStep(
                name="finalize_migration",
                stage=WorkflowStage.COMPLETION,
                description="Clean up and finalize migration",
                dependencies=["repair_issues"],
            ),
        ]

        for step in steps:
            self.steps[step.name] = step
            self.step_order.append(step.name)

    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get the current active step.

        Returns:
            Current workflow step or None
        """
        for step_name in self.step_order:
            step = self.steps[step_name]
            if step.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                return step
        return None

    def get_next_step(self) -> Optional[WorkflowStep]:
        """Get the next step that should be executed.

        Returns:
            Next workflow step or None if workflow complete
        """
        for step_name in self.step_order:
            step = self.steps[step_name]
            if step.status == WorkflowStatus.PENDING:
                # Check if all dependencies are completed
                if self._are_dependencies_completed(step):
                    return step
        return None

    def _are_dependencies_completed(self, step: WorkflowStep) -> bool:
        """Check if all dependencies for a step are completed.

        Args:
            step: Workflow step to check

        Returns:
            True if all dependencies are completed
        """
        for dep_name in step.dependencies:
            if dep_name in self.steps:
                dep_step = self.steps[dep_name]
                if dep_step.status != WorkflowStatus.COMPLETED:
                    return False
        return True

    def start_step(self, step_name: str) -> bool:
        """Start execution of a workflow step.

        Args:
            step_name: Name of step to start

        Returns:
            True if step started successfully
        """
        if step_name not in self.steps:
            self.logger.error(f"Unknown step: {step_name}")
            return False

        step = self.steps[step_name]

        if not self._are_dependencies_completed(step):
            self.logger.error(f"Dependencies not completed for step: {step_name}")
            return False

        step.start()
        self.current_stage = step.stage
        self.logger.info(f"Started workflow step: {step_name}")
        return True

    def complete_step(
        self, step_name: str, result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Complete a workflow step.

        Args:
            step_name: Name of step to complete
            result: Optional result data

        Returns:
            True if step completed successfully
        """
        if step_name not in self.steps:
            self.logger.error(f"Unknown step: {step_name}")
            return False

        step = self.steps[step_name]
        step.complete(result)
        self.logger.info(f"Completed workflow step: {step_name}")

        # Update context if relevant
        if step_name == "create_backup":
            self.context.backup_created = True
        elif step_name == "validate_migration" and result:
            self.context.test_results = result

        return True

    def fail_step(self, step_name: str, error: str) -> bool:
        """Fail a workflow step.

        Args:
            step_name: Name of step that failed
            error: Error description

        Returns:
            True if step marked as failed
        """
        if step_name not in self.steps:
            self.logger.error(f"Unknown step: {step_name}")
            return False

        step = self.steps[step_name]
        step.fail(error)
        self.current_stage = WorkflowStage.FAILED
        self.logger.error(f"Failed workflow step {step_name}: {error}")
        return True

    def can_retry_step(self, step_name: str) -> bool:
        """Check if a step can be retried.

        Args:
            step_name: Name of step to check

        Returns:
            True if step can be retried
        """
        if step_name not in self.steps:
            return False

        step = self.steps[step_name]

        # Repair step can be retried up to max attempts
        if step.stage == WorkflowStage.REPAIR:
            return self.context.repair_attempts < self.context.max_repair_attempts

        # Other failed steps can generally be retried once
        return step.status == WorkflowStatus.FAILED

    def retry_step(self, step_name: str) -> bool:
        """Retry a failed step.

        Args:
            step_name: Name of step to retry

        Returns:
            True if step reset for retry
        """
        if not self.can_retry_step(step_name):
            self.logger.error(f"Cannot retry step: {step_name}")
            return False

        step = self.steps[step_name]
        step.status = WorkflowStatus.PENDING
        step.start_time = None
        step.end_time = None
        step.error = None

        if step.stage == WorkflowStage.REPAIR:
            self.context.repair_attempts += 1

        self.logger.info(f"Reset step for retry: {step_name}")
        return True

    def is_workflow_complete(self) -> bool:
        """Check if the entire workflow is complete.

        Returns:
            True if workflow is complete
        """
        return all(
            step.status == WorkflowStatus.COMPLETED for step in self.steps.values()
        )

    def is_workflow_failed(self) -> bool:
        """Check if the workflow has failed permanently.

        Returns:
            True if workflow has failed
        """
        # Check if any critical step failed and cannot be retried
        for step in self.steps.values():
            if step.status == WorkflowStatus.FAILED:
                if not self.can_retry_step(step.name):
                    return True
        return False

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow state.

        Returns:
            Workflow summary dictionary
        """
        completed_steps = sum(
            1 for step in self.steps.values() if step.status == WorkflowStatus.COMPLETED
        )
        failed_steps = sum(
            1 for step in self.steps.values() if step.status == WorkflowStatus.FAILED
        )

        total_duration = 0.0
        for step in self.steps.values():
            if step.duration:
                total_duration += step.duration

        return {
            "context": self.context.to_dict(),
            "current_stage": self.current_stage.value,
            "total_steps": len(self.steps),
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "total_duration_seconds": total_duration,
            "is_complete": self.is_workflow_complete(),
            "is_failed": self.is_workflow_failed(),
            "repair_attempts": self.context.repair_attempts,
            "steps": {
                name: {
                    "status": step.status.value,
                    "stage": step.stage.value,
                    "duration": step.duration,
                    "error": step.error,
                }
                for name, step in self.steps.items()
            },
        }

    def save_state(self, file_path: str) -> bool:
        """Save workflow state to file.

        Args:
            file_path: Path to save state file

        Returns:
            True if saved successfully
        """
        try:
            summary = self.get_workflow_summary()
            with open(file_path, "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Saved workflow state to: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save workflow state: {e}")
            return False
