"""High-level workflow orchestration coordinator."""

import logging
from typing import Dict, Any, Optional

from .simple_workflow import MigrationWorkflow
from .config import LLMConfig, MigrationConfig


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
        if llm_config is None:
            raise ValueError(
                "llm_config is required - no default configuration available"
            )

        # Use provided migration config or create default
        if migration_config is None:
            migration_config = MigrationConfig()

        # Initialize the simple workflow
        self.workflow = MigrationWorkflow(
            project_path=project_path,
            source_library=source_library,
            target_library=target_library,
            llm_config=llm_config,
            migration_config=migration_config,
        )

        self.logger = logging.getLogger("diversifier.coordinator")

    async def execute_workflow(self, auto_proceed: bool = False) -> bool:
        """Execute the complete diversification workflow.

        Args:
            auto_proceed: If True, don't wait for user confirmation (unused in simple workflow)

        Returns:
            True if workflow completed successfully
        """
        return await self.workflow.execute()

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status.

        Returns:
            Workflow status summary
        """
        return self.workflow.get_workflow_summary()
