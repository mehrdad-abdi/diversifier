import argparse
import asyncio
import os
import sys

from .validation import (
    validate_python_project,
    validate_project_path,
    validate_library_name,
)
from .orchestration.coordinator import DiversificationCoordinator
from .orchestration.config import LoggingConfig, get_config
from .orchestration.logging_config import setup_logging


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diversifier",
        description="AI-powered tool to generate diversified project variants "
        "via dependency mutation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  diversifier /path/to/project requests httpx
  diversifier . pandas polars
  diversifier ~/myproject flask fastapi
        """,
    )

    parser.add_argument(
        "project_path",
        type=str,
        help="Path to the Python project directory to diversify",
    )

    parser.add_argument(
        "remove_lib", type=str, help="Name of the library to remove/replace"
    )

    parser.add_argument(
        "inject_lib", type=str, help="Name of the library to inject/substitute"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    parser.add_argument("--log-file", type=str, help="Write logs to file")

    return parser


async def run_diversification(args) -> int:
    """Run the diversification workflow."""
    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    logging_config = LoggingConfig(
        level=log_level, console=True, file_path=args.log_file
    )
    setup_logging(logging_config)

    try:
        # Initialize coordinator with configuration from config files/environment
        config = get_config()

        coordinator = DiversificationCoordinator(
            project_path=str(args.project_path),
            source_library=args.remove_lib,
            target_library=args.inject_lib,
            llm_config=config.llm,
        )

        # Execute workflow
        success = await coordinator.execute_workflow(
            dry_run=args.dry_run, auto_proceed=True  # For now, auto-proceed in CLI mode
        )

        if success:
            print("✅ Diversification completed successfully!")
            return 0
        else:
            print("❌ Diversification failed")
            return 1

    except Exception as e:
        print(f"❌ Diversification failed with error: {e}")
        return 1


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    project_path = validate_project_path(args.project_path)
    if not project_path:
        return 1

    is_valid_python_project, project_errors = validate_python_project(project_path)
    if not is_valid_python_project:
        print("Error: Invalid Python project:")
        for error in project_errors:
            print(f"  - {error}")
        return 1

    if not validate_library_name(args.remove_lib):
        return 1

    if not validate_library_name(args.inject_lib):
        return 1

    if args.remove_lib == args.inject_lib:
        print("Error: remove_lib and inject_lib cannot be the same")
        return 1

    if args.verbose:
        print(f"Project path: {project_path}")
        print(f"Remove library: {args.remove_lib}")
        print(f"Inject library: {args.inject_lib}")
        print(f"Dry run: {args.dry_run}")

    # Store validated path
    args.project_path = project_path

    # Run the diversification workflow
    return asyncio.run(run_diversification(args))


if __name__ == "__main__":
    sys.exit(main())
