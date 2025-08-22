import argparse
import asyncio
import atexit
import signal
import sys
from typing import Optional

from .validation import (
    validate_python_project,
    validate_project_path,
    validate_library_name,
)
from .orchestration.coordinator import DiversificationCoordinator
from .orchestration.config import LoggingConfig, get_config, ConfigManager
from .orchestration.logging_config import setup_logging
from .orchestration.langsmith_config import setup_langsmith_tracing

# Global reference to coordinator for signal handler cleanup
_coordinator: Optional[DiversificationCoordinator] = None


def _cleanup_on_exit() -> None:
    """Cleanup coordinator resources synchronously for signal handling."""
    if _coordinator:
        print("🧹 Cleaning up MCP servers and resources...")
        try:
            # Use the emergency shutdown method for synchronous cleanup
            if hasattr(_coordinator, "mcp_manager") and _coordinator.mcp_manager:
                _coordinator.mcp_manager.emergency_shutdown()
                print("  ✓ All MCP servers terminated")

            # Clear agent memories if possible
            if hasattr(_coordinator, "agent_manager") and _coordinator.agent_manager:
                try:
                    _coordinator.agent_manager.clear_all_memories()
                    print("  ✓ Agent memories cleared")
                except Exception:
                    pass  # Not critical for emergency shutdown

        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")


def _signal_handler(signum: int, frame) -> None:
    """Handle termination signals by cleaning up coordinator resources."""
    signal_name = signal.Signals(signum).name
    print(f"\n🛑 Received {signal_name}, shutting down gracefully...")

    _cleanup_on_exit()

    print("👋 Shutdown complete")
    sys.exit(130)  # Standard exit code for SIGINT


def _setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""
    try:
        signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, _signal_handler)  # Termination request
        # Also register cleanup for normal exit
        atexit.register(_cleanup_on_exit)
    except (ValueError, OSError):
        # Some signals might not be available on all platforms
        pass


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
        "-c",
        "--config",
        type=str,
        help="Path to configuration file (REQUIRED for normal operation)",
    )

    parser.add_argument(
        "--create-config",
        type=str,
        help="Create a default configuration file at the specified path and exit",
    )

    parser.add_argument(
        "project_path",
        type=str,
        nargs="?",
        help="Path to the Python project directory to diversify",
    )

    parser.add_argument(
        "remove_lib", type=str, nargs="?", help="Name of the library to remove/replace"
    )

    parser.add_argument(
        "inject_lib",
        type=str,
        nargs="?",
        help="Name of the library to inject/substitute",
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
    global _coordinator

    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    logging_config = LoggingConfig(
        level=log_level, console=True, file_path=args.log_file
    )
    setup_logging(logging_config)

    try:
        # Initialize coordinator with configuration from config file
        config = get_config(args.config)

        coordinator = DiversificationCoordinator(
            project_path=str(args.project_path),
            source_library=args.remove_lib,
            target_library=args.inject_lib,
            llm_config=config.llm,
            migration_config=config.migration,
        )

        # Set global coordinator for signal handling
        _coordinator = coordinator

        # Execute workflow
        success = await coordinator.execute_workflow()

        if success:
            print("✅ Diversification completed successfully!")
            return 0
        else:
            print("❌ Diversification failed")
            return 1

    except Exception as e:
        print(f"❌ Diversification failed with error: {e}")
        return 1
    finally:
        # Clear global coordinator reference
        _coordinator = None


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    # Setup signal handlers for graceful shutdown
    _setup_signal_handlers()

    # Setup LangSmith tracing if configured
    setup_langsmith_tracing()

    # Handle config file creation
    if args.create_config:
        try:
            config_manager = ConfigManager(args.create_config)
            config_manager.save_config_template(args.create_config)
            print(f"✅ Configuration template created: {args.create_config}")
            print(
                f"Edit the file to add your LLM provider settings, then run diversifier with --config {args.create_config}"
            )
            return 0
        except Exception as e:
            print(f"❌ Failed to create config file: {e}")
            return 1

    # Validate required arguments for normal operation
    if not args.config:
        print("❌ Error: --config is required")
        print("Use --create-config to create a configuration file first")
        return 1

    if not args.project_path or not args.remove_lib or not args.inject_lib:
        print("❌ Error: project_path, remove_lib, and inject_lib are required")
        return 1

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

    # Store validated path
    args.project_path = project_path

    # Run the diversification workflow
    return asyncio.run(run_diversification(args))


if __name__ == "__main__":
    sys.exit(main())
