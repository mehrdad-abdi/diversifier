import argparse
import asyncio
import atexit
import json
import signal
import sys
import traceback
from pathlib import Path
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
from .test_running.simple_test_runner import SimpleLLMTestRunner

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
  diversifier test-runner /path/to/project --verbose
        """,
    )

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Main diversification command (default)
    main_parser = subparsers.add_parser(
        "diversify", help="Main diversification workflow (default)"
    )
    _add_main_arguments(main_parser)

    # LLM test runner command
    test_parser = subparsers.add_parser(
        "test-runner", help="LLM-powered test runner for intelligent project analysis"
    )
    _add_test_runner_arguments(test_parser)

    # Set default command for backward compatibility
    parser.set_defaults(command="diversify")

    return parser


def _add_main_arguments(parser):
    """Add arguments for main diversification command."""
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


def _add_test_runner_arguments(parser):
    """Add arguments for LLM test runner command."""
    parser.add_argument(
        "project_path", help="Path to the project directory to analyze and test"
    )

    parser.add_argument(
        "--model",
        default="claude-3-haiku-20240307",
        help="LLM model to use for analysis (default: claude-3-haiku-20240307)",
    )

    parser.add_argument(
        "--output", "-o", help="Output file to save results (JSON format)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze the project but don't install dependencies or run tests",
    )

    parser.add_argument(
        "--step",
        choices=["analyze", "setup", "test", "all"],
        default="all",
        help="Run only specific steps (default: all)",
    )


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
    finally:
        # Clear global coordinator reference
        _coordinator = None


async def run_test_runner(args) -> int:
    """Run the LLM test runner."""

    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"❌ Error: Project path does not exist: {project_path}")
        return 1

    if not project_path.is_dir():
        print(f"❌ Error: Project path is not a directory: {project_path}")
        return 1

    print(f"🚀 Starting LLM-powered test analysis for: {project_path}")
    print(f"📋 Model: {args.model}")
    print(f"📋 Step: {args.step}")
    if args.dry_run:
        print("🔍 Dry run mode: No installation or test execution")

    try:
        runner = SimpleLLMTestRunner(str(project_path), args.model)

        # Execute based on step selection (simplified for now)
        if args.dry_run or args.step == "analyze":
            # Just run project structure analysis
            project_structure = runner.analyze_project_structure()
            requirements = await runner.detect_dev_requirements(project_structure)

            results = {
                "project_structure": project_structure,
                "requirements_analysis": requirements,
                "step": args.step,
            }
        else:
            # Run full cycle
            results = await runner.run_full_test_cycle()

        # Print results
        _print_test_runner_results(results, args.verbose)

        # Save output if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {args.output}")

        # Determine success
        overall_success = results.get("overall_success", False)
        print(
            f"\n{'🎉 Overall Result: SUCCESS' if overall_success else '💥 Overall Result: FAILED'}"
        )

        return 0 if overall_success else 1

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        return 1


def _print_test_runner_results(results, verbose: bool = False) -> None:
    """Print test runner results in a user-friendly format."""

    if "project_structure" in results:
        structure = results["project_structure"]
        print("🔍 Project Analysis:")
        print(f"  Project Path: {structure['project_path']}")
        print(f"  Configuration Files: {len(structure['project_files'])} found")
        print(f"  Test Directories: {len(structure['test_directories'])} found")
        print(f"  Test Files: {len(structure['test_files'])} found")

    if "requirements_analysis" in results:
        req = results["requirements_analysis"]
        print("\n🧠 LLM Requirements Analysis:")
        print(f"  Testing Framework: {req['testing_framework']}")
        print(f"  Dev Dependencies: {', '.join(req['dev_dependencies'])}")
        if verbose:
            print(f"  Install Commands: {', '.join(req['install_commands'])}")
            print(f"  Test Commands: {', '.join(req['test_commands'])}")
        print(f"  Analysis: {req['analysis']}")

    if "environment_setup" in results:
        setup = results["environment_setup"]
        status = "✅ Success" if setup["success"] else "❌ Failed"
        print(f"\n⚙️ Environment Setup: {status}")

    if "test_execution" in results:
        test = results["test_execution"]
        status = "✅ Success" if test["overall_success"] else "❌ Failed"
        print(f"\n🧪 Test Execution: {status}")
        summary = test["summary"]
        print(
            f"  Commands: {summary['successful_commands']}/{summary['total_commands']} successful"
        )

    if "results_analysis" in results:
        analysis = results["results_analysis"]
        print("\n🤖 LLM Analysis:")
        print(f"  Status: {analysis['status'].upper()}")
        print(f"  Summary: {analysis['summary']}")

        if analysis.get("issues_found"):
            print("  Issues Found:")
            for issue in analysis["issues_found"]:
                print(f"    ⚠️ {issue}")

        if analysis.get("recommendations"):
            print("  Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"    💡 {rec}")


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    # Setup signal handlers for graceful shutdown
    _setup_signal_handlers()

    # Setup LangSmith tracing if configured
    setup_langsmith_tracing()

    # Handle test-runner command
    if args.command == "test-runner":
        return asyncio.run(run_test_runner(args))

    # Handle config file creation for main command
    if hasattr(args, "create_config") and args.create_config:
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

    # For main diversification command (backward compatibility)
    if args.command is None or args.command == "diversify":
        # Validate required arguments for normal operation
        if not hasattr(args, "config") or not args.config:
            print("❌ Error: --config is required")
            print("Use --create-config to create a configuration file first")
            return 1

        if (
            not args.project_path
            or not hasattr(args, "remove_lib")
            or not args.remove_lib
            or not hasattr(args, "inject_lib")
            or not args.inject_lib
        ):
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
            print(f"Dry run: {args.dry_run}")

        # Store validated path
        args.project_path = project_path

        # Run the diversification workflow
        return asyncio.run(run_diversification(args))

    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
