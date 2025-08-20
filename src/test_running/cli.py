#!/usr/bin/env python3
"""CLI interface for the LLM-based test runner."""

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

from .llm_test_runner import LLMTestRunner


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM-powered test runner for intelligent project analysis and test execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tests on a project using LLM analysis
  python -m src.test_running.cli /path/to/project

  # Use a specific LLM model
  python -m src.test_running.cli /path/to/project --model claude-3-sonnet-20240229

  # Output results to a file
  python -m src.test_running.cli /path/to/project --output results.json

  # Verbose output
  python -m src.test_running.cli /path/to/project --verbose
        """,
    )

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

    return parser


def print_project_structure(structure: Dict[str, Any], verbose: bool = False) -> None:
    """Print project structure analysis results."""
    print("🔍 Project Analysis:")
    print(f"  Project Path: {structure['project_path']}")
    print(f"  Configuration Files: {len(structure['project_files'])} found")

    if verbose:
        for file_info in structure["project_files"]:
            print(f"    - {file_info['name']} ({file_info['type']})")

    print(f"  Test Directories: {len(structure['test_directories'])} found")
    for test_dir in structure["test_directories"]:
        print(f"    - {test_dir}/")

    print(f"  Test Files: {len(structure['test_files'])} found")
    if verbose:
        for test_file in structure["test_files"][:5]:  # Show first 5
            print(f"    - {test_file['path']}")
        if len(structure["test_files"]) > 5:
            print(f"    ... and {len(structure['test_files']) - 5} more")


def print_requirements_analysis(
    requirements: Dict[str, Any], verbose: bool = False
) -> None:
    """Print requirements analysis results."""
    print("\n🧠 LLM Requirements Analysis:")
    print(f"  Testing Framework: {requirements['testing_framework']}")
    print(f"  Dev Dependencies: {', '.join(requirements['dev_dependencies'])}")

    if verbose:
        print("  Install Commands:")
        for cmd in requirements["install_commands"]:
            print(f"    - {cmd}")

        print("  Test Commands:")
        for cmd in requirements["test_commands"]:
            print(f"    - {cmd}")

        if requirements["setup_commands"]:
            print("  Setup Commands:")
            for cmd in requirements["setup_commands"]:
                print(f"    - {cmd}")

    print(f"  Analysis: {requirements['analysis']}")


def print_setup_results(setup: Dict[str, Any], verbose: bool = False) -> None:
    """Print environment setup results."""
    print(f"\n⚙️ Environment Setup: {'✅ Success' if setup['success'] else '❌ Failed'}")

    if setup.get("setup_commands_executed"):
        print(f"  Setup Commands: {len(setup['setup_commands_executed'])} executed")
        if verbose:
            for cmd_result in setup["setup_commands_executed"]:
                status = "✅" if cmd_result["success"] else "❌"
                print(f"    {status} {cmd_result['command']}")

    if setup.get("install_commands_executed"):
        print(f"  Install Commands: {len(setup['install_commands_executed'])} executed")
        if verbose:
            for cmd_result in setup["install_commands_executed"]:
                status = "✅" if cmd_result["success"] else "❌"
                print(f"    {status} {cmd_result['command']}")

    if setup.get("errors"):
        print("  Errors:")
        for error in setup["errors"]:
            print(f"    ❌ {error}")


def print_test_results(test_results: Dict[str, Any], verbose: bool = False) -> None:
    """Print test execution results."""
    success = test_results["overall_success"]
    print(f"\n🧪 Test Execution: {'✅ Success' if success else '❌ Failed'}")

    summary = test_results["summary"]
    print(f"  Commands Executed: {summary['total_commands']}")
    print(f"  Successful: {summary['successful_commands']}")
    print(f"  Failed: {summary['failed_commands']}")

    if verbose:
        print("\n  Command Details:")
        for cmd_result in test_results["test_commands_executed"]:
            status = "✅" if cmd_result["success"] else "❌"
            print(
                f"    {status} {cmd_result['command']} (exit: {cmd_result['exit_code']})"
            )

            if not cmd_result["success"] and cmd_result.get("stderr"):
                # Show first few lines of error
                error_lines = cmd_result["stderr"].split("\n")[:3]
                for line in error_lines:
                    if line.strip():
                        print(f"       {line.strip()}")


def print_analysis_results(analysis: Dict[str, Any], verbose: bool = False) -> None:
    """Print LLM analysis results."""
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

    if verbose and analysis.get("test_metrics"):
        print(f"  Metrics: {analysis['test_metrics']}")


async def run_analysis_only(runner: LLMTestRunner) -> Dict[str, Any]:
    """Run only project analysis steps."""
    await runner.initialize_mcp_clients()

    try:
        project_structure = await runner.analyze_project_structure()
        requirements = await runner.detect_dev_requirements(project_structure)

        return {
            "project_structure": project_structure,
            "requirements_analysis": requirements,
            "step": "analyze",
        }
    finally:
        if runner.command_client:
            await runner.command_client.disconnect()
        if runner.filesystem_client:
            await runner.filesystem_client.disconnect()


async def run_setup_only(runner: LLMTestRunner) -> Dict[str, Any]:
    """Run project analysis and environment setup."""
    await runner.initialize_mcp_clients()

    try:
        project_structure = await runner.analyze_project_structure()
        requirements = await runner.detect_dev_requirements(project_structure)
        setup_results = await runner.setup_test_environment(requirements)

        return {
            "project_structure": project_structure,
            "requirements_analysis": requirements,
            "environment_setup": setup_results,
            "step": "setup",
        }
    finally:
        if runner.command_client:
            await runner.command_client.disconnect()
        if runner.filesystem_client:
            await runner.filesystem_client.disconnect()


async def run_test_only(runner: LLMTestRunner) -> Dict[str, Any]:
    """Run the full cycle up to test execution."""
    await runner.initialize_mcp_clients()

    try:
        project_structure = await runner.analyze_project_structure()
        requirements = await runner.detect_dev_requirements(project_structure)
        setup_results = await runner.setup_test_environment(requirements)
        test_results = await runner.run_tests(requirements)

        return {
            "project_structure": project_structure,
            "requirements_analysis": requirements,
            "environment_setup": setup_results,
            "test_execution": test_results,
            "step": "test",
        }
    finally:
        if runner.command_client:
            await runner.command_client.disconnect()
        if runner.filesystem_client:
            await runner.filesystem_client.disconnect()


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"❌ Error: Project path does not exist: {project_path}")
        sys.exit(1)

    if not project_path.is_dir():
        print(f"❌ Error: Project path is not a directory: {project_path}")
        sys.exit(1)

    print(f"🚀 Starting LLM-powered test analysis for: {project_path}")
    print(f"📋 Model: {args.model}")
    print(f"📋 Step: {args.step}")
    if args.dry_run:
        print("🔍 Dry run mode: No installation or test execution")

    try:
        runner = LLMTestRunner(str(project_path), args.model)

        # Execute based on step selection
        if args.step == "analyze":
            results = await run_analysis_only(runner)
        elif args.step == "setup":
            results = await run_setup_only(runner)
        elif args.step == "test":
            results = await run_test_only(runner)
        else:  # args.step == "all"
            if args.dry_run:
                results = await run_analysis_only(runner)
            else:
                results = await runner.run_full_test_cycle()

        # Print results
        if "project_structure" in results:
            print_project_structure(results["project_structure"], args.verbose)

        if "requirements_analysis" in results:
            print_requirements_analysis(results["requirements_analysis"], args.verbose)

        if "environment_setup" in results:
            print_setup_results(results["environment_setup"], args.verbose)

        if "test_execution" in results:
            print_test_results(results["test_execution"], args.verbose)

        if "results_analysis" in results:
            print_analysis_results(results["results_analysis"], args.verbose)

        # Overall result
        overall_success = results.get("overall_success", False)
        print(
            f"\n{'🎉 Overall Result: SUCCESS' if overall_success else '💥 Overall Result: FAILED'}"
        )

        # Save output if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {args.output}")

        # Set exit code
        sys.exit(0 if overall_success else 1)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
