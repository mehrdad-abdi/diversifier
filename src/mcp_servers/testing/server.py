#!/usr/bin/env python3
"""Testing MCP Server with stdio transport."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Tool,
    TextContent,
)


class TestingMCPServer:
    """MCP Server for test execution and analysis with security constraints."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Testing MCP Server.

        Args:
            project_root: Root directory to constrain test operations to.
                         If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.project_root = self.project_root.resolve()

        # Initialize MCP server
        self.server = Server("testing-server")

        # Register tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="discover_tests",
                    description="Discover tests in the project using pytest",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "test_path": {
                                "type": "string",
                                "description": "Path to test directory or file (default: tests/)",
                                "default": "tests/",
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Test file pattern (default: test_*.py)",
                                "default": "test_*.py",
                            },
                        },
                    },
                ),
                Tool(
                    name="run_tests",
                    description="Execute tests using pytest with optional filtering",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "test_path": {
                                "type": "string",
                                "description": "Path to test directory or specific test file",
                            },
                            "test_filter": {
                                "type": "string",
                                "description": "Pytest filter expression (-k option)",
                            },
                            "verbose": {
                                "type": "boolean",
                                "description": "Enable verbose output",
                                "default": True,
                            },
                            "capture": {
                                "type": "string",
                                "description": "Output capture mode (no, sys, fd)",
                                "default": "no",
                            },
                        },
                    },
                ),
                Tool(
                    name="run_tests_with_coverage",
                    description="Execute tests with coverage analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "test_path": {
                                "type": "string",
                                "description": "Path to test directory or specific test file",
                            },
                            "source_path": {
                                "type": "string",
                                "description": "Source code path for coverage (default: src/)",
                                "default": "src/",
                            },
                            "coverage_format": {
                                "type": "string",
                                "description": "Coverage report format (term, term-missing, html, json)",
                                "default": "term-missing",
                            },
                            "min_coverage": {
                                "type": "number",
                                "description": "Minimum required coverage percentage",
                            },
                        },
                    },
                ),
                Tool(
                    name="analyze_test_results",
                    description="Parse and analyze pytest output for structured reporting",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "output_file": {
                                "type": "string",
                                "description": "Path to pytest output file (JSON format)",
                            },
                            "output_text": {
                                "type": "string",
                                "description": "Raw pytest output text to analyze",
                            },
                        },
                    },
                ),
                Tool(
                    name="compare_test_results",
                    description="Compare test results between baseline and migrated versions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "baseline_results": {
                                "type": "string",
                                "description": "Path to baseline test results file or raw results",
                            },
                            "migration_results": {
                                "type": "string",
                                "description": "Path to migration test results file or raw results",
                            },
                            "comparison_type": {
                                "type": "string",
                                "description": "Type of comparison (pass_fail, coverage, performance)",
                                "default": "pass_fail",
                            },
                        },
                        "required": ["baseline_results", "migration_results"],
                    },
                ),
                Tool(
                    name="create_test_environment",
                    description="Set up isolated test environment with specific dependencies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "requirements": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of Python package requirements",
                            },
                            "env_name": {
                                "type": "string",
                                "description": "Name for the test environment",
                                "default": "test_env",
                            },
                        },
                    },
                ),
                Tool(
                    name="validate_test_equivalence",
                    description="Validate that migrated code produces equivalent test results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "baseline_path": {
                                "type": "string",
                                "description": "Path to baseline code version",
                            },
                            "migrated_path": {
                                "type": "string",
                                "description": "Path to migrated code version",
                            },
                            "tolerance": {
                                "type": "number",
                                "description": "Tolerance for numerical comparisons",
                                "default": 1e-10,
                            },
                        },
                        "required": ["baseline_path", "migrated_path"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "discover_tests":
                    return await self._discover_tests(
                        arguments.get("test_path", "tests/"),
                        arguments.get("pattern", "test_*.py"),
                    )
                elif name == "run_tests":
                    return await self._run_tests(
                        arguments.get("test_path"),
                        arguments.get("test_filter"),
                        arguments.get("verbose", True),
                        arguments.get("capture", "no"),
                    )
                elif name == "run_tests_with_coverage":
                    return await self._run_tests_with_coverage(
                        arguments.get("test_path"),
                        arguments.get("source_path", "src/"),
                        arguments.get("coverage_format", "term-missing"),
                        arguments.get("min_coverage"),
                    )
                elif name == "analyze_test_results":
                    return await self._analyze_test_results(
                        arguments.get("output_file"),
                        arguments.get("output_text"),
                    )
                elif name == "compare_test_results":
                    return await self._compare_test_results(
                        arguments["baseline_results"],
                        arguments["migration_results"],
                        arguments.get("comparison_type", "pass_fail"),
                    )
                elif name == "create_test_environment":
                    return await self._create_test_environment(
                        arguments.get("requirements", []),
                        arguments.get("env_name", "test_env"),
                    )
                elif name == "validate_test_equivalence":
                    return await self._validate_test_equivalence(
                        arguments["baseline_path"],
                        arguments["migrated_path"],
                        arguments.get("tolerance", 1e-10),
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _validate_path(self, file_path: str) -> Path:
        """Validate that path is within project boundaries.

        Args:
            file_path: Path to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is outside project boundaries
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path

        path = path.resolve()

        # Ensure path is within project root
        if not str(path).startswith(str(self.project_root)):
            raise ValueError(f"Path {path} is outside project boundaries")

        return path

    async def _discover_tests(self, test_path: str, pattern: str) -> list[TextContent]:
        """Discover tests in the project using pytest."""
        path = self._validate_path(test_path)

        try:
            # Use pytest to collect tests
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(path),
                "--collect-only",
                "-q",
                "--tb=no",
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Parse the collection output
            discovered_tests = {
                "test_path": str(path.relative_to(self.project_root)),
                "pattern": pattern,
                "collection_successful": result.returncode == 0,
                "tests": [],
                "test_files": [],
                "total_tests": 0,
                "errors": [],
            }

            if result.returncode == 0:
                # Parse stdout for test collection
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if "::" in line and not line.startswith("="):
                        # This is a test item
                        test_list = discovered_tests["tests"]
                        if isinstance(test_list, list):
                            test_list.append(line)
                            test_file = line.split("::")[0]
                            test_files_list = discovered_tests["test_files"]
                            if (
                                isinstance(test_files_list, list)
                                and test_file not in test_files_list
                            ):
                                test_files_list.append(test_file)

                test_count = discovered_tests["tests"]
                discovered_tests["total_tests"] = (
                    len(test_count) if isinstance(test_count, list) else 0
                )
            else:
                error_list = discovered_tests["errors"]
                if isinstance(error_list, list):
                    error_list.append(result.stderr)

            return [
                TextContent(type="text", text=json.dumps(discovered_tests, indent=2))
            ]

        except subprocess.TimeoutExpired:
            return [TextContent(type="text", text="Error: Test discovery timed out")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error discovering tests: {str(e)}")]

    async def _run_tests(
        self,
        test_path: Optional[str],
        test_filter: Optional[str],
        verbose: bool,
        capture: str,
    ) -> list[TextContent]:
        """Execute tests using pytest."""
        cmd = [sys.executable, "-m", "pytest"]

        if test_path:
            path = self._validate_path(test_path)
            cmd.append(str(path))

        if test_filter:
            cmd.extend(["-k", test_filter])

        if verbose:
            cmd.append("-v")

        cmd.extend(["-s" if capture == "no" else f"--capture={capture}"])

        # Add JSON output for structured results
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_output_file = f.name

        cmd.extend(["--json-report", f"--json-report-file={json_output_file}"])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Parse results
            test_results = {
                "command": " ".join(cmd),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            # Try to read JSON report if available
            try:
                with open(json_output_file, "r") as f:
                    json_report = json.load(f)
                test_results["json_report"] = json_report
                test_results["summary"] = {
                    "total": json_report.get("summary", {}).get("total", 0),
                    "passed": json_report.get("summary", {}).get("passed", 0),
                    "failed": json_report.get("summary", {}).get("failed", 0),
                    "skipped": json_report.get("summary", {}).get("skipped", 0),
                    "error": json_report.get("summary", {}).get("error", 0),
                }
            except Exception:
                # JSON report might not be available, use text parsing
                test_results["summary"] = self._parse_pytest_summary(result.stdout)

            # Cleanup temp file
            try:
                Path(json_output_file).unlink()
            except Exception:
                pass

            return [TextContent(type="text", text=json.dumps(test_results, indent=2))]

        except subprocess.TimeoutExpired:
            return [TextContent(type="text", text="Error: Test execution timed out")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error running tests: {str(e)}")]

    async def _run_tests_with_coverage(
        self,
        test_path: Optional[str],
        source_path: str,
        coverage_format: str,
        min_coverage: Optional[float],
    ) -> list[TextContent]:
        """Execute tests with coverage analysis."""
        source = self._validate_path(source_path)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            f"--cov={source}",
            f"--cov-report={coverage_format}",
        ]

        if test_path:
            path = self._validate_path(test_path)
            cmd.append(str(path))

        if min_coverage:
            cmd.append(f"--cov-fail-under={min_coverage}")

        # Add JSON output for structured results
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_output_file = f.name

        cmd.extend(["--json-report", f"--json-report-file={json_output_file}"])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            coverage_results = {
                "command": " ".join(cmd),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "coverage_passed": result.returncode == 0,
            }

            # Parse coverage information from output
            coverage_info = self._parse_coverage_output(result.stdout)
            coverage_results["coverage"] = coverage_info

            # Try to read JSON report
            try:
                with open(json_output_file, "r") as f:
                    json_report = json.load(f)
                coverage_results["json_report"] = json_report
            except Exception:
                pass

            # Cleanup temp file
            try:
                Path(json_output_file).unlink()
            except Exception:
                pass

            return [
                TextContent(type="text", text=json.dumps(coverage_results, indent=2))
            ]

        except subprocess.TimeoutExpired:
            return [TextContent(type="text", text="Error: Coverage analysis timed out")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error running coverage: {str(e)}")]

    def _parse_pytest_summary(self, output: str) -> Dict[str, int]:
        """Parse pytest summary from text output."""
        summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0}

        lines = output.split("\n")
        for line in lines:
            if (
                "passed" in line
                or "failed" in line
                or "skipped" in line
                or "error" in line
            ):
                # Look for summary line like "5 passed, 2 failed, 1 skipped"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i + 1 < len(parts):
                        count = int(part)
                        status = parts[i + 1].rstrip(",")
                        if status in summary:
                            summary[status] = count
                            summary["total"] += count

        return summary

    def _parse_coverage_output(self, output: str) -> Dict[str, Any]:
        """Parse coverage information from output."""
        coverage_info: Dict[str, Any] = {"files": [], "total_coverage": None}

        lines = output.split("\n")
        in_coverage_section = False

        for line in lines:
            line = line.strip()

            # Look for coverage section
            if "Name" in line and "Stmts" in line and "Miss" in line:
                in_coverage_section = True
                continue

            if in_coverage_section:
                if line.startswith("TOTAL"):
                    # Extract total coverage
                    parts = line.split()
                    if len(parts) >= 4 and parts[-1].endswith("%"):
                        coverage_info["total_coverage"] = parts[-1]
                    break
                elif line and not line.startswith("---"):
                    # Parse individual file coverage
                    parts = line.split()
                    if len(parts) >= 4 and parts[-1].endswith("%"):
                        file_info = {
                            "name": parts[0],
                            "statements": parts[1] if parts[1].isdigit() else None,
                            "missing": parts[2] if parts[2].isdigit() else None,
                            "coverage": parts[-1],
                        }
                        coverage_info["files"].append(file_info)

        return coverage_info

    async def _analyze_test_results(
        self, output_file: Optional[str], output_text: Optional[str]
    ) -> list[TextContent]:
        """Parse and analyze pytest output for structured reporting."""
        analysis: Dict[str, Any] = {
            "analysis_source": None,
            "test_results": {},
            "failures": [],
            "errors": [],
            "summary": {},
        }

        try:
            if output_file:
                path = self._validate_path(output_file)
                analysis["analysis_source"] = f"file: {path}"

                if path.suffix == ".json":
                    with open(path, "r") as f:
                        json_data = json.load(f)
                    analysis["test_results"] = json_data
                else:
                    with open(path, "r") as f:
                        output_text = f.read()

            if output_text:
                if not analysis["analysis_source"]:
                    analysis["analysis_source"] = "text input"

                # Parse text output
                analysis["summary"] = self._parse_pytest_summary(output_text)
                failures, errors = self._parse_failures_and_errors(output_text)
                analysis["failures"] = failures
                analysis["errors"] = errors

            return [TextContent(type="text", text=json.dumps(analysis, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error analyzing results: {str(e)}")]

    def _parse_failures_and_errors(
        self, output: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse failures and errors from pytest output."""
        failures: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        lines = output.split("\n")
        current_failure: Optional[Dict[str, Any]] = None
        current_error: Optional[Dict[str, Any]] = None
        in_failure_section = False
        in_error_section = False

        for line in lines:
            # Detect failure sections
            if line.startswith("FAILURES"):
                in_failure_section = True
                in_error_section = False
                continue
            elif line.startswith("ERRORS"):
                in_error_section = True
                in_failure_section = False
                continue
            elif line.startswith("=") and (
                "short test summary" in line.lower() or "failed" in line.lower()
            ):
                in_failure_section = False
                in_error_section = False
                continue

            if in_failure_section and line.startswith("_"):
                # New failure
                if current_failure:
                    failures.append(current_failure)
                current_failure = {"test_name": line.strip("_ "), "traceback": []}
            elif in_error_section and line.startswith("_"):
                # New error
                if current_error:
                    errors.append(current_error)
                current_error = {"test_name": line.strip("_ "), "traceback": []}
            elif (in_failure_section and current_failure) or (
                in_error_section and current_error
            ):
                # Add to current failure/error
                target = current_failure if in_failure_section else current_error
                if target:
                    target["traceback"].append(line)

        # Add final failure/error
        if current_failure:
            failures.append(current_failure)
        if current_error:
            errors.append(current_error)

        return failures, errors

    async def _compare_test_results(
        self, baseline_results: str, migration_results: str, comparison_type: str
    ) -> list[TextContent]:
        """Compare test results between baseline and migrated versions."""
        comparison = {
            "comparison_type": comparison_type,
            "baseline": {},
            "migration": {},
            "differences": [],
            "equivalent": False,
        }

        try:
            # Load baseline results
            baseline_data = self._load_test_results(baseline_results)
            migration_data = self._load_test_results(migration_results)

            comparison["baseline"] = baseline_data
            comparison["migration"] = migration_data

            if comparison_type == "pass_fail":
                comparison["differences"] = self._compare_pass_fail(
                    baseline_data, migration_data
                )
            elif comparison_type == "coverage":
                comparison["differences"] = self._compare_coverage(
                    baseline_data, migration_data
                )
            elif comparison_type == "performance":
                comparison["differences"] = self._compare_performance(
                    baseline_data, migration_data
                )

            differences = comparison["differences"]
            comparison["equivalent"] = (
                len(differences) == 0 if isinstance(differences, list) else False
            )

            return [TextContent(type="text", text=json.dumps(comparison, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error comparing results: {str(e)}")]

    def _load_test_results(self, results_input: str) -> Dict[str, Any]:
        """Load test results from file path or JSON string."""
        try:
            # Try to parse as JSON first
            return json.loads(results_input)
        except json.JSONDecodeError:
            # Try to load as file path
            try:
                path = self._validate_path(results_input)
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                # Parse as text output
                return {
                    "summary": self._parse_pytest_summary(results_input),
                    "text_output": results_input,
                }

    def _compare_pass_fail(self, baseline: Dict, migration: Dict) -> List[str]:
        """Compare pass/fail results between baseline and migration."""
        differences = []

        baseline_summary = baseline.get("summary", {})
        migration_summary = migration.get("summary", {})

        for key in ["passed", "failed", "skipped", "error"]:
            baseline_count = baseline_summary.get(key, 0)
            migration_count = migration_summary.get(key, 0)

            if baseline_count != migration_count:
                differences.append(
                    f"{key}: baseline={baseline_count}, migration={migration_count}"
                )

        return differences

    def _compare_coverage(self, baseline: Dict, migration: Dict) -> List[str]:
        """Compare coverage results between baseline and migration."""
        differences = []

        baseline_cov = baseline.get("coverage", {}).get("total_coverage")
        migration_cov = migration.get("coverage", {}).get("total_coverage")

        if baseline_cov != migration_cov:
            differences.append(
                f"Total coverage: baseline={baseline_cov}, migration={migration_cov}"
            )

        return differences

    def _compare_performance(self, baseline: Dict, migration: Dict) -> List[str]:
        """Compare performance metrics between baseline and migration."""
        # This is a placeholder for performance comparison logic
        # In a real implementation, you would compare test execution times
        return []

    async def _create_test_environment(
        self, requirements: List[str], env_name: str
    ) -> list[TextContent]:
        """Set up isolated test environment with specific dependencies."""
        # For now, this is a placeholder that documents the approach
        # In a full implementation, this would use virtual environments
        env_info = {
            "env_name": env_name,
            "requirements": requirements,
            "status": "placeholder - not implemented",
            "note": "This would create a virtual environment with specified requirements",
            "implementation_approach": [
                "Create virtual environment using venv or virtualenv",
                "Install specified requirements",
                "Return environment activation commands",
                "Provide cleanup instructions",
            ],
        }

        return [TextContent(type="text", text=json.dumps(env_info, indent=2))]

    async def _validate_test_equivalence(
        self, baseline_path: str, migrated_path: str, tolerance: float
    ) -> list[TextContent]:
        """Validate that migrated code produces equivalent test results."""
        baseline_dir = self._validate_path(baseline_path)
        migrated_dir = self._validate_path(migrated_path)

        validation = {
            "baseline_path": str(baseline_dir.relative_to(self.project_root)),
            "migrated_path": str(migrated_dir.relative_to(self.project_root)),
            "tolerance": tolerance,
            "validation_steps": [],
            "equivalent": False,
        }

        try:
            # Run tests in baseline directory
            baseline_results = await self._run_tests_in_directory(baseline_dir)
            validation_steps = validation["validation_steps"]
            if isinstance(validation_steps, list):
                validation_steps.append("Executed baseline tests")

            # Run tests in migrated directory
            migrated_results = await self._run_tests_in_directory(migrated_dir)
            if isinstance(validation_steps, list):
                validation_steps.append("Executed migrated tests")

            # Compare results
            comparison = await self._compare_test_results(
                json.dumps(baseline_results), json.dumps(migrated_results), "pass_fail"
            )

            comparison_data = json.loads(comparison[0].text)
            validation["comparison"] = comparison_data
            validation["equivalent"] = comparison_data.get("equivalent", False)

            validation_steps = validation["validation_steps"]
            if isinstance(validation_steps, list):
                validation_steps.append("Completed comparison")

            return [TextContent(type="text", text=json.dumps(validation, indent=2))]

        except Exception as e:
            validation["error"] = str(e)
            return [TextContent(type="text", text=json.dumps(validation, indent=2))]

    async def _run_tests_in_directory(self, directory: Path) -> Dict[str, Any]:
        """Run tests in a specific directory and return results."""
        cmd = [sys.executable, "-m", "pytest", str(directory), "-v"]

        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "summary": self._parse_pytest_summary(result.stdout),
        }

    async def run(self) -> None:
        """Run the MCP server with stdio transport."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="testing-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def main():
    """Main entry point for the Testing MCP Server."""
    import asyncio
    import sys

    # Get project root from command line argument if provided
    project_root = sys.argv[1] if len(sys.argv) > 1 else None

    server = TestingMCPServer(project_root=project_root)

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
