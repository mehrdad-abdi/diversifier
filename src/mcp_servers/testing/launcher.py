"""Testing MCP Server launcher and lifecycle management."""

import asyncio
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List


class TestingMCPClient:
    """Client for managing Testing MCP Server lifecycle and communication."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the MCP client.

        Args:
            project_root: Root directory to constrain test operations to.
        """
        self.project_root = project_root or str(Path.cwd())
        self.process: Optional[subprocess.Popen] = None

    def start_server(self) -> bool:
        """Start the Testing MCP Server process.

        Returns:
            True if server started successfully, False otherwise.
        """
        try:
            # Get path to server script
            server_script = Path(__file__).parent / "server.py"

            # Start server process with stdio communication
            self.process = subprocess.Popen(
                [sys.executable, str(server_script), self.project_root],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )

            return True

        except Exception as e:
            print(f"Failed to start Testing MCP Server: {e}")
            return False

    def stop_server(self) -> None:
        """Stop the Testing MCP Server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC request to the server.

        Args:
            method: Method name to call
            params: Parameters for the method

        Returns:
            Response from server or None if error
        """
        if not self.process:
            return None

        request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            if self.process.stdin:
                self.process.stdin.write(request_json)
                self.process.stdin.flush()

            # Read response
            if self.process.stdout:
                response_line = self.process.stdout.readline()
                if response_line:
                    return json.loads(response_line.strip())

        except Exception as e:
            print(f"Error communicating with server: {e}")

        return None

    def list_tools(self) -> Optional[Dict[str, Any]]:
        """List available tools."""
        return self.send_request("tools/list")

    def call_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a specific tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        return self.send_request("tools/call", {"name": name, "arguments": arguments})

    def discover_tests(
        self, test_path: str = "tests/", pattern: str = "test_*.py"
    ) -> Optional[Dict[str, Any]]:
        """Discover tests using pytest.

        Args:
            test_path: Path to test directory or file
            pattern: Test file pattern

        Returns:
            Test discovery results
        """
        result = self.call_tool(
            "discover_tests", {"test_path": test_path, "pattern": pattern}
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def run_tests(
        self,
        test_path: Optional[str] = None,
        test_filter: Optional[str] = None,
        verbose: bool = True,
        capture: str = "no",
    ) -> Optional[Dict[str, Any]]:
        """Execute tests using pytest.

        Args:
            test_path: Path to test directory or specific test file
            test_filter: Pytest filter expression (-k option)
            verbose: Enable verbose output
            capture: Output capture mode (no, sys, fd)

        Returns:
            Test execution results
        """
        args = {"verbose": verbose, "capture": capture}
        if test_path:
            args["test_path"] = test_path
        if test_filter:
            args["test_filter"] = test_filter

        result = self.call_tool("run_tests", args)
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def run_tests_with_coverage(
        self,
        test_path: Optional[str] = None,
        source_path: str = "src/",
        coverage_format: str = "term-missing",
        min_coverage: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute tests with coverage analysis.

        Args:
            test_path: Path to test directory or specific test file
            source_path: Source code path for coverage
            coverage_format: Coverage report format
            min_coverage: Minimum required coverage percentage

        Returns:
            Test and coverage results
        """
        args: Dict[str, Any] = {"source_path": source_path, "coverage_format": coverage_format}
        if test_path:
            args["test_path"] = test_path
        if min_coverage:
            args["min_coverage"] = min_coverage

        result = self.call_tool("run_tests_with_coverage", args)
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def analyze_test_results(
        self, output_file: Optional[str] = None, output_text: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse and analyze pytest output.

        Args:
            output_file: Path to pytest output file (JSON format)
            output_text: Raw pytest output text to analyze

        Returns:
            Structured analysis results
        """
        args = {}
        if output_file:
            args["output_file"] = output_file
        if output_text:
            args["output_text"] = output_text

        result = self.call_tool("analyze_test_results", args)
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def compare_test_results(
        self,
        baseline_results: str,
        migration_results: str,
        comparison_type: str = "pass_fail",
    ) -> Optional[Dict[str, Any]]:
        """Compare test results between baseline and migrated versions.

        Args:
            baseline_results: Path to baseline test results file or raw results
            migration_results: Path to migration test results file or raw results
            comparison_type: Type of comparison (pass_fail, coverage, performance)

        Returns:
            Comparison results
        """
        result = self.call_tool(
            "compare_test_results",
            {
                "baseline_results": baseline_results,
                "migration_results": migration_results,
                "comparison_type": comparison_type,
            },
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def validate_test_equivalence(
        self, baseline_path: str, migrated_path: str, tolerance: float = 1e-10
    ) -> Optional[Dict[str, Any]]:
        """Validate that migrated code produces equivalent test results.

        Args:
            baseline_path: Path to baseline code version
            migrated_path: Path to migrated code version
            tolerance: Tolerance for numerical comparisons

        Returns:
            Validation results
        """
        result = self.call_tool(
            "validate_test_equivalence",
            {
                "baseline_path": baseline_path,
                "migrated_path": migrated_path,
                "tolerance": tolerance,
            },
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def create_test_environment(
        self, requirements: List[str], env_name: str = "test_env"
    ) -> Optional[Dict[str, Any]]:
        """Set up isolated test environment with specific dependencies.

        Args:
            requirements: List of Python package requirements
            env_name: Name for the test environment

        Returns:
            Environment setup results
        """
        result = self.call_tool(
            "create_test_environment",
            {"requirements": requirements, "env_name": env_name},
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def __enter__(self):
        """Context manager entry."""
        if self.start_server():
            return self
        else:
            raise RuntimeError("Failed to start Testing MCP Server")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()


# Example usage
async def example_usage():
    """Example of how to use the Testing MCP Client."""

    with TestingMCPClient() as client:
        # List available tools
        tools = client.list_tools()
        print("Available tools:", tools)

        # Discover tests
        discovered = client.discover_tests()
        if discovered:
            print(f"Discovered {discovered.get('total_tests', 0)} tests")

        # Run tests with coverage
        coverage_results = client.run_tests_with_coverage()
        if coverage_results:
            coverage = coverage_results.get("coverage", {}).get("total_coverage")
            print(f"Test coverage: {coverage}")

        # Compare baseline vs migration (example)
        baseline_results = '{"summary": {"passed": 5, "failed": 0}}'
        migration_results = '{"summary": {"passed": 5, "failed": 0}}'

        comparison = client.compare_test_results(baseline_results, migration_results)
        if comparison:
            print(f"Results equivalent: {comparison.get('equivalent', False)}")


if __name__ == "__main__":
    asyncio.run(example_usage())
