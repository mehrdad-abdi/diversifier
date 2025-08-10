"""Tests for Testing MCP Server functionality."""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.mcp_servers.testing.launcher import TestingMCPClient
from src.mcp_servers.testing.server import TestingMCPServer


class TestTestingMCPServer:
    """Test cases for TestingMCPServer."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)

            # Create basic project structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()

            # Create sample Python files
            (project_path / "src" / "__init__.py").write_text("")
            (project_path / "src" / "sample.py").write_text(
                """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
            )

            # Create sample test files
            (project_path / "tests" / "__init__.py").write_text("")
            (project_path / "tests" / "test_sample.py").write_text(
                """
import pytest
from src.sample import add, multiply

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0

def test_failing():
    assert False, "This test always fails"
"""
            )

            yield project_path

    @pytest.fixture
    def server(self, temp_project_dir):
        """Create TestingMCPServer instance for testing."""
        return TestingMCPServer(project_root=str(temp_project_dir))

    def test_server_initialization(self, temp_project_dir):
        """Test TestingMCPServer initialization."""
        server = TestingMCPServer(project_root=str(temp_project_dir))

        assert server.project_root.resolve() == temp_project_dir.resolve()
        assert server.server is not None
        assert server.server.name == "testing-server"

    def test_path_validation_within_project(self, server, temp_project_dir):
        """Test path validation for paths within project boundaries."""
        # Test relative path
        validated = server._validate_path("tests/test_sample.py")
        expected = temp_project_dir / "tests" / "test_sample.py"
        assert validated.resolve() == expected.resolve()

        # Test absolute path within project
        absolute_path = str(temp_project_dir / "src" / "sample.py")
        validated = server._validate_path(absolute_path)
        assert validated.resolve() == (temp_project_dir / "src" / "sample.py").resolve()

    def test_path_validation_outside_project(self, server):
        """Test path validation rejects paths outside project boundaries."""
        with pytest.raises(ValueError, match="outside project boundaries"):
            server._validate_path("/etc/passwd")

        with pytest.raises(ValueError, match="outside project boundaries"):
            server._validate_path("../../../etc/passwd")

    @patch("subprocess.run")
    def test_discover_tests_success(self, mock_run, server):
        """Test successful test discovery."""
        # Mock successful pytest --collect-only output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="tests/test_sample.py::test_add\ntests/test_sample.py::test_multiply\ntests/test_sample.py::test_failing\n",
        )

        result = asyncio.run(server._discover_tests("tests/", "test_*.py"))

        assert len(result) == 1
        response = json.loads(result[0].text)

        assert response["collection_successful"] is True
        assert response["total_tests"] == 3
        assert "test_add" in str(response["tests"])
        assert any("test_sample.py" in f for f in response["test_files"])

        # Verify subprocess call
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pytest" in args
        assert "--collect-only" in args

    @patch("subprocess.run")
    def test_discover_tests_failure(self, mock_run, server):
        """Test test discovery failure."""
        # Mock failed pytest --collect-only output
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Collection failed: SyntaxError in test file",
        )

        result = asyncio.run(server._discover_tests("tests/", "test_*.py"))

        assert len(result) == 1
        response = json.loads(result[0].text)

        assert response["collection_successful"] is False
        assert len(response["errors"]) > 0
        assert response["total_tests"] == 0

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    @patch("builtins.open")
    @patch("pathlib.Path.unlink")
    def test_run_tests_success(
        self, mock_unlink, mock_open, mock_tempfile, mock_run, server
    ):
        """Test successful test execution."""
        # Mock temporary file for JSON output
        mock_tempfile.return_value.__enter__.return_value.name = (
            "/tmp/pytest_output.json"
        )

        # Mock successful pytest execution
        mock_run.return_value = Mock(
            returncode=0, stdout="===== 2 passed, 1 failed =====", stderr=""
        )

        # Mock JSON report file content
        json_report = {
            "summary": {"total": 3, "passed": 2, "failed": 1, "skipped": 0, "error": 0}
        }
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            json_report
        )

        result = asyncio.run(server._run_tests("tests/", "test_sample", True, "no"))

        assert len(result) == 1
        response = json.loads(result[0].text)

        assert response["success"] is True
        assert response["exit_code"] == 0
        assert "summary" in response
        assert response["summary"]["total"] == 3
        assert response["summary"]["passed"] == 2
        assert response["summary"]["failed"] == 1

        # Verify subprocess call
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pytest" in args
        assert "-k" in args
        assert "test_sample" in args

    @patch("subprocess.run")
    def test_run_tests_timeout(self, mock_run, server):
        """Test test execution timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 300)

        result = asyncio.run(server._run_tests(None, None, True, "no"))

        assert len(result) == 1
        assert "timed out" in result[0].text

    def test_parse_pytest_summary(self, server):
        """Test parsing of pytest summary output."""
        output = """
===== test session starts =====
platform linux -- Python 3.9.0
collected 5 items

tests/test_sample.py::test_add PASSED
tests/test_sample.py::test_multiply PASSED
tests/test_sample.py::test_failing FAILED

===== 2 passed, 1 failed in 0.15s =====
"""

        summary = server._parse_pytest_summary(output)

        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["total"] == 3

    def test_parse_coverage_output(self, server):
        """Test parsing of coverage output."""
        output = """
Name                 Stmts   Miss  Cover
----------------------------------------
src/__init__.py          0      0   100%
src/sample.py           6      1    83%
----------------------------------------
TOTAL                   6      1    83%
"""

        coverage_info = server._parse_coverage_output(output)

        assert coverage_info["total_coverage"] == "83%"
        assert len(coverage_info["files"]) == 2
        assert coverage_info["files"][0]["name"] == "src/__init__.py"
        assert coverage_info["files"][0]["coverage"] == "100%"

    def test_parse_failures_and_errors(self, server):
        """Test parsing of failures and errors from pytest output."""
        output = """
FAILURES
_______ test_failing _______

    def test_failing():
>       assert False, "This test always fails"
E       AssertionError: This test always fails
E       assert False

tests/test_sample.py:10: AssertionError

ERRORS
_______ test_error _______

ImportError: No module named 'missing_module'

===== short test summary info =====
FAILED tests/test_sample.py::test_failing - AssertionError: This test always fails
ERROR tests/test_sample.py::test_error - ImportError: No module named 'missing_module'
"""

        failures, errors = server._parse_failures_and_errors(output)

        assert len(failures) == 1
        assert len(errors) == 1
        assert "test_failing" in failures[0]["test_name"]
        assert "test_error" in errors[0]["test_name"]
        assert "AssertionError" in " ".join(failures[0]["traceback"])

    def test_load_test_results_json_string(self, server):
        """Test loading test results from JSON string."""
        json_data = {"summary": {"passed": 5, "failed": 0}}
        result = server._load_test_results(json.dumps(json_data))

        assert result == json_data

    @patch("builtins.open")
    def test_load_test_results_file(self, mock_open, server, temp_project_dir):
        """Test loading test results from file."""
        json_data = {"summary": {"passed": 5, "failed": 0}}
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            json_data
        )

        result = server._load_test_results("results.json")

        assert result == json_data

    def test_compare_pass_fail(self, server):
        """Test comparison of pass/fail results."""
        baseline = {"summary": {"passed": 5, "failed": 0, "skipped": 1}}
        migration = {"summary": {"passed": 4, "failed": 1, "skipped": 1}}

        differences = server._compare_pass_fail(baseline, migration)

        assert len(differences) == 2
        assert "passed: baseline=5, migration=4" in differences
        assert "failed: baseline=0, migration=1" in differences

    def test_compare_coverage(self, server):
        """Test comparison of coverage results."""
        baseline = {"coverage": {"total_coverage": "85%"}}
        migration = {"coverage": {"total_coverage": "82%"}}

        differences = server._compare_coverage(baseline, migration)

        assert len(differences) == 1
        assert "Total coverage: baseline=85%, migration=82%" in differences[0]


class TestTestingMCPClient:
    """Test cases for TestingMCPClient."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def client(self, temp_project_dir):
        """Create TestingMCPClient instance for testing."""
        return TestingMCPClient(project_root=str(temp_project_dir))

    def test_client_initialization(self, temp_project_dir):
        """Test TestingMCPClient initialization."""
        client = TestingMCPClient(project_root=str(temp_project_dir))

        assert client.project_root == str(temp_project_dir)
        assert client.process is None

    @patch("subprocess.Popen")
    def test_start_server_success(self, mock_popen, client):
        """Test successful server startup."""
        mock_process = Mock()
        mock_popen.return_value = mock_process

        result = client.start_server()

        assert result is True
        assert client.process == mock_process
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_start_server_failure(self, mock_popen, client):
        """Test server startup failure."""
        mock_popen.side_effect = Exception("Failed to start process")

        result = client.start_server()

        assert result is False
        assert client.process is None

    def test_stop_server(self, client):
        """Test server shutdown."""
        # Mock process
        mock_process = Mock()
        mock_process.wait.return_value = None
        client.process = mock_process

        client.stop_server()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        assert client.process is None

    def test_stop_server_force_kill(self, client):
        """Test server shutdown with force kill."""
        # Mock process that doesn't terminate gracefully
        mock_process = Mock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired("test", 5)
        client.process = mock_process

        client.stop_server()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert client.process is None

    def test_send_request_no_process(self, client):
        """Test sending request when no process is running."""
        result = client.send_request("test_method")

        assert result is None

    def test_send_request_success(self, client):
        """Test successful request sending."""
        # Mock process with stdin/stdout
        mock_process = Mock()
        mock_stdin = Mock()
        mock_stdout = Mock()
        mock_stdout.readline.return_value = '{"result": "success"}\n'

        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout

        client.process = mock_process

        result = client.send_request("test_method", {"param": "value"})

        assert result == {"result": "success"}
        mock_stdin.write.assert_called_once()
        mock_stdin.flush.assert_called_once()

    def test_discover_tests_integration(self, client):
        """Test discover_tests method integration."""
        # Mock successful call_tool response
        client.call_tool = Mock(
            return_value={
                "result": [
                    {
                        "text": json.dumps(
                            {
                                "total_tests": 5,
                                "tests": ["test1", "test2", "test3", "test4", "test5"],
                            }
                        )
                    }
                ]
            }
        )

        result = client.discover_tests("tests/", "test_*.py")

        assert result is not None
        assert result["total_tests"] == 5
        assert len(result["tests"]) == 5

        client.call_tool.assert_called_once_with(
            "discover_tests", {"test_path": "tests/", "pattern": "test_*.py"}
        )

    def test_run_tests_with_coverage_integration(self, client):
        """Test run_tests_with_coverage method integration."""
        # Mock successful call_tool response
        coverage_data = {"coverage_passed": True, "coverage": {"total_coverage": "85%"}}

        client.call_tool = Mock(
            return_value={"result": [{"text": json.dumps(coverage_data)}]}
        )

        result = client.run_tests_with_coverage(
            test_path="tests/",
            source_path="src/",
            coverage_format="term-missing",
            min_coverage=80.0,
        )

        assert result is not None
        assert result["coverage_passed"] is True
        assert result["coverage"]["total_coverage"] == "85%"

    def test_context_manager_success(self, client):
        """Test context manager successful operation."""
        client.start_server = Mock(return_value=True)
        client.stop_server = Mock()

        with client as c:
            assert c == client

        client.start_server.assert_called_once()
        client.stop_server.assert_called_once()

    def test_context_manager_failure(self, client):
        """Test context manager with startup failure."""
        client.start_server = Mock(return_value=False)

        with pytest.raises(RuntimeError, match="Failed to start Testing MCP Server"):
            with client:
                pass
