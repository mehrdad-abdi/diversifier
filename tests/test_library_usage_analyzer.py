"""Tests for LibraryUsageAnalyzer."""

import ast
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.orchestration.test_generation.library_usage_analyzer import (
    LibraryUsageAnalyzer,
    LibraryUsageLocation,
    LibraryUsageSummary,
    LibraryUsageType,
)
from src.orchestration.mcp_manager import MCPManager, MCPServerType


class TestLibraryUsageAnalyzer:
    """Test LibraryUsageAnalyzer class."""

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = AsyncMock(spec=MCPManager)
        manager.is_server_available.return_value = False
        return manager

    @pytest.fixture
    def analyzer(self, mock_mcp_manager, tmp_path):
        """Create a LibraryUsageAnalyzer instance for testing."""
        return LibraryUsageAnalyzer(str(tmp_path), mock_mcp_manager)

    def test_init(self, mock_mcp_manager, tmp_path):
        """Test LibraryUsageAnalyzer initialization."""
        analyzer = LibraryUsageAnalyzer(str(tmp_path), mock_mcp_manager)

        assert analyzer.project_root == Path(tmp_path)
        assert analyzer.mcp_manager == mock_mcp_manager
        assert analyzer.logger.name == "diversifier.library_usage_analyzer"

    @pytest.mark.asyncio
    async def test_analyze_library_usage_empty_project(self, analyzer):
        """Test analyzing library usage in an empty project."""
        with patch.object(analyzer, "_get_python_files", return_value=[]):
            summary = await analyzer.analyze_library_usage("requests")

            assert summary.target_library == "requests"
            assert summary.total_usages == 0
            assert len(summary.usage_locations) == 0
            assert len(summary.affected_files) == 0

    @pytest.mark.asyncio
    async def test_analyze_library_usage_with_files(self, analyzer):
        """Test analyzing library usage with actual files."""
        test_files = ["/test/file1.py", "/test/file2.py"]
        test_content = "import requests\nrequests.get('url')"

        with (
            patch.object(analyzer, "_get_python_files", return_value=test_files),
            patch.object(analyzer, "_read_file", return_value=test_content),
        ):

            summary = await analyzer.analyze_library_usage("requests")

            assert summary.target_library == "requests"
            assert summary.total_usages > 0
            assert len(summary.affected_files) == 2

    @pytest.mark.asyncio
    async def test_analyze_library_usage_with_parse_error(self, analyzer):
        """Test analyzing library usage when file parsing fails."""
        test_files = ["/test/invalid.py"]
        invalid_content = "import requests\n  invalid python syntax"

        with (
            patch.object(analyzer, "_get_python_files", return_value=test_files),
            patch.object(analyzer, "_read_file", return_value=invalid_content),
        ):

            summary = await analyzer.analyze_library_usage("requests")

            # Should handle parse errors gracefully
            assert summary.target_library == "requests"
            assert summary.total_usages == 0

    def test_analyze_file_usage_import_statements(self, analyzer):
        """Test analyzing import statements correctly identifies usage types."""
        code = """
import requests
import requests.auth
from requests import Session
from requests.exceptions import RequestException
"""
        tree = ast.parse(code)
        usage_locations = analyzer._analyze_file_usage(
            tree, "requests", "/test/file.py"
        )

        assert len(usage_locations) == 4

        # Check correct usage types are assigned
        import_usage = next(
            loc for loc in usage_locations if loc.usage_context == "import requests"
        )
        assert import_usage.usage_type == LibraryUsageType.IMPORT
        assert import_usage.module_path == "requests"

        from_import_usage = next(
            loc
            for loc in usage_locations
            if "from requests import" in loc.usage_context
        )
        assert from_import_usage.usage_type == LibraryUsageType.FROM_IMPORT
        assert from_import_usage.module_path == "requests"

    def test_analyze_file_usage_import_aliases(self, analyzer):
        """Test analyzing import statements with aliases."""
        code = """
import requests as req
import pandas as pd
from requests import Session as ReqSession
from requests.exceptions import RequestException as ReqError
"""
        tree = ast.parse(code)

        # Test requests aliases
        requests_locations = analyzer._analyze_file_usage(
            tree, "requests", "/test/file.py"
        )
        assert (
            len(requests_locations) == 3
        )  # import requests as req, from requests import Session as ReqSession, from requests.exceptions import RequestException as ReqError

        # Check import alias is captured
        import_alias = next(
            loc for loc in requests_locations if loc.usage_context == "import requests"
        )
        assert import_alias.usage_type == LibraryUsageType.IMPORT
        assert import_alias.module_path == "requests"

        # Test pandas (should not match requests)
        pandas_locations = analyzer._analyze_file_usage(tree, "pandas", "/test/file.py")
        assert len(pandas_locations) == 1
        assert pandas_locations[0].usage_context == "import pandas"

    def test_analyze_file_usage_aliased_usage(self, analyzer):
        """Test that aliased imports don't trigger false positives in usage detection."""
        code = """
import requests as req
import pandas as pd

def test():
    # This should NOT be detected as requests usage since it's aliased
    response = req.get('url')
    
    # This should NOT be detected as requests usage
    df = pd.DataFrame()
    
    # This SHOULD be detected if we had 'import requests' without alias
    # But since we only have aliased import, the visitor won't detect this
    pass
"""
        tree = ast.parse(code)
        usage_locations = analyzer._analyze_file_usage(
            tree, "requests", "/test/file.py"
        )

        # Should only find the import statement, not the req.get() call
        # since the current implementation looks for direct library name matches
        assert len(usage_locations) == 1
        assert usage_locations[0].usage_type == LibraryUsageType.IMPORT

    def test_analyze_file_usage_function_calls(self, analyzer):
        """Test analyzing function calls correctly identifies method calls."""
        code = """
import requests

def test_function():
    response = requests.get('url')
    session = requests.Session()
    requests.post('url', data={})
"""
        tree = ast.parse(code)
        usage_locations = analyzer._analyze_file_usage(
            tree, "requests", "/test/file.py"
        )

        # Should find import + method calls
        method_calls = [
            loc
            for loc in usage_locations
            if loc.usage_type == LibraryUsageType.METHOD_CALL
        ]
        assert len(method_calls) >= 3

        # Check function context is captured correctly
        for loc in method_calls:
            assert loc.function_name == "test_function"

    def test_analyze_file_usage_class_context(self, analyzer):
        """Test analyzing usage within class context captures class and method names."""
        code = """
import requests

class TestClass:
    def method(self):
        return requests.get('url')
        
    async def async_method(self):
        return requests.post('url')
"""
        tree = ast.parse(code)
        usage_locations = analyzer._analyze_file_usage(
            tree, "requests", "/test/file.py"
        )

        method_calls = [
            loc
            for loc in usage_locations
            if loc.usage_type == LibraryUsageType.METHOD_CALL
        ]

        for loc in method_calls:
            assert loc.class_name == "TestClass"
            assert loc.function_name in ["method", "async_method"]

    def test_analyze_file_usage_attribute_access(self, analyzer):
        """Test analyzing attribute access correctly identifies attribute usage."""
        code = """
import requests

def test():
    exc = requests.exceptions.RequestException
    timeout = requests.adapters.DEFAULT_TIMEOUT
"""
        tree = ast.parse(code)
        usage_locations = analyzer._analyze_file_usage(
            tree, "requests", "/test/file.py"
        )

        attribute_accesses = [
            loc
            for loc in usage_locations
            if loc.usage_type == LibraryUsageType.ATTRIBUTE_ACCESS
        ]
        assert len(attribute_accesses) >= 2

    def test_analyze_file_usage_nested_attributes(self, analyzer):
        """Test analyzing nested attribute access is correctly identified."""
        code = """
import requests

def test():
    exc = requests.exceptions.ConnectionError
"""
        tree = ast.parse(code)
        usage_locations = analyzer._analyze_file_usage(
            tree, "requests", "/test/file.py"
        )

        # Should detect nested attribute access
        found_usage = any(
            "requests.exceptions" in loc.usage_context
            for loc in usage_locations
            if loc.usage_type == LibraryUsageType.ATTRIBUTE_ACCESS
        )
        assert found_usage

    @pytest.mark.asyncio
    async def test_get_python_files_with_mcp(self, analyzer):
        """Test getting Python files using MCP server."""
        analyzer.mcp_manager.is_server_available.return_value = True
        analyzer.mcp_manager.call_tool.return_value = {
            "files": ["/test/file1.py", "/test/file2.py"]
        }

        files = await analyzer._get_python_files()

        assert files == ["/test/file1.py", "/test/file2.py"]
        analyzer.mcp_manager.call_tool.assert_called_once_with(
            MCPServerType.FILESYSTEM,
            "find_files",
            {"pattern": "**/*.py", "directory": str(analyzer.project_root)},
        )

    @pytest.mark.asyncio
    async def test_get_python_files_fallback(self, analyzer, tmp_path):
        """Test getting Python files with fallback method."""
        # Create test files
        (tmp_path / "file1.py").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.py").touch()
        (tmp_path / "not_python.txt").touch()

        analyzer.mcp_manager.is_server_available.return_value = False

        files = await analyzer._get_python_files()

        # Should find .py files only
        assert len(files) == 2
        assert any("file1.py" in f for f in files)
        assert any("file2.py" in f for f in files)
        assert not any("not_python.txt" in f for f in files)

    @pytest.mark.asyncio
    async def test_get_python_files_mcp_error(self, analyzer, tmp_path):
        """Test getting Python files when MCP call fails."""
        # Create test file for fallback
        (tmp_path / "file.py").touch()

        analyzer.mcp_manager.is_server_available.return_value = True
        analyzer.mcp_manager.call_tool.side_effect = Exception("MCP error")

        files = await analyzer._get_python_files()

        # Should fall back to manual discovery
        assert len(files) == 1
        assert "file.py" in files[0]

    @pytest.mark.asyncio
    async def test_read_file_with_mcp(self, analyzer):
        """Test reading file using MCP server."""
        analyzer.mcp_manager.is_server_available.return_value = True
        analyzer.mcp_manager.call_tool.return_value = {
            "result": [{"text": "import requests"}]
        }

        content = await analyzer._read_file("/test/file.py")

        assert content == "import requests"
        analyzer.mcp_manager.call_tool.assert_called_once_with(
            MCPServerType.FILESYSTEM, "read_file", {"file_path": "/test/file.py"}
        )

    @pytest.mark.asyncio
    async def test_read_file_fallback(self, analyzer, tmp_path):
        """Test reading file with fallback method."""
        test_file = tmp_path / "file.py"
        test_content = "import requests\nrequests.get('url')"
        test_file.write_text(test_content)

        analyzer.mcp_manager.is_server_available.return_value = False

        content = await analyzer._read_file(str(test_file))

        assert content == test_content

    @pytest.mark.asyncio
    async def test_read_file_error(self, analyzer):
        """Test reading file when file doesn't exist."""
        analyzer.mcp_manager.is_server_available.return_value = False

        content = await analyzer._read_file("/nonexistent/file.py")

        assert content is None

    def test_get_affected_functions(self, analyzer):
        """Test getting affected functions from usage summary."""
        locations = [
            LibraryUsageLocation(
                file_path="/test/file.py",
                line_number=1,
                column_offset=0,
                usage_type=LibraryUsageType.METHOD_CALL,
                usage_context="requests.get('url')",
                function_name="test_func",
            ),
            LibraryUsageLocation(
                file_path="/test/file.py",
                line_number=2,
                column_offset=0,
                usage_type=LibraryUsageType.METHOD_CALL,
                usage_context="requests.post('url')",
                function_name="test_func",
            ),
            LibraryUsageLocation(
                file_path="/test/file.py",
                line_number=10,
                column_offset=0,
                usage_type=LibraryUsageType.METHOD_CALL,
                usage_context="requests.put('url')",
                function_name="other_func",
                class_name="TestClass",
            ),
        ]

        summary = LibraryUsageSummary(
            target_library="requests",
            total_usages=3,
            usage_locations=locations,
        )

        affected_functions = analyzer.get_affected_functions(summary)

        assert len(affected_functions) == 2
        assert "/test/file.py::test_func" in affected_functions
        assert "/test/file.py::TestClass::other_func" in affected_functions
        assert len(affected_functions["/test/file.py::test_func"]) == 2
        assert len(affected_functions["/test/file.py::TestClass::other_func"]) == 1

    def test_get_usage_statistics(self, analyzer):
        """Test getting usage statistics."""
        locations = [
            LibraryUsageLocation(
                file_path="/test/file1.py",
                line_number=1,
                column_offset=0,
                usage_type=LibraryUsageType.IMPORT,
                usage_context="import requests",
            ),
            LibraryUsageLocation(
                file_path="/test/file1.py",
                line_number=2,
                column_offset=0,
                usage_type=LibraryUsageType.METHOD_CALL,
                usage_context="requests.get('url')",
            ),
            LibraryUsageLocation(
                file_path="/test/file2.py",
                line_number=1,
                column_offset=0,
                usage_type=LibraryUsageType.METHOD_CALL,
                usage_context="requests.post('url')",
            ),
        ]

        summary = LibraryUsageSummary(
            target_library="requests",
            total_usages=3,
            usage_locations=locations,
            imported_modules={"requests"},
            used_functions={"get", "post"},
            used_classes={"Session"},
            affected_files={"/test/file1.py", "/test/file2.py"},
        )

        stats = analyzer.get_usage_statistics(summary)

        assert stats["total_usages"] == 3
        assert stats["affected_files"] == 2
        assert stats["imported_modules"] == 1
        assert stats["used_functions"] == 2
        assert stats["used_classes"] == 1
        assert stats["usage_by_type"]["import"] == 1
        assert stats["usage_by_type"]["method_call"] == 2
        assert stats["usage_by_file"]["/test/file1.py"] == 2
        assert stats["usage_by_file"]["/test/file2.py"] == 1
        assert len(stats["most_used_files"]) == 2

    def test_library_usage_visitor_complex_code(self, analyzer):
        """Test LibraryUsageVisitor correctly categorizes complex real-world usage patterns."""
        code = """
import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException, Timeout

class APIClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth('user', 'pass')
    
    async def fetch_data(self, url):
        try:
            response = self.session.get(url, timeout=30)
            return response.json()
        except (RequestException, Timeout) as e:
            self.handle_error(e)
            
    def handle_error(self, error):
        if isinstance(error, requests.exceptions.Timeout):
            print("Request timed out")
        elif isinstance(error, requests.exceptions.ConnectionError):
            print("Connection failed")

def global_function():
    response = requests.post('https://api.example.com', 
                           headers={'Content-Type': 'application/json'})
    return response.status_code
"""
        tree = ast.parse(code)
        usage_locations = analyzer._analyze_file_usage(
            tree, "requests", "/test/complex.py"
        )

        # Should find various usage patterns
        assert len(usage_locations) > 5

        # Check different usage types are correctly detected
        usage_types = {loc.usage_type for loc in usage_locations}
        assert LibraryUsageType.IMPORT in usage_types
        assert LibraryUsageType.FROM_IMPORT in usage_types
        assert LibraryUsageType.METHOD_CALL in usage_types
        assert LibraryUsageType.ATTRIBUTE_ACCESS in usage_types

        # Check context information is correctly captured
        class_usages = [loc for loc in usage_locations if loc.class_name == "APIClient"]
        function_usages = [
            loc for loc in usage_locations if loc.function_name == "global_function"
        ]

        assert len(class_usages) > 0
        assert len(function_usages) > 0

        # Verify import statements are correctly categorized
        import_usages = [
            loc for loc in usage_locations if loc.usage_type == LibraryUsageType.IMPORT
        ]
        from_import_usages = [
            loc
            for loc in usage_locations
            if loc.usage_type == LibraryUsageType.FROM_IMPORT
        ]

        assert len(import_usages) >= 1  # import requests
        assert len(from_import_usages) >= 2  # from requests import statements
