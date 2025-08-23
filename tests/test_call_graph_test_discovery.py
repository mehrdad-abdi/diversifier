"""Tests for CallGraphTestDiscoveryAnalyzer."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.orchestration.test_generation.call_graph_test_discovery import (
    CallGraphTestDiscoveryAnalyzer,
    CallGraphBuilder,
    CallGraphNode,
    NodeType,
    CoveragePath,
)
from src.orchestration.test_generation.library_usage_analyzer import (
    LibraryUsageSummary,
    LibraryUsageLocation,
    LibraryUsageType,
)
from src.orchestration.mcp_manager import MCPManager


class TestCallGraphBuilder:
    """Test CallGraphBuilder class."""

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = AsyncMock(spec=MCPManager)
        manager.is_server_available.return_value = False
        return manager

    @pytest.fixture
    def builder(self, mock_mcp_manager, tmp_path):
        """Create a CallGraphBuilder instance for testing."""
        return CallGraphBuilder(str(tmp_path), mock_mcp_manager, ["tests"])

    def test_init(self, mock_mcp_manager, tmp_path):
        """Test CallGraphBuilder initialization."""
        builder = CallGraphBuilder(str(tmp_path), mock_mcp_manager, ["tests"])

        assert builder.project_root == Path(tmp_path)
        assert builder.mcp_manager == mock_mcp_manager
        assert builder.nodes == {}

    @pytest.mark.asyncio
    async def test_create_nodes_from_simple_file(self, builder):
        """Test creating nodes from a simple Python file."""
        file_content = """
def helper_function():
    return "helper"

class MyClass:
    def method1(self):
        return "method1"
    
    async def async_method(self):
        return "async"

def test_something():
    assert True

def TestClassExample():
    pass
"""

        with patch.object(builder, "_read_file", return_value=file_content):
            await builder._create_nodes_from_file("/src/example.py")

        # Should create 5 nodes
        assert len(builder.nodes) == 5

        # Check function node
        helper_node = builder.nodes["/src/example.py::helper_function"]
        assert helper_node.function_name == "helper_function"
        assert helper_node.class_name is None
        assert helper_node.node_type == NodeType.FUNCTION

        # Check method node
        method_node = builder.nodes["/src/example.py::MyClass::method1"]
        assert method_node.function_name == "method1"
        assert method_node.class_name == "MyClass"
        assert method_node.node_type == NodeType.METHOD

        # Check async method node
        async_node = builder.nodes["/src/example.py::MyClass::async_method"]
        assert async_node.node_type == NodeType.METHOD

        # Check test-named functions in source (should be regular functions, not test functions)
        test_node = builder.nodes["/src/example.py::test_something"]
        assert (
            test_node.node_type == NodeType.FUNCTION
        )  # Not TEST_FUNCTION because not in test dir

        test_class_node = builder.nodes["/src/example.py::TestClassExample"]
        assert (
            test_class_node.node_type == NodeType.FUNCTION
        )  # Not TEST_FUNCTION because not in test dir

    @pytest.mark.asyncio
    async def test_create_nodes_from_test_file(self, builder):
        """Test creating nodes from a file in the test directory."""
        test_file_content = """
def test_function():
    assert True

def TestClass():
    pass
    
def regular_function():
    return "not a test"
"""

        with patch.object(builder, "_read_file", return_value=test_file_content):
            await builder._create_nodes_from_file("/tests/test_module.py")

        # Should create 3 nodes
        assert len(builder.nodes) == 3

        # Check test function is identified as TEST_FUNCTION
        test_node = builder.nodes["/tests/test_module.py::test_function"]
        assert test_node.node_type == NodeType.TEST_FUNCTION

        test_class_node = builder.nodes["/tests/test_module.py::TestClass"]
        assert test_class_node.node_type == NodeType.TEST_FUNCTION

        # Regular function should still be FUNCTION even in test directory
        regular_node = builder.nodes["/tests/test_module.py::regular_function"]
        assert regular_node.node_type == NodeType.FUNCTION

    @pytest.mark.asyncio
    async def test_establish_call_relationships_simple(self, builder):
        """Test establishing call relationships between functions."""
        file_content = """
def low_level():
    return "data"

def mid_level():
    return low_level()

def high_level():
    result = mid_level()
    return result

def test_high_level():
    assert high_level() == "data"
"""

        # First create nodes
        with patch.object(builder, "_read_file", return_value=file_content):
            await builder._create_nodes_from_file("/src/example.py")
            await builder._establish_call_relationships("/src/example.py")

        # Check call relationships
        mid_node = builder.nodes["/src/example.py::mid_level"]
        high_node = builder.nodes["/src/example.py::high_level"]
        test_node = builder.nodes["/src/example.py::test_high_level"]
        low_node = builder.nodes["/src/example.py::low_level"]

        # mid_level calls low_level
        assert "low_level" in mid_node.calls
        assert "/src/example.py::mid_level" in low_node.called_by

        # high_level calls mid_level
        assert "mid_level" in high_node.calls
        assert "/src/example.py::high_level" in mid_node.called_by

        # test calls high_level
        assert "high_level" in test_node.calls
        assert "/src/example.py::test_high_level" in high_node.called_by

    @pytest.mark.asyncio
    async def test_establish_call_relationships_method_calls(self, builder):
        """Test establishing call relationships with method calls."""
        file_content = """
class APIClient:
    def fetch_data(self):
        return self.make_request()
    
    def make_request(self):
        import requests
        return requests.get("url")

def test_api():
    client = APIClient()
    result = client.fetch_data()
    return result
"""

        with patch.object(builder, "_read_file", return_value=file_content):
            await builder._create_nodes_from_file("/src/api.py")
            await builder._establish_call_relationships("/src/api.py")

        fetch_node = builder.nodes["/src/api.py::APIClient::fetch_data"]
        test_node = builder.nodes["/src/api.py::test_api"]

        # fetch_data calls self.make_request (should be captured as make_request)
        assert "self.make_request" in fetch_node.calls

        # test_api calls client.fetch_data (should be captured as client.fetch_data)
        assert "client.fetch_data" in test_node.calls

    @pytest.mark.asyncio
    async def test_build_complete_call_graph(self, builder):
        """Test building a complete call graph from multiple files."""
        # Mock multiple files
        source_content = """
def process_data():
    return fetch_from_api()

def fetch_from_api():
    import requests
    return requests.get("api/data")
"""

        test_content = """
from src.api import process_data

def test_data_processing():
    result = process_data()
    assert result is not None

def test_direct_api():
    from src.api import fetch_from_api
    data = fetch_from_api()
    assert data
"""

        files = ["/src/api.py", "/tests/test_api.py"]
        file_contents = {
            "/src/api.py": source_content,
            "/tests/test_api.py": test_content,
        }

        async def mock_read_file(file_path):
            return file_contents.get(file_path, "")

        with (
            patch.object(builder, "_get_all_python_files", return_value=files),
            patch.object(builder, "_read_file", side_effect=mock_read_file),
        ):

            call_graph = await builder.build_call_graph()

        # Should have nodes from both files
        assert len(call_graph) == 4  # 2 from source + 2 from tests

        # Check source nodes exist
        assert "/src/api.py::process_data" in call_graph
        assert "/src/api.py::fetch_from_api" in call_graph

        # Check test nodes exist and are marked as test functions
        assert "/tests/test_api.py::test_data_processing" in call_graph
        test_node = call_graph["/tests/test_api.py::test_data_processing"]
        assert test_node.node_type == NodeType.TEST_FUNCTION


class TestCallGraphTestDiscoveryAnalyzer:
    """Test CallGraphTestDiscoveryAnalyzer class."""

    @pytest.fixture
    def mock_mcp_manager(self):
        """Create a mock MCP manager."""
        manager = AsyncMock(spec=MCPManager)
        manager.is_server_available.return_value = False
        return manager

    @pytest.fixture
    def analyzer(self, mock_mcp_manager, tmp_path):
        """Create a CallGraphTestDiscoveryAnalyzer instance for testing."""
        return CallGraphTestDiscoveryAnalyzer(
            str(tmp_path), mock_mcp_manager, ["tests"]
        )

    @pytest.fixture
    def sample_call_graph(self):
        """Create a sample call graph for testing."""
        nodes = {
            "/src/api.py::fetch_data": CallGraphNode(
                file_path="/src/api.py",
                function_name="fetch_data",
                class_name="APIClient",
                line_number=10,
                node_type=NodeType.METHOD,
                calls={"make_request"},
                called_by={"/tests/test_api.py::test_fetch_data"},
            ),
            "/src/api.py::make_request": CallGraphNode(
                file_path="/src/api.py",
                function_name="make_request",
                class_name="APIClient",
                line_number=15,
                node_type=NodeType.METHOD,
                calls=set(),  # This calls requests.get (library usage)
                called_by={"/src/api.py::fetch_data"},
            ),
            "/tests/test_api.py::test_fetch_data": CallGraphNode(
                file_path="/tests/test_api.py",
                function_name="test_fetch_data",
                class_name=None,
                line_number=5,
                node_type=NodeType.TEST_FUNCTION,
                calls={"fetch_data"},
                called_by=set(),
            ),
            "/src/utils.py::helper": CallGraphNode(
                file_path="/src/utils.py",
                function_name="helper",
                class_name=None,
                line_number=8,
                node_type=NodeType.FUNCTION,
                calls=set(),  # This calls requests.post (library usage)
                called_by=set(),  # No callers - uncovered
            ),
        }
        return nodes

    @pytest.fixture
    def sample_library_usage(self):
        """Create sample library usage for testing."""
        usage_locations = [
            LibraryUsageLocation(
                file_path="/src/api.py",
                line_number=17,
                column_offset=8,
                usage_type=LibraryUsageType.METHOD_CALL,
                usage_context="requests.get('api/data')",
                function_name="make_request",
                class_name="APIClient",
            ),
            LibraryUsageLocation(
                file_path="/src/utils.py",
                line_number=10,
                column_offset=4,
                usage_type=LibraryUsageType.METHOD_CALL,
                usage_context="requests.post('api/submit')",
                function_name="helper",
                class_name=None,
            ),
        ]

        return LibraryUsageSummary(
            target_library="requests",
            total_usages=2,
            usage_locations=usage_locations,
            affected_files={"/src/api.py", "/src/utils.py"},
        )

    @pytest.mark.asyncio
    async def test_discover_test_coverage_with_call_chain(
        self, analyzer, sample_library_usage, sample_call_graph
    ):
        """Test discovering test coverage through call chains."""
        # Mock the call graph
        analyzer.call_graph = sample_call_graph

        with patch.object(
            analyzer.call_graph_builder,
            "build_call_graph",
            return_value=sample_call_graph,
        ):
            result = await analyzer.discover_test_coverage(sample_library_usage)

        # Should find one coverage path: test_fetch_data -> fetch_data -> make_request (library usage)
        assert len(result.coverage_paths) == 1

        coverage_path = result.coverage_paths[0]
        assert coverage_path.test_node.function_name == "test_fetch_data"
        assert coverage_path.library_usage.function_name == "make_request"
        assert coverage_path.depth >= 1  # Should have intermediate nodes

        # Should have one uncovered usage (helper function)
        assert len(result.uncovered_usages) == 1
        assert result.uncovered_usages[0].function_name == "helper"

        # Coverage percentage should be 50% (1 out of 2 usages covered)
        assert result.coverage_percentage == 50.0

    @pytest.mark.asyncio
    async def test_find_test_coverage_paths_deep_chain(self, analyzer):
        """Test finding coverage paths through deep call chains."""
        # Create a deeper call chain: test -> level3 -> level2 -> level1 (library usage)
        deep_call_graph = {
            "/src/deep.py::level1": CallGraphNode(
                file_path="/src/deep.py",
                function_name="level1",
                class_name=None,
                line_number=5,
                node_type=NodeType.FUNCTION,
                calls=set(),  # Library usage here
                called_by={"/src/deep.py::level2"},
            ),
            "/src/deep.py::level2": CallGraphNode(
                file_path="/src/deep.py",
                function_name="level2",
                class_name=None,
                line_number=10,
                node_type=NodeType.FUNCTION,
                calls={"level1"},
                called_by={"/src/deep.py::level3"},
            ),
            "/src/deep.py::level3": CallGraphNode(
                file_path="/src/deep.py",
                function_name="level3",
                class_name=None,
                line_number=15,
                node_type=NodeType.FUNCTION,
                calls={"level2"},
                called_by={"/tests/test_deep.py::test_deep_feature"},
            ),
            "/tests/test_deep.py::test_deep_feature": CallGraphNode(
                file_path="/tests/test_deep.py",
                function_name="test_deep_feature",
                class_name=None,
                line_number=8,
                node_type=NodeType.TEST_FUNCTION,
                calls={"level3"},
                called_by=set(),
            ),
        }

        analyzer.call_graph = deep_call_graph

        usage_location = LibraryUsageLocation(
            file_path="/src/deep.py",
            line_number=7,
            column_offset=4,
            usage_type=LibraryUsageType.METHOD_CALL,
            usage_context="requests.get('deep')",
            function_name="level1",
            class_name=None,
        )

        paths = await analyzer._find_test_coverage_paths(usage_location)

        assert len(paths) == 1
        path = paths[0]
        assert path.test_node.function_name == "test_deep_feature"
        assert path.library_usage.function_name == "level1"
        assert path.depth == 4  # test -> level3 -> level2 -> level1

        # Check the call chain is correct
        call_chain_functions = [node.function_name for node in path.call_chain]
        assert call_chain_functions == [
            "test_deep_feature",
            "level3",
            "level2",
            "level1",
        ]

    @pytest.mark.asyncio
    async def test_find_test_coverage_paths_multiple_tests(self, analyzer):
        """Test finding multiple test coverage paths for the same library usage."""
        # Create scenario where one library usage is covered by multiple tests
        multi_test_graph = {
            "/src/shared.py::shared_function": CallGraphNode(
                file_path="/src/shared.py",
                function_name="shared_function",
                class_name=None,
                line_number=5,
                node_type=NodeType.FUNCTION,
                calls=set(),  # Library usage here
                called_by={
                    "/tests/test1.py::test_feature_a",
                    "/tests/test2.py::test_feature_b",
                },
            ),
            "/tests/test1.py::test_feature_a": CallGraphNode(
                file_path="/tests/test1.py",
                function_name="test_feature_a",
                class_name=None,
                line_number=10,
                node_type=NodeType.TEST_FUNCTION,
                calls={"shared_function"},
                called_by=set(),
            ),
            "/tests/test2.py::test_feature_b": CallGraphNode(
                file_path="/tests/test2.py",
                function_name="test_feature_b",
                class_name=None,
                line_number=15,
                node_type=NodeType.TEST_FUNCTION,
                calls={"shared_function"},
                called_by=set(),
            ),
        }

        analyzer.call_graph = multi_test_graph

        usage_location = LibraryUsageLocation(
            file_path="/src/shared.py",
            line_number=7,
            column_offset=4,
            usage_type=LibraryUsageType.METHOD_CALL,
            usage_context="requests.post('shared')",
            function_name="shared_function",
            class_name=None,
        )

        paths = await analyzer._find_test_coverage_paths(usage_location)

        # Should find 2 coverage paths
        assert len(paths) == 2

        test_names = {path.test_node.function_name for path in paths}
        assert test_names == {"test_feature_a", "test_feature_b"}

    @pytest.mark.asyncio
    async def test_find_test_coverage_paths_no_coverage(self, analyzer):
        """Test finding coverage paths when no tests cover the usage."""
        isolated_graph = {
            "/src/isolated.py::isolated_function": CallGraphNode(
                file_path="/src/isolated.py",
                function_name="isolated_function",
                class_name=None,
                line_number=5,
                node_type=NodeType.FUNCTION,
                calls=set(),  # Library usage here
                called_by=set(),  # No callers
            )
        }

        analyzer.call_graph = isolated_graph

        usage_location = LibraryUsageLocation(
            file_path="/src/isolated.py",
            line_number=7,
            column_offset=4,
            usage_type=LibraryUsageType.METHOD_CALL,
            usage_context="requests.delete('isolated')",
            function_name="isolated_function",
            class_name=None,
        )

        paths = await analyzer._find_test_coverage_paths(usage_location)

        # Should find no coverage paths
        assert len(paths) == 0

    def test_find_library_usage_nodes(
        self, analyzer, sample_library_usage, sample_call_graph
    ):
        """Test finding nodes that contain library usage points."""
        analyzer.call_graph = sample_call_graph

        usage_nodes = analyzer._find_library_usage_nodes(sample_library_usage)

        # Should find 2 usage nodes
        assert len(usage_nodes) == 2

        node_functions = {node.function_name for node in usage_nodes}
        assert node_functions == {"make_request", "helper"}

    def test_call_graph_node_full_name(self):
        """Test CallGraphNode full_name property."""
        # Function node
        func_node = CallGraphNode(
            file_path="/src/utils.py",
            function_name="helper",
            class_name=None,
            line_number=10,
            node_type=NodeType.FUNCTION,
        )
        assert func_node.full_name == "/src/utils.py::helper"

        # Method node
        method_node = CallGraphNode(
            file_path="/src/api.py",
            function_name="fetch",
            class_name="APIClient",
            line_number=15,
            node_type=NodeType.METHOD,
        )
        assert method_node.full_name == "/src/api.py::APIClient::fetch"

    def test_coverage_path_depth(self):
        """Test CoveragePath depth calculation."""
        test_node = CallGraphNode(
            file_path="/tests/test.py",
            function_name="test_func",
            class_name=None,
            line_number=5,
            node_type=NodeType.TEST_FUNCTION,
        )

        usage_location = LibraryUsageLocation(
            file_path="/src/api.py",
            line_number=10,
            column_offset=4,
            usage_type=LibraryUsageType.METHOD_CALL,
            usage_context="requests.get('url')",
            function_name="api_call",
            class_name=None,
        )

        # Direct call (depth 1)
        direct_path = CoveragePath(
            test_node=test_node, library_usage=usage_location, call_chain=[test_node]
        )
        assert direct_path.depth == 1

        # Indirect call through 2 intermediate functions (depth 3)
        intermediate1 = CallGraphNode(
            "/src/mid1.py", "func1", None, 8, NodeType.FUNCTION
        )
        intermediate2 = CallGraphNode(
            "/src/mid2.py", "func2", None, 12, NodeType.FUNCTION
        )

        indirect_path = CoveragePath(
            test_node=test_node,
            library_usage=usage_location,
            call_chain=[test_node, intermediate1, intermediate2],
        )
        assert indirect_path.depth == 3

    @pytest.mark.asyncio
    async def test_circular_call_graph_no_infinite_loop(self, analyzer):
        """Test that circular call graphs don't cause infinite loops."""
        # Create circular call graph: A → B → C → A with test entry point
        circular_graph = {
            "/src/a.py::func_a": CallGraphNode(
                file_path="/src/a.py",
                function_name="func_a",
                class_name=None,
                line_number=5,
                node_type=NodeType.FUNCTION,
                calls={"func_b"},
                called_by={
                    "/src/c.py::func_c",
                    "/tests/test.py::test_func",
                },  # Cycle + test entry
            ),
            "/src/b.py::func_b": CallGraphNode(
                file_path="/src/b.py",
                function_name="func_b",
                class_name=None,
                line_number=5,
                node_type=NodeType.FUNCTION,
                calls={"func_c"},
                called_by={"/src/a.py::func_a"},
            ),
            "/src/c.py::func_c": CallGraphNode(
                file_path="/src/c.py",
                function_name="func_c",
                class_name=None,
                line_number=5,
                node_type=NodeType.FUNCTION,
                calls={"func_a"},  # Creates cycle back to A
                called_by={"/src/b.py::func_b"},
            ),
            "/tests/test.py::test_func": CallGraphNode(
                file_path="/tests/test.py",
                function_name="test_func",
                class_name=None,
                line_number=5,
                node_type=NodeType.TEST_FUNCTION,
                calls={"func_a"},
                called_by=set(),
            ),
        }

        analyzer.call_graph = circular_graph

        # Library usage in func_a (part of the cycle)
        usage_location = LibraryUsageLocation(
            file_path="/src/a.py",
            line_number=7,
            column_offset=4,
            usage_type=LibraryUsageType.METHOD_CALL,
            usage_context="requests.get('url')",
            function_name="func_a",
            class_name=None,
        )

        # This should NOT hang in infinite loop and should find the test
        paths = await analyzer._find_test_coverage_paths(usage_location)

        assert len(paths) == 1
        assert paths[0].test_node.function_name == "test_func"
        # Should contain the usage node in call chain
        assert any(node.function_name == "func_a" for node in paths[0].call_chain)

    @pytest.mark.asyncio
    async def test_multiple_paths_shared_function_bug_fix(self, analyzer):
        """Test fix for bug where shared visited set prevented multiple path discovery."""
        # This tests the critical bug fix where multiple tests calling the same function
        # were not all discovered due to shared visited set

        shared_func_graph = {
            "/src/shared.py::shared_func": CallGraphNode(
                file_path="/src/shared.py",
                function_name="shared_func",
                class_name=None,
                line_number=5,
                node_type=NodeType.FUNCTION,
                calls=set(),  # Library usage here
                called_by={
                    "/tests/test1.py::test_a",
                    "/tests/test2.py::test_b",
                    "/tests/test3.py::test_c",
                },
            ),
            "/tests/test1.py::test_a": CallGraphNode(
                file_path="/tests/test1.py",
                function_name="test_a",
                class_name=None,
                line_number=5,
                node_type=NodeType.TEST_FUNCTION,
                calls={"shared_func"},
                called_by=set(),
            ),
            "/tests/test2.py::test_b": CallGraphNode(
                file_path="/tests/test2.py",
                function_name="test_b",
                class_name=None,
                line_number=5,
                node_type=NodeType.TEST_FUNCTION,
                calls={"shared_func"},
                called_by=set(),
            ),
            "/tests/test3.py::test_c": CallGraphNode(
                file_path="/tests/test3.py",
                function_name="test_c",
                class_name=None,
                line_number=5,
                node_type=NodeType.TEST_FUNCTION,
                calls={"shared_func"},
                called_by=set(),
            ),
        }

        analyzer.call_graph = shared_func_graph

        usage_location = LibraryUsageLocation(
            file_path="/src/shared.py",
            line_number=7,
            column_offset=4,
            usage_type=LibraryUsageType.METHOD_CALL,
            usage_context="requests.post('data')",
            function_name="shared_func",
            class_name=None,
        )

        paths = await analyzer._find_test_coverage_paths(usage_location)

        # Should find ALL test functions, not just the first one
        assert len(paths) == 3

        test_names = {path.test_node.function_name for path in paths}
        assert test_names == {"test_a", "test_b", "test_c"}

        # All paths should have depth 2 (test → shared_func)
        for path in paths:
            assert path.depth == 2

    @pytest.mark.asyncio
    async def test_complex_multiple_entry_points(self, analyzer):
        """Test complex scenario with multiple entry points at different call chain depths."""
        # Complex graph testing multiple paths with different depths:
        # test1 → high → mid → low (library usage) [depth 4]
        # test2 → mid → low (library usage)         [depth 3]
        # test3 → low (library usage)               [depth 2]

        complex_graph = {
            "/src/deep.py::low_level": CallGraphNode(
                file_path="/src/deep.py",
                function_name="low_level",
                class_name=None,
                line_number=5,
                node_type=NodeType.FUNCTION,
                calls=set(),  # Library usage here
                called_by={"/src/deep.py::mid_level", "/tests/test3.py::test_direct"},
            ),
            "/src/deep.py::mid_level": CallGraphNode(
                file_path="/src/deep.py",
                function_name="mid_level",
                class_name=None,
                line_number=10,
                node_type=NodeType.FUNCTION,
                calls={"low_level"},
                called_by={"/src/deep.py::high_level", "/tests/test2.py::test_mid"},
            ),
            "/src/deep.py::high_level": CallGraphNode(
                file_path="/src/deep.py",
                function_name="high_level",
                class_name=None,
                line_number=15,
                node_type=NodeType.FUNCTION,
                calls={"mid_level"},
                called_by={"/tests/test1.py::test_high"},
            ),
            "/tests/test1.py::test_high": CallGraphNode(
                file_path="/tests/test1.py",
                function_name="test_high",
                class_name=None,
                line_number=5,
                node_type=NodeType.TEST_FUNCTION,
                calls={"high_level"},
                called_by=set(),
            ),
            "/tests/test2.py::test_mid": CallGraphNode(
                file_path="/tests/test2.py",
                function_name="test_mid",
                class_name=None,
                line_number=5,
                node_type=NodeType.TEST_FUNCTION,
                calls={"mid_level"},
                called_by=set(),
            ),
            "/tests/test3.py::test_direct": CallGraphNode(
                file_path="/tests/test3.py",
                function_name="test_direct",
                class_name=None,
                line_number=5,
                node_type=NodeType.TEST_FUNCTION,
                calls={"low_level"},
                called_by=set(),
            ),
        }

        analyzer.call_graph = complex_graph

        usage_location = LibraryUsageLocation(
            file_path="/src/deep.py",
            line_number=7,
            column_offset=4,
            usage_type=LibraryUsageType.METHOD_CALL,
            usage_context="requests.get('deep')",
            function_name="low_level",
            class_name=None,
        )

        paths = await analyzer._find_test_coverage_paths(usage_location)

        # Should find all 3 test functions with different depths
        assert len(paths) == 3

        test_names = {path.test_node.function_name for path in paths}
        assert test_names == {"test_high", "test_mid", "test_direct"}

        # Verify depths are correct
        depths = {path.test_node.function_name: path.depth for path in paths}
        assert depths["test_direct"] == 2  # test_direct → low_level
        assert depths["test_mid"] == 3  # test_mid → mid_level → low_level
        assert (
            depths["test_high"] == 4
        )  # test_high → high_level → mid_level → low_level

        # Verify call chains are correct
        for path in paths:
            if path.test_node.function_name == "test_high":
                expected_chain = ["test_high", "high_level", "mid_level", "low_level"]
                actual_chain = [node.function_name for node in path.call_chain]
                assert actual_chain == expected_chain
            elif path.test_node.function_name == "test_mid":
                expected_chain = ["test_mid", "mid_level", "low_level"]
                actual_chain = [node.function_name for node in path.call_chain]
                assert actual_chain == expected_chain
            elif path.test_node.function_name == "test_direct":
                expected_chain = ["test_direct", "low_level"]
                actual_chain = [node.function_name for node in path.call_chain]
                assert actual_chain == expected_chain

    def test_configurable_test_path(self, mock_mcp_manager, tmp_path):
        """Test that CallGraphTestDiscoveryAnalyzer uses configurable test paths."""
        # Test with custom test paths
        custom_test_paths = ["app/tests"]
        analyzer = CallGraphTestDiscoveryAnalyzer(
            str(tmp_path), mock_mcp_manager, custom_test_paths
        )

        assert analyzer.test_paths == custom_test_paths
        assert analyzer.call_graph_builder.test_paths == ["app/tests"]

        # Test _is_test_file method
        assert analyzer.call_graph_builder._is_test_file("app/tests/test_module.py")
        assert analyzer.call_graph_builder._is_test_file(
            "app/tests/subfolder/test_utils.py"
        )
        assert not analyzer.call_graph_builder._is_test_file("tests/test_module.py")
        assert not analyzer.call_graph_builder._is_test_file("src/module.py")
        assert not analyzer.call_graph_builder._is_test_file("app/src/utils.py")

    def test_multiple_test_paths(self, mock_mcp_manager, tmp_path):
        """Test that CallGraphTestDiscoveryAnalyzer supports multiple test directories."""
        # Test with multiple test paths
        test_paths = ["tests", "app/tests", "backend/tests"]
        analyzer = CallGraphTestDiscoveryAnalyzer(
            str(tmp_path), mock_mcp_manager, test_paths
        )

        assert analyzer.test_paths == test_paths
        assert analyzer.call_graph_builder.test_paths == test_paths

        # Test _is_test_file method with multiple paths
        test_cases = [
            ("tests/test_api.py", True),  # First test path
            ("app/tests/test_models.py", True),  # Second test path
            ("backend/tests/test_views.py", True),  # Third test path
            ("tests/unit/test_utils.py", True),  # Subdirectory in first path
            ("app/tests/integration/test_e2e.py", True),  # Subdirectory in second path
            ("src/api.py", False),  # Not in any test path
            ("other/test_something.py", False),  # Not in configured test paths
            ("app/src/test_helper.py", False),  # Not in test directory
        ]

        for file_path, expected in test_cases:
            result = analyzer.call_graph_builder._is_test_file(file_path)
            assert (
                result == expected
            ), f"Expected {expected} for path '{file_path}', got {result}"
