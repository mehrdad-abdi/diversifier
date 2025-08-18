"""Call graph-based test discovery for finding test coverage of library usage points."""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum

from ..mcp_manager import MCPManager, MCPServerType
from .library_usage_analyzer import LibraryUsageSummary, LibraryUsageLocation


class NodeType(Enum):
    """Types of nodes in the call graph."""

    FUNCTION = "function"
    METHOD = "method"
    TEST_FUNCTION = "test_function"


@dataclass
class CallGraphNode:
    """Represents a function/method in the call graph."""

    file_path: str
    function_name: str
    class_name: Optional[str]
    line_number: int
    node_type: NodeType
    calls: Set[str] = field(default_factory=set)  # Functions this node calls
    called_by: Set[str] = field(default_factory=set)  # Functions that call this node

    @property
    def full_name(self) -> str:
        """Get the full qualified name of this node."""
        if self.class_name:
            return f"{self.file_path}::{self.class_name}::{self.function_name}"
        return f"{self.file_path}::{self.function_name}"


@dataclass
class CoveragePath:
    """Represents a path from a test function to library usage."""

    test_node: CallGraphNode
    library_usage: LibraryUsageLocation
    call_chain: List[CallGraphNode] = field(
        default_factory=list
    )  # Path from test to usage

    @property
    def depth(self) -> int:
        """Get the depth of the call chain."""
        return len(self.call_chain)


@dataclass
class CallGraphTestDiscoveryResult:
    """Results of call graph-based test discovery."""

    total_nodes: int
    test_nodes: int
    library_usage_nodes: int
    coverage_paths: List[CoveragePath] = field(default_factory=list)
    uncovered_usages: List[LibraryUsageLocation] = field(default_factory=list)
    coverage_percentage: float = 0.0


class CallGraphBuilder:
    """Builds a call graph from Python source code using AST analysis."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize the call graph builder.

        Args:
            project_root: Root directory of the project
            mcp_manager: MCP manager for file system operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.call_graph_builder")
        self.nodes: Dict[str, CallGraphNode] = {}  # full_name -> CallGraphNode

    async def build_call_graph(self) -> Dict[str, CallGraphNode]:
        """Build the complete call graph for the project.

        Returns:
            Dictionary mapping full function names to CallGraphNode objects
        """
        self.logger.info("Building call graph for project")

        # Get all Python files (source + tests)
        python_files = await self._get_all_python_files()

        # First pass: Create nodes for all functions/methods
        for file_path in python_files:
            await self._create_nodes_from_file(file_path)

        # Second pass: Establish call relationships
        for file_path in python_files:
            await self._establish_call_relationships(file_path)

        self.logger.info(f"Built call graph with {len(self.nodes)} nodes")
        return self.nodes

    async def _get_all_python_files(self) -> List[str]:
        """Get all Python files in the project (source + tests)."""
        try:
            if self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
                result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "find_python_files",
                    {},
                )

                if result and "result" in result and "content" in result["result"]:
                    import json
                    content = result["result"]["content"][0]["text"]
                    data = json.loads(content)
                    if "files" in data:
                        return [str(self.project_root / file_info["path"]) for file_info in data["files"]]

            # Fallback to manual discovery
            return [str(p) for p in self.project_root.rglob("*.py") if p.is_file()]

        except Exception as e:
            self.logger.warning(f"Error finding Python files: {e}")
            return [str(p) for p in self.project_root.rglob("*.py") if p.is_file()]

    async def _create_nodes_from_file(self, file_path: str):
        """Create call graph nodes for all functions in a file."""
        try:
            content = await self._read_file(file_path)
            if not content:
                return

            tree = ast.parse(content, filename=file_path)

            class NodeCreatorVisitor(ast.NodeVisitor):
                def __init__(self, builder):
                    self.builder = builder
                    self.current_class = None

                def visit_ClassDef(self, node: ast.ClassDef):
                    old_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = old_class

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self._create_function_node(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    self._create_function_node(node)

                def _create_function_node(self, node):
                    # Determine node type
                    node_type = NodeType.FUNCTION
                    if self.current_class:
                        node_type = NodeType.METHOD
                    if self._is_test_function(node.name):
                        node_type = NodeType.TEST_FUNCTION

                    # Create the node
                    call_node = CallGraphNode(
                        file_path=file_path,
                        function_name=node.name,
                        class_name=self.current_class,
                        line_number=node.lineno,
                        node_type=node_type,
                    )

                    self.builder.nodes[call_node.full_name] = call_node

                def _is_test_function(self, name: str) -> bool:
                    """Check if a function name indicates it's a test function."""
                    return (
                        name.startswith("test_")
                        or name.endswith("_test")
                        or name.startswith("Test")
                    )

            visitor = NodeCreatorVisitor(self)
            visitor.visit(tree)

        except Exception as e:
            self.logger.warning(f"Error creating nodes from file {file_path}: {e}")

    async def _establish_call_relationships(self, file_path: str):
        """Establish call relationships between nodes in a file."""
        try:
            content = await self._read_file(file_path)
            if not content:
                return

            tree = ast.parse(content, filename=file_path)

            class CallRelationshipVisitor(ast.NodeVisitor):
                def __init__(self, builder):
                    self.builder = builder
                    self.current_class = None
                    self.current_function = None

                def visit_ClassDef(self, node: ast.ClassDef):
                    old_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = old_class

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self._visit_function(node)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    self._visit_function(node)

                def _visit_function(self, node):
                    old_function = self.current_function
                    self.current_function = node.name

                    # Find the current node
                    if self.current_class:
                        current_full_name = (
                            f"{file_path}::{self.current_class}::{node.name}"
                        )
                    else:
                        current_full_name = f"{file_path}::{node.name}"

                    if current_full_name in self.builder.nodes:
                        current_node = self.builder.nodes[current_full_name]

                        # Extract function calls within this function
                        calls = self._extract_function_calls(node)
                        current_node.calls.update(calls)

                        # Update reverse relationships
                        for called_name in calls:
                            self._add_caller_relationship(
                                called_name, current_full_name
                            )

                    self.generic_visit(node)
                    self.current_function = old_function

                def _extract_function_calls(self, node: ast.AST) -> Set[str]:
                    """Extract function calls made within a function."""
                    calls = set()

                    class CallVisitor(ast.NodeVisitor):
                        def visit_Call(self, call_node: ast.Call):
                            call_name = self._get_call_name(call_node)
                            if call_name:
                                calls.add(call_name)
                            self.generic_visit(call_node)

                        def _get_call_name(self, call_node: ast.Call) -> Optional[str]:
                            """Extract the function name being called."""
                            if isinstance(call_node.func, ast.Name):
                                # Direct function call: func()
                                return call_node.func.id
                            elif isinstance(call_node.func, ast.Attribute):
                                # Method call: obj.method() or module.func()
                                if isinstance(call_node.func.value, ast.Name):
                                    # obj.method()
                                    return f"{call_node.func.value.id}.{call_node.func.attr}"
                                elif isinstance(call_node.func.value, ast.Attribute):
                                    # Nested: obj.attr.method()
                                    base = self._get_attribute_chain(
                                        call_node.func.value
                                    )
                                    if base:
                                        return f"{base}.{call_node.func.attr}"
                            return None

                        def _get_attribute_chain(
                            self, node: ast.Attribute
                        ) -> Optional[str]:
                            """Get the full attribute chain (e.g., 'obj.attr.subattr')."""
                            parts = [node.attr]
                            current = node.value

                            while isinstance(current, ast.Attribute):
                                parts.append(current.attr)
                                current = current.value

                            if isinstance(current, ast.Name):
                                parts.append(current.id)
                                return ".".join(reversed(parts))
                            return None

                    call_visitor = CallVisitor()
                    call_visitor.visit(node)
                    return calls

                def _add_caller_relationship(
                    self, called_name: str, caller_full_name: str
                ):
                    """Add reverse relationship: called function knows who calls it."""
                    # Try to resolve the called function to a full name
                    resolved_names = self._resolve_function_name(called_name)

                    for resolved_name in resolved_names:
                        if resolved_name in self.builder.nodes:
                            self.builder.nodes[resolved_name].called_by.add(
                                caller_full_name
                            )

                def _resolve_function_name(self, call_name: str) -> List[str]:
                    """Resolve a function call to possible full names."""
                    candidates = []

                    # Simple function call
                    if "." not in call_name:
                        # Look for functions with this name in current file
                        for full_name in self.builder.nodes:
                            if full_name.startswith(file_path) and full_name.endswith(
                                f"::{call_name}"
                            ):
                                candidates.append(full_name)

                        # Look for functions with this name in other files (less precise)
                        for full_name in self.builder.nodes:
                            if full_name.endswith(f"::{call_name}"):
                                candidates.append(full_name)

                    else:
                        # Method call: obj.method or module.func
                        # This is more complex and would require import analysis
                        # For now, just try simple matching
                        parts = call_name.split(".")
                        method_name = parts[-1]

                        for full_name in self.builder.nodes:
                            if full_name.endswith(f"::{method_name}"):
                                candidates.append(full_name)

                    return candidates

            visitor = CallRelationshipVisitor(self)
            visitor.visit(tree)

        except Exception as e:
            self.logger.warning(
                f"Error establishing call relationships for {file_path}: {e}"
            )

    async def _read_file(self, file_path: str) -> Optional[str]:
        """Read content of a file."""
        try:
            if self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
                result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM, "read_file", {"file_path": file_path}
                )

                if result and "result" in result and "content" in result["result"]:
                    return result["result"]["content"][0]["text"]

            # Fallback to direct file reading
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        except Exception as e:
            self.logger.warning(f"Error reading file {file_path}: {e}")
            return None


class CallGraphTestDiscoveryAnalyzer:
    """Analyzes test coverage using call graph backward traversal."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize the call graph test discovery analyzer.

        Args:
            project_root: Root directory of the project
            mcp_manager: MCP manager for file system operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.call_graph_test_discovery")
        self.call_graph_builder = CallGraphBuilder(project_root, mcp_manager)
        self.call_graph: Dict[str, CallGraphNode] = {}

    async def discover_test_coverage(
        self, library_usage: LibraryUsageSummary
    ) -> CallGraphTestDiscoveryResult:
        """Discover test coverage using call graph analysis.

        Args:
            library_usage: Summary of library usage in the project

        Returns:
            Test discovery results with call graph analysis
        """
        self.logger.info(
            f"Starting call graph analysis for {library_usage.total_usages} library usages"
        )

        # Build the complete call graph
        self.call_graph = await self.call_graph_builder.build_call_graph()

        # Find nodes that contain library usage
        library_usage_nodes = self._find_library_usage_nodes(library_usage)

        # Perform backward traversal from each library usage point
        coverage_paths = []
        for usage_location in library_usage.usage_locations:
            paths = await self._find_test_coverage_paths(usage_location)
            coverage_paths.extend(paths)

        # Identify uncovered usages
        covered_usages = [path.library_usage for path in coverage_paths]
        uncovered_usages = [
            usage
            for usage in library_usage.usage_locations
            if usage not in covered_usages
        ]

        # Calculate statistics
        test_nodes = [
            node
            for node in self.call_graph.values()
            if node.node_type == NodeType.TEST_FUNCTION
        ]
        coverage_percentage = (
            len(covered_usages) / len(library_usage.usage_locations) * 100
            if library_usage.usage_locations
            else 0
        )

        result = CallGraphTestDiscoveryResult(
            total_nodes=len(self.call_graph),
            test_nodes=len(test_nodes),
            library_usage_nodes=len(library_usage_nodes),
            coverage_paths=coverage_paths,
            uncovered_usages=uncovered_usages,
            coverage_percentage=coverage_percentage,
        )

        self.logger.info(
            f"Call graph analysis complete: {len(coverage_paths)} coverage paths found, "
            f"{coverage_percentage:.1f}% coverage"
        )

        return result

    def _find_library_usage_nodes(
        self, library_usage: LibraryUsageSummary
    ) -> List[CallGraphNode]:
        """Find call graph nodes that contain library usage points."""
        library_usage_nodes = []

        for usage_location in library_usage.usage_locations:
            # Find the node that contains this usage
            for node in self.call_graph.values():
                if (
                    node.file_path == usage_location.file_path
                    and node.function_name == usage_location.function_name
                    and node.class_name == usage_location.class_name
                ):
                    library_usage_nodes.append(node)
                    break

        return library_usage_nodes

    async def _find_test_coverage_paths(
        self, usage_location: LibraryUsageLocation
    ) -> List[CoveragePath]:
        """Find all paths from test functions to a library usage location.

        Args:
            usage_location: Library usage location to find coverage for

        Returns:
            List of coverage paths
        """
        # Find the node containing this usage
        usage_node = None
        for node in self.call_graph.values():
            if (
                node.file_path == usage_location.file_path
                and node.function_name == usage_location.function_name
                and node.class_name == usage_location.class_name
            ):
                usage_node = node
                break

        if not usage_node:
            self.logger.warning(
                f"Could not find call graph node for usage: {usage_location.usage_context}"
            )
            return []

        # Perform backward traversal to find test functions
        coverage_paths = []

        def traverse_backwards(current_node: CallGraphNode, path: List[CallGraphNode]):
            """Recursively traverse backwards to find test functions."""
            # Add current node to path
            current_path = [current_node] + path

            # Check for cycles (prevents infinite loops)
            if len(current_path) != len(set(node.full_name for node in current_path)):
                return

            # If we reached a test function, create a coverage path
            if current_node.node_type == NodeType.TEST_FUNCTION:
                coverage_paths.append(
                    CoveragePath(
                        test_node=current_node,
                        library_usage=usage_location,
                        call_chain=current_path.copy(),
                    )
                )
                return

            # Continue traversing to callers
            for caller_full_name in current_node.called_by:
                if caller_full_name in self.call_graph:
                    caller_node = self.call_graph[caller_full_name]
                    traverse_backwards(caller_node, current_path)

        # Start traversal from the usage node
        traverse_backwards(usage_node, [])

        return coverage_paths
