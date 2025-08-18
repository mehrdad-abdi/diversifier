"""Library usage analysis using Python AST parsing."""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from ..mcp_manager import MCPManager, MCPServerType


class LibraryUsageType(Enum):
    """Types of library usage patterns."""

    IMPORT = "import"
    FROM_IMPORT = "from_import"
    FUNCTION_CALL = "function_call"
    CLASS_INSTANTIATION = "class_instantiation"
    METHOD_CALL = "method_call"
    ATTRIBUTE_ACCESS = "attribute_access"


@dataclass
class LibraryUsageLocation:
    """Represents a specific location where a library is used."""

    file_path: str
    line_number: int
    column_offset: int
    usage_type: LibraryUsageType
    usage_context: str  # The actual code that uses the library
    function_name: Optional[str] = None  # Function/method containing this usage
    class_name: Optional[str] = None  # Class containing this usage
    module_path: Optional[str] = None  # Full module path for imports


@dataclass
class LibraryUsageSummary:
    """Summary of library usage across the project."""

    target_library: str
    total_usages: int
    usage_locations: List[LibraryUsageLocation] = field(default_factory=list)
    imported_modules: Set[str] = field(default_factory=set)
    used_functions: Set[str] = field(default_factory=set)
    used_classes: Set[str] = field(default_factory=set)
    affected_files: Set[str] = field(default_factory=set)


class LibraryUsageAnalyzer:
    """Analyzes library usage patterns in Python code using AST parsing."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize the library usage analyzer.

        Args:
            project_root: Root directory of the project
            mcp_manager: MCP manager for file system operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.library_usage_analyzer")

    async def analyze_library_usage(self, target_library: str) -> LibraryUsageSummary:
        """Analyze usage of a specific library across the project.

        Args:
            target_library: Name of the library to analyze (e.g., 'requests', 'httpx')

        Returns:
            Summary of library usage patterns
        """
        self.logger.info(f"Analyzing usage of library: {target_library}")

        # Get all Python files in the project
        python_files = await self._get_python_files()

        usage_locations = []
        imported_modules = set()
        used_functions = set()
        used_classes = set()
        affected_files = set()

        for file_path in python_files:
            try:
                # Read file content
                content = await self._read_file(file_path)
                if not content:
                    continue

                # Parse AST
                tree = ast.parse(content, filename=file_path)

                # Analyze library usage in this file
                file_usage = self._analyze_file_usage(tree, target_library, file_path)

                if file_usage:
                    usage_locations.extend(file_usage)
                    affected_files.add(file_path)

                    # Collect usage metadata
                    for usage in file_usage:
                        if usage.usage_type in [
                            LibraryUsageType.IMPORT,
                            LibraryUsageType.FROM_IMPORT,
                        ]:
                            if usage.module_path:
                                imported_modules.add(usage.module_path)
                        elif usage.usage_type == LibraryUsageType.FUNCTION_CALL:
                            if usage.function_name:
                                used_functions.add(usage.function_name)
                        elif usage.usage_type == LibraryUsageType.CLASS_INSTANTIATION:
                            if usage.class_name:
                                used_classes.add(usage.class_name)

            except Exception as e:
                self.logger.warning(f"Error analyzing file {file_path}: {e}")
                continue

        summary = LibraryUsageSummary(
            target_library=target_library,
            total_usages=len(usage_locations),
            usage_locations=usage_locations,
            imported_modules=imported_modules,
            used_functions=used_functions,
            used_classes=used_classes,
            affected_files=affected_files,
        )

        self.logger.info(
            f"Found {summary.total_usages} usages of {target_library} in {len(affected_files)} files"
        )
        return summary

    def _analyze_file_usage(
        self, tree: ast.AST, target_library: str, file_path: str
    ) -> List[LibraryUsageLocation]:
        """Analyze library usage in a single file's AST.

        Args:
            tree: AST tree of the file
            target_library: Target library name
            file_path: Path to the file being analyzed

        Returns:
            List of usage locations in this file
        """
        usage_locations = []

        class LibraryUsageVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_class = None
                self.current_function = None

            def visit_ClassDef(self, node: ast.ClassDef):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node: ast.FunctionDef):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function

            def visit_Import(self, node: ast.Import):
                for alias in node.names:
                    if alias.name == target_library or alias.name.startswith(
                        f"{target_library}."
                    ):
                        usage_locations.append(
                            LibraryUsageLocation(
                                file_path=file_path,
                                line_number=node.lineno,
                                column_offset=node.col_offset,
                                usage_type=LibraryUsageType.IMPORT,
                                usage_context=f"import {alias.name}",
                                function_name=self.current_function,
                                class_name=self.current_class,
                                module_path=alias.name,
                            )
                        )

            def visit_ImportFrom(self, node: ast.ImportFrom):
                if node.module and (
                    node.module == target_library
                    or node.module.startswith(f"{target_library}.")
                ):
                    for alias in node.names:
                        usage_locations.append(
                            LibraryUsageLocation(
                                file_path=file_path,
                                line_number=node.lineno,
                                column_offset=node.col_offset,
                                usage_type=LibraryUsageType.FROM_IMPORT,
                                usage_context=f"from {node.module} import {alias.name}",
                                function_name=self.current_function,
                                class_name=self.current_class,
                                module_path=node.module,
                            )
                        )

            def visit_Call(self, node: ast.Call):
                # Check if this is a function call that might be using the target library
                if isinstance(node.func, ast.Name):
                    # Direct function call (e.g., requests())
                    if self._is_library_usage(node.func.id):
                        usage_locations.append(
                            LibraryUsageLocation(
                                file_path=file_path,
                                line_number=node.lineno,
                                column_offset=node.col_offset,
                                usage_type=LibraryUsageType.FUNCTION_CALL,
                                usage_context=self._get_node_source(node),
                                function_name=self.current_function,
                                class_name=self.current_class,
                            )
                        )
                elif isinstance(node.func, ast.Attribute):
                    # Method call (e.g., requests.get())
                    if self._is_library_attribute_usage(node.func):
                        usage_locations.append(
                            LibraryUsageLocation(
                                file_path=file_path,
                                line_number=node.lineno,
                                column_offset=node.col_offset,
                                usage_type=LibraryUsageType.METHOD_CALL,
                                usage_context=self._get_node_source(node),
                                function_name=self.current_function,
                                class_name=self.current_class,
                            )
                        )

                self.generic_visit(node)

            def visit_Attribute(self, node: ast.Attribute):
                # Check attribute access (e.g., requests.exceptions)
                if self._is_library_attribute_usage(node):
                    usage_locations.append(
                        LibraryUsageLocation(
                            file_path=file_path,
                            line_number=node.lineno,
                            column_offset=node.col_offset,
                            usage_type=LibraryUsageType.ATTRIBUTE_ACCESS,
                            usage_context=self._get_node_source(node),
                            function_name=self.current_function,
                            class_name=self.current_class,
                        )
                    )

                self.generic_visit(node)

            def _is_library_usage(self, name: str) -> bool:
                """Check if a name refers to the target library."""
                return name == target_library

            def _is_library_attribute_usage(self, node: ast.Attribute) -> bool:
                """Check if an attribute access refers to the target library."""
                if isinstance(node.value, ast.Name):
                    return node.value.id == target_library
                elif isinstance(node.value, ast.Attribute):
                    # Handle nested attributes like library.module.function
                    return self._get_full_attribute_name(node).startswith(
                        target_library
                    )
                return False

            def _get_full_attribute_name(self, node: ast.Attribute) -> str:
                """Get the full attribute name (e.g., 'requests.exceptions.RequestException')."""
                parts = [node.attr]
                current = node.value

                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value

                if isinstance(current, ast.Name):
                    parts.append(current.id)

                return ".".join(reversed(parts))

            def _get_node_source(self, node: ast.AST) -> str:
                """Get source code representation of a node (simplified)."""
                try:
                    return ast.unparse(node)
                except AttributeError:
                    # ast.unparse not available in Python < 3.9
                    return f"<usage at line {getattr(node, 'lineno', 'unknown')}>"

        visitor = LibraryUsageVisitor()
        visitor.visit(tree)

        return usage_locations

    async def _get_python_files(self) -> List[str]:
        """Get all Python files in the project."""
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

            # Fallback to manual file discovery
            return [str(p) for p in self.project_root.rglob("*.py") if p.is_file()]

        except Exception as e:
            self.logger.warning(f"Error finding Python files: {e}")
            # Fallback to manual file discovery
            return [str(p) for p in self.project_root.rglob("*.py") if p.is_file()]

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

    def get_affected_functions(
        self, usage_summary: LibraryUsageSummary
    ) -> Dict[str, List[LibraryUsageLocation]]:
        """Group library usages by the functions that contain them.

        Args:
            usage_summary: Library usage summary

        Returns:
            Dictionary mapping function names to their usage locations
        """
        functions: Dict[str, List[LibraryUsageLocation]] = {}

        for usage in usage_summary.usage_locations:
            if usage.function_name:
                key = f"{usage.file_path}::{usage.function_name}"
                if usage.class_name:
                    key = (
                        f"{usage.file_path}::{usage.class_name}::{usage.function_name}"
                    )

                if key not in functions:
                    functions[key] = []
                functions[key].append(usage)

        return functions

    def get_usage_statistics(
        self, usage_summary: LibraryUsageSummary
    ) -> Dict[str, Any]:
        """Get detailed statistics about library usage.

        Args:
            usage_summary: Library usage summary

        Returns:
            Dictionary with usage statistics
        """
        usage_by_type: Dict[str, int] = {}
        usage_by_file: Dict[str, int] = {}

        for usage in usage_summary.usage_locations:
            # Count by usage type
            usage_type = usage.usage_type.value
            usage_by_type[usage_type] = usage_by_type.get(usage_type, 0) + 1

            # Count by file
            usage_by_file[usage.file_path] = usage_by_file.get(usage.file_path, 0) + 1

        return {
            "total_usages": usage_summary.total_usages,
            "affected_files": len(usage_summary.affected_files),
            "imported_modules": len(usage_summary.imported_modules),
            "used_functions": len(usage_summary.used_functions),
            "used_classes": len(usage_summary.used_classes),
            "usage_by_type": usage_by_type,
            "usage_by_file": usage_by_file,
            "most_used_files": sorted(
                usage_by_file.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }
