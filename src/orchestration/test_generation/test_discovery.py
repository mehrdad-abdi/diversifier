"""Test discovery for finding existing tests that cover library usage points."""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..mcp_manager import MCPManager, MCPServerType
from .library_usage_analyzer import LibraryUsageSummary, LibraryUsageLocation


@dataclass
class TestFunction:
    """Represents a discovered test function."""

    file_path: str
    function_name: str
    class_name: Optional[str]
    line_number: int
    source_code: str
    imports: List[str] = field(default_factory=list)
    calls_functions: List[str] = field(default_factory=list)
    covers_files: List[str] = field(default_factory=list)


@dataclass
class TestCoverage:
    """Represents test coverage for library usage points."""

    usage_location: LibraryUsageLocation
    covering_tests: List[TestFunction] = field(default_factory=list)
    confidence_score: float = (
        0.0  # 0.0 to 1.0, how confident we are the test covers this usage
    )


@dataclass
class TestDiscoveryResult:
    """Results of test discovery analysis."""

    total_tests_found: int
    relevant_tests: List[TestFunction] = field(default_factory=list)
    usage_coverage: List[TestCoverage] = field(default_factory=list)
    uncovered_usages: List[LibraryUsageLocation] = field(default_factory=list)
    coverage_percentage: float = 0.0


class TestDiscoveryAnalyzer:
    """Analyzes existing tests to find coverage of library usage points."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize the test discovery analyzer.

        Args:
            project_root: Root directory of the project
            mcp_manager: MCP manager for file system operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.test_discovery")

    async def discover_test_coverage(
        self, library_usage: LibraryUsageSummary, target_library: str
    ) -> TestDiscoveryResult:
        """Discover existing tests that cover library usage points.

        Args:
            library_usage: Summary of library usage in the project
            target_library: Name of the target library

        Returns:
            Test discovery results with coverage analysis
        """
        self.logger.info(
            f"Discovering test coverage for {library_usage.total_usages} library usages"
        )

        # Find all test files
        test_files = await self._find_test_files()

        # Parse test functions from all test files
        all_test_functions = []
        for test_file in test_files:
            test_functions = await self._parse_test_functions(test_file, target_library)
            all_test_functions.extend(test_functions)

        self.logger.info(
            f"Found {len(all_test_functions)} test functions in {len(test_files)} test files"
        )

        # Analyze which tests cover which library usage points
        coverage_analysis = await self._analyze_test_coverage(
            library_usage, all_test_functions, target_library
        )

        # Identify uncovered usages
        covered_usages = [cov.usage_location for cov in coverage_analysis]
        uncovered_usages = [
            usage
            for usage in library_usage.usage_locations
            if usage not in covered_usages
        ]

        # Calculate coverage percentage
        coverage_percentage = (
            len(coverage_analysis) / len(library_usage.usage_locations) * 100
            if library_usage.usage_locations
            else 0
        )

        # Filter relevant tests (tests that actually cover some library usage)
        relevant_tests = []
        for coverage in coverage_analysis:
            for test in coverage.covering_tests:
                if test not in relevant_tests:
                    relevant_tests.append(test)

        result = TestDiscoveryResult(
            total_tests_found=len(all_test_functions),
            relevant_tests=relevant_tests,
            usage_coverage=coverage_analysis,
            uncovered_usages=uncovered_usages,
            coverage_percentage=coverage_percentage,
        )

        self.logger.info(
            f"Test coverage analysis: {len(coverage_analysis)}/{len(library_usage.usage_locations)} "
            f"usages covered ({coverage_percentage:.1f}%)"
        )

        return result

    async def _find_test_files(self) -> List[str]:
        """Find all test files in the project."""
        test_patterns = [
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/*.py",
            "**/test/*.py",
        ]

        test_files = set()

        for pattern in test_patterns:
            try:
                if self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
                    result = await self.mcp_manager.call_tool(
                        MCPServerType.FILESYSTEM,
                        "find_files",
                        {"pattern": pattern, "directory": str(self.project_root)},
                    )

                    if result and "files" in result:
                        test_files.update(result["files"])
                else:
                    # Fallback to manual discovery
                    test_files.update(
                        str(p)
                        for p in self.project_root.rglob(pattern.replace("**/", ""))
                        if p.is_file()
                    )

            except Exception as e:
                self.logger.warning(
                    f"Error finding test files with pattern {pattern}: {e}"
                )

        return list(test_files)

    async def _parse_test_functions(
        self, test_file: str, target_library: str
    ) -> List[TestFunction]:
        """Parse test functions from a test file.

        Args:
            test_file: Path to the test file
            target_library: Target library name to avoid tests that directly import it

        Returns:
            List of test functions found in the file
        """
        try:
            content = await self._read_file(test_file)
            if not content:
                return []

            tree = ast.parse(content, filename=test_file)

            # Extract imports and test functions
            imports = self._extract_imports(tree)

            # Skip files that directly import the target library (not interesting for us)
            if self._imports_target_library(imports, target_library):
                self.logger.debug(
                    f"Skipping {test_file} - directly imports {target_library}"
                )
                return []

            test_functions = []

            class TestFunctionVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_class = None

                def visit_ClassDef(self, node: ast.ClassDef):
                    old_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = old_class

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    if self._is_test_function(node.name):
                        test_function = TestFunction(
                            file_path=test_file,
                            function_name=node.name,
                            class_name=self.current_class,
                            line_number=node.lineno,
                            source_code=self._get_function_source(node, content or ""),
                            imports=imports,
                            calls_functions=self._extract_function_calls(node),
                        )
                        test_functions.append(test_function)

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                    if self._is_test_function(node.name):
                        test_function = TestFunction(
                            file_path=test_file,
                            function_name=node.name,
                            class_name=self.current_class,
                            line_number=node.lineno,
                            source_code=self._get_function_source(node, content or ""),
                            imports=imports,
                            calls_functions=self._extract_function_calls(node),
                        )
                        test_functions.append(test_function)

                def _is_test_function(self, name: str) -> bool:
                    """Check if a function name indicates it's a test function."""
                    return (
                        name.startswith("test_")
                        or name.endswith("_test")
                        or name.startswith("Test")
                    )

                def _get_function_source(self, node: ast.AST, file_content: str) -> str:
                    """Extract source code of a function."""
                    try:
                        # Try to get source using ast.unparse if available
                        return ast.unparse(node)
                    except AttributeError:
                        # Fallback: extract lines from file content
                        lines = file_content.split("\n")
                        start_line = (
                            getattr(node, "lineno", 1) - 1
                        )  # Convert to 0-based indexing

                        # Find the end of the function by looking for the next function/class at the same indentation level
                        end_line = len(lines)
                        base_indent = len(lines[start_line]) - len(
                            lines[start_line].lstrip()
                        )

                        for i in range(start_line + 1, len(lines)):
                            line = lines[i]
                            if (
                                line.strip()
                                and (len(line) - len(line.lstrip())) <= base_indent
                            ):
                                # Found a line at the same or lesser indentation
                                if not line.lstrip().startswith(('"""', "'''", "#")):
                                    end_line = i
                                    break

                        return "\n".join(lines[start_line:end_line])

                def _extract_function_calls(self, node: ast.AST) -> List[str]:
                    """Extract function calls made within a test function."""
                    calls = []

                    class CallVisitor(ast.NodeVisitor):
                        def visit_Call(self, call_node: ast.Call):
                            if isinstance(call_node.func, ast.Name):
                                calls.append(call_node.func.id)
                            elif isinstance(call_node.func, ast.Attribute):
                                # Handle method calls like obj.method()
                                calls.append(call_node.func.attr)
                            self.generic_visit(call_node)

                    call_visitor = CallVisitor()
                    call_visitor.visit(node)

                    return calls

            visitor = TestFunctionVisitor()
            visitor.visit(tree)

            return test_functions

        except Exception as e:
            self.logger.warning(f"Error parsing test file {test_file}: {e}")
            return []

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from an AST."""
        imports = []

        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            def visit_ImportFrom(self, node: ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")

        visitor = ImportVisitor()
        visitor.visit(tree)

        return imports

    def _imports_target_library(self, imports: List[str], target_library: str) -> bool:
        """Check if imports include the target library."""
        return any(
            imp == target_library or imp.startswith(f"{target_library}.")
            for imp in imports
        )

    async def _analyze_test_coverage(
        self,
        library_usage: LibraryUsageSummary,
        test_functions: List[TestFunction],
        target_library: str,
    ) -> List[TestCoverage]:
        """Analyze which test functions cover which library usage points.

        Args:
            library_usage: Summary of library usage points
            test_functions: List of discovered test functions
            target_library: Name of the target library

        Returns:
            List of test coverage mappings
        """
        coverage_analysis = []

        for usage in library_usage.usage_locations:
            covering_tests = []

            # Find tests that might cover this usage location
            for test_func in test_functions:
                confidence = self._calculate_coverage_confidence(
                    usage, test_func, target_library
                )

                if (
                    confidence > 0.3
                ):  # Threshold for considering a test as covering the usage
                    covering_tests.append(test_func)

            if covering_tests:
                # Sort by confidence (we'll implement a simple heuristic here)
                covering_tests.sort(
                    key=lambda t: self._calculate_coverage_confidence(
                        usage, t, target_library
                    ),
                    reverse=True,
                )

                coverage = TestCoverage(
                    usage_location=usage,
                    covering_tests=covering_tests,
                    confidence_score=self._calculate_coverage_confidence(
                        usage, covering_tests[0], target_library
                    ),
                )
                coverage_analysis.append(coverage)

        return coverage_analysis

    def _calculate_coverage_confidence(
        self, usage: LibraryUsageLocation, test_func: TestFunction, target_library: str
    ) -> float:
        """Calculate confidence that a test function covers a library usage point.

        Args:
            usage: Library usage location
            test_func: Test function
            target_library: Target library name

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0

        # 1. Check if test is in the same file as the usage
        if (
            test_func.file_path.replace("/test_", "/").replace("_test.py", ".py")
            == usage.file_path
        ):
            confidence += 0.4
        elif (
            Path(test_func.file_path).stem.replace("test_", "").replace("_test", "")
            in usage.file_path
        ):
            confidence += 0.3

        # 2. Check if test calls the function containing the usage
        if usage.function_name and usage.function_name in test_func.calls_functions:
            confidence += 0.5

        # 3. Check if test mentions similar function/class names
        usage_context_lower = usage.usage_context.lower()
        test_source_lower = test_func.source_code.lower()

        if usage.function_name and usage.function_name.lower() in test_source_lower:
            confidence += 0.3

        if usage.class_name and usage.class_name.lower() in test_source_lower:
            confidence += 0.2

        # 4. Check for similar patterns or keywords
        # Extract meaningful words from usage context
        usage_words = set(re.findall(r"\w+", usage_context_lower))
        test_words = set(re.findall(r"\w+", test_source_lower))

        # Remove common words and the target library name
        common_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        usage_words = usage_words - common_words - {target_library.lower()}

        if usage_words and test_words:
            word_overlap = len(usage_words.intersection(test_words)) / len(usage_words)
            confidence += word_overlap * 0.2

        # 5. Bonus for test file naming conventions
        if "_test" in test_func.file_path or "test_" in test_func.file_path:
            confidence += 0.1

        return min(confidence, 1.0)  # Cap at 1.0

    async def _read_file(self, file_path: str) -> Optional[str]:
        """Read content of a file."""
        try:
            if self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
                result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM, "read_file", {"file_path": file_path}
                )

                if result and "result" in result:
                    return result["result"][0]["text"]

            # Fallback to direct file reading
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        except Exception as e:
            self.logger.warning(f"Error reading file {file_path}: {e}")
            return None
