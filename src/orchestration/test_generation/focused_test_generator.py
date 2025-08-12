"""Focused test generation for library usage points using LLM analysis."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..agent import DiversificationAgent, AgentType
from ..mcp_manager import MCPManager, MCPServerType
from .library_usage_analyzer import LibraryUsageSummary, LibraryUsageLocation
from .test_discovery import TestDiscoveryResult
from ..config import LLMConfig


@dataclass
class GeneratedTest:
    """Represents a generated test for a library usage point."""

    test_name: str
    test_code: str
    usage_location: LibraryUsageLocation
    description: str
    dependencies: List[str]
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None


@dataclass
class TestGenerationResult:
    """Results of focused test generation."""

    generated_tests: List[GeneratedTest]
    target_library: str
    total_usage_points: int
    tests_generated: int
    generation_success_rate: float


class FocusedTestGenerator:
    """Generates focused unit tests for library usage points using LLM analysis."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize the focused test generator.

        Args:
            project_root: Root directory of the project
            mcp_manager: MCP manager for operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.focused_test_generator")

    async def generate_focused_tests(
        self,
        library_usage: LibraryUsageSummary,
        test_discovery: TestDiscoveryResult,
        target_library: str,
        llm_config: LLMConfig,
    ) -> TestGenerationResult:
        """Generate focused unit tests for library usage points.

        Args:
            library_usage: Summary of library usage in the project
            test_discovery: Results of test discovery analysis
            target_library: Name of the target library
            llm_config: LLM configuration for test generation

        Returns:
            Test generation results
        """
        self.logger.info(
            f"Generating focused tests for {len(test_discovery.uncovered_usages)} uncovered usages"
        )

        # Create test generator agent
        generator_agent = DiversificationAgent(
            agent_type=AgentType.MIGRATOR,  # Use correct agent type
            llm_config=llm_config,
        )

        generated_tests = []

        # Generate tests for uncovered usage points
        for usage in test_discovery.uncovered_usages:
            try:
                test = await self._generate_test_for_usage(
                    usage, target_library, generator_agent, library_usage
                )
                if test:
                    generated_tests.append(test)
            except Exception as e:
                self.logger.warning(
                    f"Failed to generate test for usage at {usage.file_path}:{usage.line_number}: {e}"
                )

        # Also generate additional tests for low-confidence covered usages
        low_confidence_usages = [
            cov.usage_location
            for cov in test_discovery.usage_coverage
            if cov.confidence_score < 0.6
        ]

        for usage in low_confidence_usages[:5]:  # Limit to 5 additional tests
            try:
                test = await self._generate_test_for_usage(
                    usage, target_library, generator_agent, library_usage
                )
                if test:
                    generated_tests.append(test)
            except Exception as e:
                self.logger.warning(
                    f"Failed to generate additional test for usage at {usage.file_path}:{usage.line_number}: {e}"
                )

        success_rate = len(generated_tests) / (
            len(test_discovery.uncovered_usages) + len(low_confidence_usages)
        )

        result = TestGenerationResult(
            generated_tests=generated_tests,
            target_library=target_library,
            total_usage_points=len(test_discovery.uncovered_usages)
            + len(low_confidence_usages),
            tests_generated=len(generated_tests),
            generation_success_rate=success_rate,
        )

        self.logger.info(
            f"Generated {len(generated_tests)} focused tests (success rate: {success_rate:.1%})"
        )

        return result

    async def _generate_test_for_usage(
        self,
        usage: LibraryUsageLocation,
        target_library: str,
        agent: DiversificationAgent,
        library_usage: LibraryUsageSummary,
    ) -> Optional[GeneratedTest]:
        """Generate a focused test for a specific library usage point.

        Args:
            usage: Library usage location to generate test for
            target_library: Target library name
            agent: LLM agent for test generation
            library_usage: Full library usage summary for context

        Returns:
            Generated test or None if generation failed
        """
        # Read the source file to get context
        source_content = await self._read_file(usage.file_path)
        if not source_content:
            return None

        # Extract the function/method containing this usage
        function_context = self._extract_function_context(source_content, usage)

        # Generate test using LLM
        prompt = self._create_test_generation_prompt(
            usage, target_library, function_context, library_usage
        )

        try:
            result = agent.invoke(prompt)
            response_text = self._extract_response_text(result)

            # Parse the generated test
            test = self._parse_generated_test(response_text, usage)
            return test

        except Exception as e:
            self.logger.error(f"Error generating test with LLM: {e}")
            return None

    def _extract_function_context(
        self, source_content: str, usage: LibraryUsageLocation
    ) -> str:
        """Extract the function/method context around a library usage point.

        Args:
            source_content: Full source file content
            usage: Library usage location

        Returns:
            Function context string
        """
        lines = source_content.split("\n")

        # Find the line with the usage
        usage_line_idx = usage.line_number - 1  # Convert to 0-based indexing

        if usage_line_idx >= len(lines):
            return usage.usage_context

        # Find the start of the function/method
        function_start = usage_line_idx
        for i in range(usage_line_idx, -1, -1):
            line = lines[i]
            if re.match(r"^\s*(def|async def)\s+", line):
                function_start = i
                break

        # Find the end of the function/method
        function_end = len(lines)
        if function_start < len(lines):
            base_indent = len(lines[function_start]) - len(
                lines[function_start].lstrip()
            )

            for i in range(function_start + 1, len(lines)):
                line = lines[i]
                if line.strip():
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= base_indent and not line.lstrip().startswith(
                        ('"""', "'''", "#")
                    ):
                        function_end = i
                        break

        # Extract function context with some padding
        start_idx = max(0, function_start - 2)
        end_idx = min(len(lines), function_end + 2)

        return "\n".join(lines[start_idx:end_idx])

    def _create_test_generation_prompt(
        self,
        usage: LibraryUsageLocation,
        target_library: str,
        function_context: str,
        library_usage: LibraryUsageSummary,
    ) -> str:
        """Create a prompt for generating a focused test.

        Args:
            usage: Library usage location
            target_library: Target library name
            function_context: Context of the function containing the usage
            library_usage: Full library usage summary

        Returns:
            LLM prompt for test generation
        """
        return f"""You are tasked with generating a focused unit test for a specific library usage point in a Python project.

## Context
Target Library: {target_library}
File: {usage.file_path}
Line: {usage.line_number}
Usage Type: {usage.usage_type.value}
Usage Context: {usage.usage_context}
Function: {usage.function_name or 'module level'}
Class: {usage.class_name or 'none'}

## Function Context
```python
{function_context}
```

## Library Usage Summary
The project uses {target_library} in {len(library_usage.affected_files)} files with {library_usage.total_usages} total usages.
Common patterns: {', '.join(list(library_usage.used_functions)[:5])}

## Task
Generate a focused unit test that:
1. Tests the specific function/method containing the library usage WITHOUT importing {target_library} directly
2. Calls the function/method that contains the {target_library} usage
3. Validates the behavior and output of that function/method
4. Uses mocking if necessary to simulate external dependencies
5. Is independent and can run without complex setup (no databases, no external APIs, no Docker)
6. Focuses on the business logic that uses the library, not the library itself

## Requirements
- Test name should start with "test_"
- Use pytest conventions
- Include necessary imports
- Use mocking for external dependencies (HTTP calls, file I/O, etc.)
- Validate return values, side effects, or raised exceptions
- Be concise but thorough

## Response Format
Return a JSON response with the following structure:
```json
{{
    "test_name": "test_function_name",
    "test_code": "complete pytest test function code",
    "description": "brief description of what this test validates",
    "dependencies": ["list", "of", "required", "imports"],
    "setup_code": "optional setup code if needed",
    "teardown_code": "optional teardown code if needed"
}}
```

Generate the focused unit test now:"""

    def _extract_response_text(self, result: Dict[str, Any]) -> str:
        """Extract response text from agent result."""
        if "output" in result:
            return result["output"]
        elif "messages" in result:
            last_message = result["messages"][-1]
            return (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )
        else:
            return str(result)

    def _parse_generated_test(
        self, response_text: str, usage: LibraryUsageLocation
    ) -> Optional[GeneratedTest]:
        """Parse the LLM response into a GeneratedTest object.

        Args:
            response_text: Raw LLM response
            usage: Original usage location

        Returns:
            ParsedGeneratedTest object or None if parsing failed
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                test_data = json.loads(json_match.group(1))
            else:
                # Try to parse the entire response as JSON
                test_data = json.loads(response_text)

            return GeneratedTest(
                test_name=test_data.get(
                    "test_name", f"test_{usage.function_name or 'usage'}"
                ),
                test_code=test_data.get("test_code", ""),
                usage_location=usage,
                description=test_data.get(
                    "description", "Generated test for library usage"
                ),
                dependencies=test_data.get("dependencies", []),
                setup_code=test_data.get("setup_code"),
                teardown_code=test_data.get("teardown_code"),
            )

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to parse generated test response: {e}")

            # Fallback: try to extract code blocks
            code_blocks = re.findall(r"```python\n(.*?)\n```", response_text, re.DOTALL)
            if code_blocks:
                return GeneratedTest(
                    test_name=f"test_{usage.function_name or 'usage'}",
                    test_code=code_blocks[0],
                    usage_location=usage,
                    description="Generated test for library usage",
                    dependencies=["pytest"],
                )

            return None

    async def export_generated_tests(
        self, generation_result: TestGenerationResult, output_dir: Optional[str] = None
    ) -> str:
        """Export generated tests to test files.

        Args:
            generation_result: Test generation results
            output_dir: Output directory for test files

        Returns:
            Path to output directory
        """
        if output_dir is None:
            output_dir = str(self.project_root / "generated_tests")

        output_path = Path(output_dir)

        # Group tests by source file
        tests_by_file: Dict[str, List[GeneratedTest]] = {}
        for test in generation_result.generated_tests:
            source_file = Path(test.usage_location.file_path).stem
            test_file_name = f"test_{source_file}_library_usage.py"

            if test_file_name not in tests_by_file:
                tests_by_file[test_file_name] = []
            tests_by_file[test_file_name].append(test)

        # Generate test files
        for test_file_name, tests in tests_by_file.items():
            test_file_content = self._generate_test_file_content(
                tests, generation_result.target_library
            )
            test_file_path = output_path / test_file_name

            await self._write_file(str(test_file_path), test_file_content)

        # Export generation metadata
        metadata = {
            "target_library": generation_result.target_library,
            "total_usage_points": generation_result.total_usage_points,
            "tests_generated": generation_result.tests_generated,
            "generation_success_rate": generation_result.generation_success_rate,
            "test_files_created": len(tests_by_file),
            "generated_tests": [
                {
                    "test_name": test.test_name,
                    "description": test.description,
                    "source_file": test.usage_location.file_path,
                    "line_number": test.usage_location.line_number,
                    "usage_type": test.usage_location.usage_type.value,
                }
                for test in generation_result.generated_tests
            ],
        }

        metadata_path = output_path / "test_generation_metadata.json"
        await self._write_file(str(metadata_path), json.dumps(metadata, indent=2))

        self.logger.info(
            f"Exported {generation_result.tests_generated} tests to {output_dir}"
        )

        return str(output_path)

    def _generate_test_file_content(
        self, tests: List[GeneratedTest], target_library: str
    ) -> str:
        """Generate complete test file content.

        Args:
            tests: List of tests to include in the file
            target_library: Target library name

        Returns:
            Complete test file content
        """
        # Collect all dependencies
        all_dependencies = set()
        for test in tests:
            all_dependencies.update(test.dependencies)

        # Generate imports
        imports = ["import pytest"]
        for dep in sorted(all_dependencies):
            if dep != "pytest":
                imports.append(f"import {dep}")

        # Add common mocking imports
        if any(
            "mock" in test.test_code.lower() or "@patch" in test.test_code
            for test in tests
        ):
            imports.append("from unittest.mock import Mock, patch, MagicMock")

        imports_section = "\n".join(imports)

        # Generate setup/teardown if needed
        setup_sections = []
        for test in tests:
            if test.setup_code:
                setup_sections.append(test.setup_code)

        teardown_sections = []
        for test in tests:
            if test.teardown_code:
                teardown_sections.append(test.teardown_code)

        # Generate test methods
        test_methods = []
        for test in tests:
            test_method = f'''
def {test.test_name}():
    """
    {test.description}
    
    Tests library usage at: {test.usage_location.file_path}:{test.usage_location.line_number}
    Usage type: {test.usage_location.usage_type.value}
    """
{self._indent_code(test.test_code, 4)}
'''
            test_methods.append(test_method)

        return f'''"""
Focused unit tests for {target_library} library usage points.

Generated by Diversifier Focused Test Generator.
These tests validate functions that use {target_library} without importing it directly.
"""

{imports_section}


# Test Configuration
TARGET_LIBRARY = "{target_library}"

{''.join(test_methods)}
'''

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = " " * spaces
        lines = code.split("\n")

        # Skip empty lines at the start
        while lines and not lines[0].strip():
            lines.pop(0)

        # Remove common leading whitespace
        if lines:
            common_indent = len(lines[0]) - len(lines[0].lstrip())
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    line_indent = len(line) - len(line.lstrip())
                    common_indent = min(common_indent, line_indent)

            # Remove common indentation and add new indentation
            result_lines = []
            for line in lines:
                if line.strip():
                    result_lines.append(indent + line[common_indent:])
                else:
                    result_lines.append("")

            return "\n".join(result_lines)

        return code

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

    async def _write_file(self, file_path: str, content: str) -> None:
        """Write content to a file."""
        if self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
            try:
                await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "write_file",
                    {"file_path": file_path, "content": content},
                )
            except Exception as e:
                self.logger.warning(f"MCP write failed, using fallback: {e}")
                # Fallback to direct write
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w") as f:
                    f.write(content)
        else:
            # Direct file write
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
