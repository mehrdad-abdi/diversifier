#!/usr/bin/env python3
"""LLM-based test runner that intelligently analyzes projects and runs tests."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain.chat_models import init_chat_model
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..config import LLMConfig, MigrationConfig
from ..mcp_manager import MCPManager, MCPServerType


class UnrecoverableTestRunnerError(Exception):
    """Exception raised when the test runner encounters an unrecoverable error."""

    def __init__(
        self, message: str, error_type: str, original_error: Optional[Exception] = None
    ):
        """Initialize unrecoverable test runner error.

        Args:
            message: Human-readable error description
            error_type: Type of error (validation, network, llm_service, unexpected)
            original_error: The original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.original_error = original_error


class TestEnvironmentResult(BaseModel):
    """Test environment setup and execution result model."""

    testing_framework: str = Field(
        description="The main testing framework (e.g., 'pytest', 'unittest')"
    )
    dev_dependencies: List[str] = Field(
        description="List of development packages needed"
    )
    install_commands: List[str] = Field(
        description="List of shell commands to install dependencies"
    )
    test_commands: List[str] = Field(description="List of shell commands to run tests")
    setup_commands: List[str] = Field(
        description="List of any setup commands needed before testing"
    )
    setup_successful: bool = Field(
        description="Whether the environment setup completed successfully"
    )
    tests_executed: int = Field(description="Number of tests that were executed")
    tests_passed: int = Field(description="Number of tests that passed")
    tests_failed: int = Field(description="Number of tests that failed")
    test_output: str = Field(description="Output from test execution")
    test_stderr: str = Field(description="Error output from test execution", default="")
    analysis: str = Field(
        description="Brief explanation of setup process and test results"
    )


class LLMTestRunner:
    """LLM-powered test runner that uses MCP servers for project analysis and test execution."""

    def __init__(
        self,
        project_path: str,
        llm_config: LLMConfig,
        migration_config: MigrationConfig,
        mcp_manager: MCPManager,
    ):
        """Initialize the LLM test runner.

        Args:
            project_path: Path to the target project
            llm_config: LLM configuration for model initialization
            migration_config: Migration configuration for test analysis
            mcp_manager: MCP manager for file system and command operations
        """
        self.project_path = Path(project_path).resolve()
        self.llm_config = llm_config
        self.migration_config = migration_config
        self.mcp_manager = mcp_manager

        # Initialize LLM with provided configuration
        model_id = f"{llm_config.provider.lower()}:{llm_config.model_name}"
        self.llm = init_chat_model(  # type: ignore[call-overload]
            model=model_id,
            temperature=0,  # Use low temperature for consistent test analysis
            **llm_config.additional_params,
        )

    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze project structure to understand testing setup and dependencies."""
        # Get command connection from manager
        command_connection = self.mcp_manager.get_connection(MCPServerType.COMMAND)
        if not command_connection:
            raise ValueError("Command MCP connection not available from manager")
        command_client = command_connection.client

        # Find common project files from configuration using find_files
        project_files = []
        common_files = self.migration_config.common_project_files

        for filename in common_files:
            # Use find_files to check if specific file exists
            result = command_client.call_tool(
                "find_files",
                {
                    "pattern": filename,
                    "directory": ".",
                },
            )
            if (
                result
                and "result" in result
                and "content" in result["result"]
                and len(result["result"]["content"]) > 0
            ):
                find_result = json.loads(result["result"]["content"][0]["text"])
                if find_result.get("total_matches", 0) > 0:
                    # File exists - add it as a file type
                    project_files.append(
                        {
                            "name": filename,
                            "type": "file",
                        }
                    )

        # Find test directories using configured test paths
        test_dirs = []
        for test_path_config in self.migration_config.test_paths:
            test_path = test_path_config.rstrip("/")

            # Use find_files to check if test directory exists by looking for any files in it
            result = command_client.call_tool(
                "find_files",
                {
                    "pattern": "*",
                    "directory": test_path,
                },
            )
            if (
                result
                and "result" in result
                and "content" in result["result"]
                and len(result["result"]["content"]) > 0
            ):
                content_text = result["result"]["content"][0]["text"]
                # If we can search the directory and it doesn't contain error, it exists
                if "Error" not in content_text:
                    test_dirs.append(test_path)

        return {
            "project_path": str(self.project_path),
            "project_files": project_files,
            "test_directories": test_dirs,
        }

    async def setup_and_run_tests(
        self,
        project_structure: Dict[str, Any],
        test_functions: List[str],
    ) -> Dict[str, Any]:
        """Use ReAct agent to analyze project, setup development environment, and run tests.

        This method uses a ReAct agent with tool execution capabilities to:
        1. Analyze the project files and detect requirements
        2. Execute setup and installation commands to prepare the environment
        3. Run the tests and return results

        The agent has access to tools to read files, find files, and execute commands,
        allowing it to validate that each step works before proceeding.

        Args:
            project_structure: Project structure analysis from analyze_project_structure
            test_functions: List of specific test functions to target (required)

        Returns:
            Dict containing test execution results in coordinator-expected format
        """

        # Create tool instances using the @tool decorator
        @tool
        def read_file_tool(file_path: str) -> str:
            """Read the contents of a file to understand project structure and dependencies.

            Args:
                file_path: Path to the file to read (relative to project root)

            Returns:
                The file contents as a string
            """
            try:
                filesystem_connection = self.mcp_manager.get_connection(
                    MCPServerType.FILESYSTEM
                )
                if not filesystem_connection:
                    return "Error: Filesystem MCP connection not available"

                result = filesystem_connection.client.call_tool(
                    "read_file", {"file_path": file_path}
                )
                if (
                    result
                    and "result" in result
                    and "content" in result["result"]
                    and len(result["result"]["content"]) > 0
                ):
                    return result["result"]["content"][0]["text"]
                else:
                    return f"Error: Could not read file {file_path}"
            except Exception as e:
                return f"Error reading {file_path}: {str(e)}"

        @tool
        def find_files_tool(pattern: str, directory: str = ".") -> str:
            """Find files matching a pattern in the project.

            Args:
                pattern: File pattern to search for (e.g., "*.py", "test_*.py", "requirements*.txt")
                directory: Directory to search in (default: ".")

            Returns:
                JSON string with matching files information
            """
            try:
                command_connection = self.mcp_manager.get_connection(
                    MCPServerType.COMMAND
                )
                if not command_connection:
                    return '{"error": "Command MCP connection not available"}'

                result = command_connection.client.call_tool(
                    "find_files", {"pattern": pattern, "directory": directory}
                )
                if (
                    result
                    and "result" in result
                    and "content" in result["result"]
                    and len(result["result"]["content"]) > 0
                ):
                    return result["result"]["content"][0]["text"]
                else:
                    return (
                        f'{{"pattern": "{pattern}", "matches": [], "total_matches": 0}}'
                    )
            except Exception as e:
                return f'{{"error": "Could not find files: {str(e)}"}}'

        @tool
        def execute_command_tool(command: str, timeout: int = 300) -> str:
            """Execute a shell command to setup environment and run tests.

            Args:
                command: The shell command to execute
                timeout: Command timeout in seconds (default: 300)

            Returns:
                JSON string with command execution results including stdout, stderr, and exit code
            """
            try:
                command_connection = self.mcp_manager.get_connection(
                    MCPServerType.COMMAND
                )
                if not command_connection:
                    return '{"error": "Command MCP connection not available"}'

                result = command_connection.client.call_tool(
                    "execute_command", {"command": command, "timeout": timeout}
                )
                if (
                    result
                    and "result" in result
                    and "content" in result["result"]
                    and len(result["result"]["content"]) > 0
                ):
                    return result["result"]["content"][0]["text"]
                else:
                    return f'{{"command": "{command}", "error": "No result returned", "success": false}}'
            except Exception as e:
                return f'{{"command": "{command}", "error": "Command execution failed: {str(e)}", "success": false}}'

        # Create the tools list
        tools = [read_file_tool, find_files_tool, execute_command_tool]

        # Build the detailed prompt for the agent - filesystem connection is required
        filesystem_connection = self.mcp_manager.get_connection(
            MCPServerType.FILESYSTEM
        )
        if not filesystem_connection:
            raise ValueError(
                "Filesystem MCP connection not available - this is required for test execution"
            )

        file_contents_text = ""
        # Read initial project files for context
        for project_file in project_structure["project_files"]:
            if project_file["type"] == "file":
                filename = project_file["name"]
                try:
                    result = filesystem_connection.client.call_tool(
                        "read_file", {"file_path": filename}
                    )
                    if (
                        result
                        and "result" in result
                        and "content" in result["result"]
                        and len(result["result"]["content"]) > 0
                    ):
                        file_content = result["result"]["content"][0]["text"]
                        # Limit to first 50 lines for initial context
                        content_lines = file_content.split("\n")[:50]
                        file_contents_text += f"\n=== {filename} (first 50 lines) ===\n"
                        file_contents_text += "\n".join(content_lines) + "\n"
                except Exception:
                    continue

        # Build target test specification - test_functions is required
        test_target = f"""
SPECIFIC TEST TARGETS:
You must run ONLY these specific test functions:
{test_functions}

Generate precise test commands targeting only these functions, such as:
- pytest path/to/test_file.py::test_function_name
- pytest -k "test_function_name"  
- pytest path/to/test_file.py::TestClass::test_method
"""

        # Load prompt from file following project pattern
        prompt_dir = Path(__file__).parent.parent / "prompts"
        prompt_path = prompt_dir / "react_test_setup.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"ReAct prompt file not found: {prompt_path}")

        react_prompt_content = prompt_path.read_text().strip()
        react_prompt_template = PromptTemplate.from_template(react_prompt_content)

        # Create the ReAct agent
        agent = create_react_agent(self.llm, tools, react_prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True,
        )

        # Prepare the input for the agent
        agent_input = {
            "input": "Set up the development environment and run the tests for this Python project.",
            "project_files": [f["name"] for f in project_structure["project_files"]],
            "test_directories": project_structure["test_directories"],
            "file_contents": file_contents_text,
            "test_target": test_target,
        }

        try:
            # Execute the agent
            result = await agent_executor.ainvoke(agent_input)
            agent_output = result.get("output", "")

            # Use LLM to extract structured information from agent output
            return await self._extract_structured_results(agent_output, test_functions)

        except Exception as e:
            error_msg = f"ReAct agent execution failed: {str(e)}"
            raise UnrecoverableTestRunnerError(error_msg, "agent_execution", e)

    async def _extract_structured_results(
        self, agent_output: str, test_functions: List[str]
    ) -> Dict[str, Any]:
        """Use LLM to extract structured information from ReAct agent output.

        Args:
            agent_output: Natural language output from the ReAct agent
            test_functions: List of test functions that were targeted

        Returns:
            Dict containing structured test results in TestEnvironmentResult format
        """
        # Create a simple LLM for extraction (reuse the same model)
        extraction_llm = self.llm.with_structured_output(TestEnvironmentResult)

        extraction_prompt = f"""
Extract structured information from this ReAct agent execution log.

The agent was tasked with setting up a Python project environment and running these specific test functions:
{test_functions}

Here is the complete agent execution log:
---
{agent_output}
---

Analyze the log and extract:
1. What testing framework was detected/used
2. What dependencies were installed
3. What commands were executed for setup and testing
4. Whether the setup was successful
5. How many tests were executed, passed, and failed
6. The complete output from test execution
7. Any error output from test execution
8. A summary analysis of what happened

Be precise about the numbers - extract exact counts from the test output.
If no specific numbers are found, make reasonable inferences based on the number of target test functions ({len(test_functions)}).
"""

        result = await extraction_llm.ainvoke(
            [{"role": "user", "content": extraction_prompt}]
        )
        return result.model_dump()
