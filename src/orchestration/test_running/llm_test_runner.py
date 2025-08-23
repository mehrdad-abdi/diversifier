#!/usr/bin/env python3
"""LLM-based test runner that intelligently analyzes projects and runs tests."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import LangChainException
from pydantic import BaseModel, Field, ValidationError

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


class DevRequirements(BaseModel):
    """Development requirements analysis model."""

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
    analysis: str = Field(description="Brief explanation of findings")


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
            max_tokens=llm_config.max_tokens,
            **llm_config.additional_params,
        )

    def _load_prompt(self, prompt_name: str) -> str:
        """Load prompt template from file."""
        prompt_dir = Path(__file__).parent.parent / "prompts"
        prompt_path = prompt_dir / f"{prompt_name}.txt"

        if prompt_path.exists():
            return prompt_path.read_text().strip()
        else:
            # Raise error for missing prompt files instead of fallback
            raise FileNotFoundError(
                f"Prompt file {prompt_name}.txt not found in {prompt_dir}"
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

    async def detect_dev_requirements(
        self,
        project_structure: Dict[str, Any],
        test_functions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Use LLM to analyze project files and detect development requirements."""
        # Get filesystem connection from manager for file reading
        filesystem_connection = self.mcp_manager.get_connection(
            MCPServerType.FILESYSTEM
        )
        if not filesystem_connection:
            raise ValueError("Filesystem MCP connection not available from manager")
        filesystem_client = filesystem_connection.client

        # Read project files collected by analyze_project_structure
        file_contents = {}
        for project_file in project_structure["project_files"]:
            if project_file["type"] == "file":
                filename = project_file["name"]
                try:
                    result = filesystem_client.call_tool(
                        "read_file", {"file_path": filename}
                    )
                    if (
                        result
                        and "result" in result
                        and "content" in result["result"]
                        and len(result["result"]["content"]) > 0
                    ):
                        # The text field contains the raw file content, not JSON
                        file_content = result["result"]["content"][0]["text"]
                        # Limit content to first 100 lines for analysis
                        content_lines = file_content.split("\n")[:100]
                        file_contents[filename] = "\n".join(content_lines)
                except Exception:
                    pass  # Skip files we can't read

        # Load system prompt from file
        system_prompt = self._load_prompt("test_dev_requirements")

        file_contents_text = "\n\n".join(
            [
                f"=== {filename} ===\n{content}"
                for filename, content in file_contents.items()
            ]
        )

        # Build the prompt based on whether specific test functions are requested
        if test_functions:
            test_scope_info = f"""
IMPORTANT: You need to generate test commands that run ONLY these specific test functions:
{test_functions}

Generate precise test commands that target only these functions, not the entire test suite.
For example, if using pytest, use patterns like:
- pytest path/to/test_file.py::test_function_name
- pytest -k "test_function_name"
- pytest path/to/test_file.py::TestClass::test_method
"""
        else:
            test_scope_info = """
Generate test commands to run the complete test suite for this project.
"""

        human_prompt = f"""Analyze this Python project to determine how to set up and run its tests:

Project structure:
- Project files found: {[f['name'] for f in project_structure['project_files']]}
- Test directories: {project_structure['test_directories']}

File contents:
{file_contents_text}

{test_scope_info}

Please provide your analysis in the structured format with precise test commands."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        # Create tools for the LLM to read additional files, find files, and execute commands
        def read_file_tool(file_path: str) -> str:
            """Read the contents of a file to understand project structure and dependencies.

            Args:
                file_path: Path to the file to read (relative to project root)

            Returns:
                The file contents as a string
            """
            try:
                result = filesystem_client.call_tool(
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
                command_client = command_connection.client
                result = command_client.call_tool(
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

        def execute_command_tool(command: str, timeout: int = 300) -> str:
            """Execute a shell command to test setup steps and see results.

            Args:
                command: The shell command to execute
                timeout: Command timeout in seconds (default: 300)

            Returns:
                JSON string with command execution results
            """
            try:
                command_connection = self.mcp_manager.get_connection(
                    MCPServerType.COMMAND
                )
                if not command_connection:
                    return '{"error": "Command MCP connection not available"}'
                command_client = command_connection.client
                result = command_client.call_tool(
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

        # Bind tools to the LLM
        llm_with_tools = self.llm.bind_tools(
            [read_file_tool, find_files_tool, execute_command_tool]
        )

        # Use structured output with Pydantic model - no retry logic
        structured_llm = llm_with_tools.with_structured_output(DevRequirements)

        try:
            response = await structured_llm.ainvoke(messages)

            # Convert Pydantic model to dict
            return response.model_dump()

        except ValidationError as e:
            # Pydantic validation errors indicate malformed LLM response
            error_msg = (
                f"LLM failed to produce valid structured output. "
                f"The LLM response does not match the expected DevRequirements schema: {str(e)}"
            )
            raise UnrecoverableTestRunnerError(error_msg, "validation", e)

        except (ConnectionError, TimeoutError) as e:
            # Network/connectivity issues
            error_msg = (
                f"Network connectivity issues prevented LLM analysis: {str(e)}. "
                f"Please check your internet connection and API configuration."
            )
            raise UnrecoverableTestRunnerError(error_msg, "network", e)

        except LangChainException as e:
            # LangChain-specific errors (API key issues, model issues, etc.)
            error_msg = (
                f"LLM service error: {str(e)}. "
                f"This may be due to API key issues, model availability, or service limits."
            )
            raise UnrecoverableTestRunnerError(error_msg, "llm_service", e)

        except Exception as e:
            # Unexpected errors that we cannot recover from
            error_msg = (
                f"Unexpected error during development requirements analysis: {str(e)}. "
                f"Error type: {type(e).__name__}"
            )
            raise UnrecoverableTestRunnerError(error_msg, "unexpected", e)

    async def setup_test_environment(
        self, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up the test environment by installing dependencies."""
        # Get command connection from manager
        command_connection = self.mcp_manager.get_connection(MCPServerType.COMMAND)
        if not command_connection:
            raise ValueError("Command MCP connection not available from manager")
        command_client = command_connection.client

        setup_results = {
            "setup_commands_executed": [],
            "install_commands_executed": [],
            "success": True,
            "errors": [],
        }

        # Execute setup commands
        for command in requirements.get("setup_commands", []):
            try:
                result = command_client.call_tool(
                    "execute_command", {"command": command, "timeout": 300}
                )
                if (
                    result
                    and "result" in result
                    and "content" in result["result"]
                    and len(result["result"]["content"]) > 0
                ):
                    command_result = json.loads(result["result"]["content"][0]["text"])
                else:
                    command_result = {"success": False, "exit_code": -1}
                setup_results["setup_commands_executed"].append(  # type: ignore[attr-defined]
                    {
                        "command": command,
                        "success": command_result["success"],
                        "exit_code": command_result["exit_code"],
                    }
                )

                if not command_result["success"]:
                    setup_results["success"] = False
                    setup_results["errors"].append(f"Setup command failed: {command}")  # type: ignore[attr-defined]

            except Exception as e:
                setup_results["success"] = False
                setup_results["errors"].append(  # type: ignore[attr-defined]
                    f"Error executing setup command '{command}': {str(e)}"
                )

        # Execute install commands
        for command in requirements.get("install_commands", []):
            try:
                result = command_client.call_tool(
                    "execute_command",
                    {"command": command, "timeout": 600},  # Longer timeout for installs
                )
                if (
                    result
                    and "result" in result
                    and "content" in result["result"]
                    and len(result["result"]["content"]) > 0
                ):
                    command_result = json.loads(result["result"]["content"][0]["text"])
                else:
                    command_result = {"success": False, "exit_code": -1}
                setup_results["install_commands_executed"].append(  # type: ignore[attr-defined]
                    {
                        "command": command,
                        "success": command_result["success"],
                        "exit_code": command_result["exit_code"],
                    }
                )

                if not command_result["success"]:
                    setup_results["success"] = False
                    setup_results["errors"].append(f"Install command failed: {command}")  # type: ignore[attr-defined]

            except Exception as e:
                setup_results["success"] = False
                setup_results["errors"].append(  # type: ignore[attr-defined]
                    f"Error executing install command '{command}': {str(e)}"
                )

        return setup_results

    async def setup_test_environment_iteratively(
        self,
        project_structure: Dict[str, Any],
        test_functions: Optional[List[str]] = None,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Set up test environment iteratively with LLM troubleshooting and error correction.

        This method uses an LLM with command execution capabilities to:
        1. Analyze the project and detect requirements
        2. Try to set up the environment
        3. If errors occur, diagnose and fix them iteratively
        4. Validate that tests can run successfully

        Args:
            project_structure: Project structure analysis
            test_functions: Optional list of specific test functions to target
            max_iterations: Maximum number of fix iterations (default: 3)

        Returns:
            Dict containing setup results and final working requirements
        """
        # Get connections for the iterative setup process
        filesystem_connection = self.mcp_manager.get_connection(
            MCPServerType.FILESYSTEM
        )
        command_connection = self.mcp_manager.get_connection(MCPServerType.COMMAND)

        if not filesystem_connection or not command_connection:
            raise ValueError("Required MCP connections not available")

        filesystem_client = filesystem_connection.client
        command_client = command_connection.client

        # Initial analysis and requirements detection with tools
        initial_requirements = await self.detect_dev_requirements(
            project_structure, test_functions
        )

        setup_log = []
        current_requirements = initial_requirements.copy()

        for iteration in range(max_iterations):
            setup_log.append(f"=== Iteration {iteration + 1} ===")

            # Try to set up environment with current requirements
            setup_results = await self.setup_test_environment(current_requirements)

            if setup_results["success"]:
                # Try to run a quick test to validate setup
                test_results = await self.run_tests(current_requirements)

                if test_results["overall_success"]:
                    setup_log.append(
                        f"✅ Setup successful on iteration {iteration + 1}"
                    )
                    return {
                        "success": True,
                        "requirements": current_requirements,
                        "setup_results": setup_results,
                        "test_results": test_results,
                        "iterations": iteration + 1,
                        "setup_log": setup_log,
                    }
                else:
                    # Setup worked but tests failed - need to troubleshoot test execution
                    setup_log.append(
                        f"⚠️  Setup completed but tests failed on iteration {iteration + 1}"
                    )
                    error_context = {
                        "setup_success": True,
                        "test_failure": True,
                        "test_output": test_results.get("test_commands_executed", []),
                        "current_requirements": current_requirements,
                    }
            else:
                # Setup failed - need to troubleshoot setup issues
                setup_log.append(
                    f"❌ Setup failed on iteration {iteration + 1}: {setup_results.get('errors', [])}"
                )
                error_context = {
                    "setup_success": False,
                    "setup_errors": setup_results.get("errors", []),
                    "setup_commands": setup_results.get("setup_commands_executed", []),
                    "install_commands": setup_results.get(
                        "install_commands_executed", []
                    ),
                    "current_requirements": current_requirements,
                }

            # Use LLM to analyze errors and suggest fixes
            if iteration < max_iterations - 1:  # Don't try to fix on last iteration
                try:
                    fixed_requirements = await self._diagnose_and_fix_setup_issues(
                        project_structure, error_context, test_functions, setup_log
                    )
                    current_requirements = fixed_requirements
                    setup_log.append(
                        f"🔧 LLM suggested fixes for iteration {iteration + 2}"
                    )
                except Exception as e:
                    setup_log.append(f"❌ Failed to generate fixes: {str(e)}")
                    break

        # All iterations exhausted without success
        setup_log.append(
            f"❌ Failed to set up environment after {max_iterations} iterations"
        )
        return {
            "success": False,
            "requirements": current_requirements,
            "setup_results": setup_results if "setup_results" in locals() else {},
            "iterations": max_iterations,
            "setup_log": setup_log,
            "error": f"Could not set up test environment after {max_iterations} attempts",
        }

    async def _diagnose_and_fix_setup_issues(
        self,
        project_structure: Dict[str, Any],
        error_context: Dict[str, Any],
        test_functions: Optional[List[str]],
        setup_log: List[str],
    ) -> Dict[str, Any]:
        """Use LLM to diagnose setup issues and suggest fixes.

        Args:
            project_structure: Project structure information
            error_context: Context about what failed (setup or test execution)
            test_functions: Specific test functions if any
            setup_log: Log of previous attempts

        Returns:
            Updated requirements with fixes
        """
        # Build diagnostic prompt
        if error_context.get("setup_success", False):
            # Test execution failed after successful setup
            diagnostic_prompt = f"""
The project setup completed successfully, but test execution failed. 

SETUP LOG:
{chr(10).join(setup_log)}

FAILED TEST EXECUTION:
{error_context.get('test_output', [])}

CURRENT REQUIREMENTS:
{error_context.get('current_requirements', {})}

Please analyze the test execution failure and provide updated requirements with corrected test commands.
"""
        else:
            # Setup itself failed
            diagnostic_prompt = f"""
The project setup failed with the following issues:

SETUP LOG:
{chr(10).join(setup_log)}

SETUP ERRORS:
{error_context.get('setup_errors', [])}

FAILED COMMANDS:
Setup: {error_context.get('setup_commands', [])}
Install: {error_context.get('install_commands', [])}

CURRENT REQUIREMENTS:
{error_context.get('current_requirements', {})}

Please analyze these setup failures and provide updated requirements with corrected commands.
"""

        if test_functions:
            diagnostic_prompt += f"""

SPECIFIC TEST FUNCTIONS TO TARGET:
{test_functions}

Make sure the corrected test commands target only these specific functions.
"""

        diagnostic_prompt += """

Use the available tools to:
1. Investigate the specific errors by reading relevant files
2. Try alternative setup approaches by executing test commands  
3. Validate your proposed solutions work before recommending them

Provide updated requirements that should resolve the issues."""

        # Get filesystem connection for tools
        filesystem_connection = self.mcp_manager.get_connection(
            MCPServerType.FILESYSTEM
        )
        if not filesystem_connection:
            raise ValueError("Filesystem MCP connection not available")
        filesystem_client = filesystem_connection.client

        # Create the same tools as in detect_dev_requirements
        def read_file_tool(file_path: str) -> str:
            """Read the contents of a file to understand project structure and dependencies."""
            try:
                result = filesystem_client.call_tool(
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

        def find_files_tool(pattern: str, directory: str = ".") -> str:
            """Find files matching a pattern in the project."""
            try:
                command_connection = self.mcp_manager.get_connection(
                    MCPServerType.COMMAND
                )
                if not command_connection:
                    return '{"error": "Command MCP connection not available"}'
                command_client = command_connection.client
                result = command_client.call_tool(
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

        def execute_command_tool(command: str, timeout: int = 300) -> str:
            """Execute a shell command to test setup steps and see results."""
            try:
                command_connection = self.mcp_manager.get_connection(
                    MCPServerType.COMMAND
                )
                if not command_connection:
                    return '{"error": "Command MCP connection not available"}'
                command_client = command_connection.client
                result = command_client.call_tool(
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

        # Create LLM with tools for diagnosis and fixing
        llm_with_tools = self.llm.bind_tools(
            [read_file_tool, find_files_tool, execute_command_tool]
        )
        structured_llm = llm_with_tools.with_structured_output(DevRequirements)

        messages = [
            SystemMessage(
                content="You are an expert Python developer debugging test environment setup issues. Use the available tools to investigate problems and validate your solutions before recommending them."
            ),
            HumanMessage(content=diagnostic_prompt),
        ]

        try:
            response = await structured_llm.ainvoke(messages)
            return response.model_dump()
        except Exception as e:
            raise UnrecoverableTestRunnerError(
                f"Failed to diagnose and fix setup issues: {str(e)}",
                "diagnostic_failure",
                e,
            )

    async def run_tests(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tests using the detected testing framework and commands."""
        # Get command connection from manager
        command_connection = self.mcp_manager.get_connection(MCPServerType.COMMAND)
        if not command_connection:
            raise ValueError("Command MCP connection not available from manager")
        command_client = command_connection.client

        test_results = {
            "test_commands_executed": [],
            "overall_success": True,
            "summary": {
                "total_commands": 0,
                "successful_commands": 0,
                "failed_commands": 0,
            },
        }

        test_commands = requirements.get("test_commands", ["pytest"])

        for command in test_commands:
            try:
                result = command_client.call_tool(
                    "execute_command", {"command": command, "timeout": 600}
                )
                if (
                    result
                    and "result" in result
                    and "content" in result["result"]
                    and len(result["result"]["content"]) > 0
                ):
                    command_result = json.loads(result["result"]["content"][0]["text"])
                else:
                    command_result = {"success": False, "exit_code": -1}

                test_execution = {
                    "command": command,
                    "success": command_result["success"],
                    "exit_code": command_result["exit_code"],
                    "stdout": command_result.get("stdout", ""),
                    "stderr": command_result.get("stderr", ""),
                }

                test_results["test_commands_executed"].append(test_execution)  # type: ignore[attr-defined]
                test_results["summary"]["total_commands"] += 1  # type: ignore[index]

                if command_result["success"]:
                    test_results["summary"]["successful_commands"] += 1  # type: ignore[index]
                else:
                    test_results["summary"]["failed_commands"] += 1  # type: ignore[index]
                    test_results["overall_success"] = False

            except Exception as e:
                test_execution = {
                    "command": command,
                    "success": False,
                    "exit_code": -1,
                    "error": str(e),
                    "stdout": "",
                    "stderr": "",
                }

                test_results["test_commands_executed"].append(test_execution)  # type: ignore[attr-defined]
                test_results["summary"]["total_commands"] += 1  # type: ignore[index]
                test_results["summary"]["failed_commands"] += 1  # type: ignore[index]
                test_results["overall_success"] = False

        return test_results
