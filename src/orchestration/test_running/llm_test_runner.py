#!/usr/bin/env python3
"""LLM-based test runner that intelligently analyzes projects and runs tests."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from ...mcp_servers.command.client import CommandMCPClient
from ...mcp_servers.filesystem.client import FilesystemMCPClient


class LLMTestRunner:
    """LLM-powered test runner that uses MCP servers for project analysis and test execution."""

    def __init__(self, project_path: str, llm_model: str = "claude-3-haiku-20240307"):
        """Initialize the LLM test runner.

        Args:
            project_path: Path to the target project
            llm_model: LLM model to use for analysis
        """
        self.project_path = Path(project_path).resolve()
        self.llm = ChatAnthropic(model_name=llm_model, temperature=0)

        # MCP clients for different operations
        self.command_client: Optional[CommandMCPClient] = None
        self.filesystem_client: Optional[FilesystemMCPClient] = None

    def initialize_mcp_clients(self) -> None:
        """Initialize MCP clients for command and filesystem operations."""
        # Initialize command client
        self.command_client = CommandMCPClient(str(self.project_path))
        self.command_client.start_server()

        # Initialize filesystem client
        self.filesystem_client = FilesystemMCPClient(str(self.project_path))
        self.filesystem_client.start_server()

    async def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze project structure to understand testing setup and dependencies."""
        if not self.command_client or not self.filesystem_client:
            raise ValueError("MCP clients not initialized")

        # Find common project files
        project_files = []
        common_files = [
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            "requirements-dev.txt",
            "Pipfile",
            "poetry.lock",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "README.md",
            "README.rst",
            "tox.ini",
            "pytest.ini",
            ".pytest.ini",
            "setup.cfg",
            "Makefile",
            "docker-compose.yml",
            "Dockerfile",
        ]

        for filename in common_files:
            result = self.command_client.call_tool(
                "check_file_exists", {"path": filename}
            )
            if result and "result" in result:
                check_result = json.loads(result["result"][0]["text"])
                if check_result["exists"]:
                    project_files.append(
                        {
                            "name": filename,
                            "type": "file" if check_result["is_file"] else "directory",
                        }
                    )

        # Find test directories
        test_dirs = []
        common_test_dirs = ["tests", "test", "testing", "spec", "specs"]

        for dirname in common_test_dirs:
            result = await self.command_client.call_tool(
                "check_file_exists", {"path": dirname}
            )
            check_result = json.loads(result[0].text)
            if check_result["exists"] and check_result["is_directory"]:
                test_dirs.append(dirname)

        # Find test files in project root
        result = await self.command_client.call_tool(
            "find_files", {"pattern": "test_*.py"}
        )
        test_files_root = json.loads(result[0].text)["matches"]

        result = await self.command_client.call_tool(
            "find_files", {"pattern": "*_test.py"}
        )
        test_files_suffix = json.loads(result[0].text)["matches"]

        return {
            "project_path": str(self.project_path),
            "project_files": project_files,
            "test_directories": test_dirs,
            "test_files": test_files_root + test_files_suffix,
        }

    async def detect_dev_requirements(
        self, project_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to analyze project files and detect development requirements."""

        # Read key project files
        file_contents = {}
        key_files = [
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            "requirements-dev.txt",
            "README.md",
        ]

        for filename in key_files:
            for project_file in project_structure["project_files"]:
                if project_file["name"] == filename and project_file["type"] == "file":
                    try:
                        result = await self.command_client.call_tool(
                            "read_file_content", {"path": filename, "max_lines": 100}
                        )
                        content_result = json.loads(result[0].text)
                        if not content_result.get("truncated", False):
                            file_contents[filename] = "\n".join(
                                content_result["content"]
                            )
                    except Exception:
                        pass  # Skip files we can't read

        # Analyze with LLM
        system_prompt = """You are an expert Python developer analyzing a project to determine its development and testing requirements.

Your task is to:
1. Analyze the provided project files
2. Determine the testing framework being used (pytest, unittest, nose, etc.)
3. Identify development dependencies needed to run tests
4. Suggest the appropriate commands to install dependencies and run tests
5. Identify any special setup or configuration needed

Respond with a JSON object containing:
- "testing_framework": The main testing framework (e.g., "pytest", "unittest")
- "dev_dependencies": List of development packages needed
- "install_commands": List of shell commands to install dependencies  
- "test_commands": List of shell commands to run tests
- "setup_commands": List of any setup commands needed before testing
- "analysis": Brief explanation of your findings

Be conservative and practical in your recommendations. Prefer standard, widely-used approaches."""

        file_contents_text = "\n\n".join(
            [
                f"=== {filename} ===\n{content}"
                for filename, content in file_contents.items()
            ]
        )

        human_prompt = f"""Analyze this Python project to determine how to set up and run its tests:

Project structure:
- Project files found: {[f['name'] for f in project_structure['project_files']]}
- Test directories: {project_structure['test_directories']}
- Test files found: {len(project_structure['test_files'])} test files

File contents:
{file_contents_text}

Please provide your analysis in the JSON format specified."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            # Extract JSON from response
            response_text = response.content
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")

            return json.loads(json_text)

        except Exception as e:
            # Fallback analysis
            return {
                "testing_framework": "pytest",
                "dev_dependencies": ["pytest"],
                "install_commands": ["pip install pytest"],
                "test_commands": ["pytest"],
                "setup_commands": [],
                "analysis": f"Fallback analysis due to parsing error: {str(e)}",
            }

    async def setup_test_environment(
        self, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set up the test environment by installing dependencies."""
        setup_results = {
            "setup_commands_executed": [],
            "install_commands_executed": [],
            "success": True,
            "errors": [],
        }

        # Execute setup commands
        for command in requirements.get("setup_commands", []):
            try:
                result = await self.command_client.call_tool(
                    "execute_command", {"command": command, "timeout": 300}
                )
                command_result = json.loads(result[0].text)
                setup_results["setup_commands_executed"].append(
                    {
                        "command": command,
                        "success": command_result["success"],
                        "exit_code": command_result["exit_code"],
                    }
                )

                if not command_result["success"]:
                    setup_results["success"] = False
                    setup_results["errors"].append(f"Setup command failed: {command}")

            except Exception as e:
                setup_results["success"] = False
                setup_results["errors"].append(
                    f"Error executing setup command '{command}': {str(e)}"
                )

        # Execute install commands
        for command in requirements.get("install_commands", []):
            try:
                result = await self.command_client.call_tool(
                    "execute_command",
                    {"command": command, "timeout": 600},  # Longer timeout for installs
                )
                command_result = json.loads(result[0].text)
                setup_results["install_commands_executed"].append(
                    {
                        "command": command,
                        "success": command_result["success"],
                        "exit_code": command_result["exit_code"],
                    }
                )

                if not command_result["success"]:
                    setup_results["success"] = False
                    setup_results["errors"].append(f"Install command failed: {command}")

            except Exception as e:
                setup_results["success"] = False
                setup_results["errors"].append(
                    f"Error executing install command '{command}': {str(e)}"
                )

        return setup_results

    async def run_tests(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tests using the detected testing framework and commands."""
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
                result = await self.command_client.call_tool(
                    "execute_command", {"command": command, "timeout": 600}
                )
                command_result = json.loads(result[0].text)

                test_execution = {
                    "command": command,
                    "success": command_result["success"],
                    "exit_code": command_result["exit_code"],
                    "stdout": command_result.get("stdout", ""),
                    "stderr": command_result.get("stderr", ""),
                }

                test_results["test_commands_executed"].append(test_execution)
                test_results["summary"]["total_commands"] += 1

                if command_result["success"]:
                    test_results["summary"]["successful_commands"] += 1
                else:
                    test_results["summary"]["failed_commands"] += 1
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

                test_results["test_commands_executed"].append(test_execution)
                test_results["summary"]["total_commands"] += 1
                test_results["summary"]["failed_commands"] += 1
                test_results["overall_success"] = False

        return test_results

    async def analyze_test_results(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to analyze test results and provide insights."""

        system_prompt = """You are an expert at analyzing test results and providing actionable insights.

Your task is to analyze the provided test execution results and provide:
1. A summary of what happened during test execution
2. Identification of any failures or errors
3. Suggestions for fixing issues (if any)
4. Overall assessment of test health

Respond with a JSON object containing:
- "summary": Brief summary of test execution
- "status": "passed", "failed", or "error"
- "issues_found": List of issues identified
- "recommendations": List of actionable recommendations
- "test_metrics": Any metrics you can extract (e.g., number of tests, duration)
"""

        human_prompt = f"""Analyze these test execution results:

Overall Success: {test_results['overall_success']}
Summary: {test_results['summary']}

Detailed Results:
{json.dumps(test_results['test_commands_executed'], indent=2)}

Please provide your analysis in the JSON format specified."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content

            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")

            return json.loads(json_text)

        except Exception as e:
            # Fallback analysis
            status = "passed" if test_results["overall_success"] else "failed"
            return {
                "summary": f"Test execution {'completed successfully' if test_results['overall_success'] else 'failed'}",
                "status": status,
                "issues_found": (
                    [] if test_results["overall_success"] else ["Test execution failed"]
                ),
                "recommendations": (
                    []
                    if test_results["overall_success"]
                    else ["Review test output for specific failures"]
                ),
                "test_metrics": test_results["summary"],
                "analysis_error": f"LLM analysis failed: {str(e)}",
            }

    async def run_full_test_cycle(self) -> Dict[str, Any]:
        """Execute the complete test cycle: analyze, setup, run, and report."""
        try:
            await self.initialize_mcp_clients()

            # Step 1: Analyze project structure
            project_structure = await self.analyze_project_structure()

            # Step 2: Detect requirements using LLM
            requirements = await self.detect_dev_requirements(project_structure)

            # Step 3: Set up test environment
            setup_results = await self.setup_test_environment(requirements)

            # Step 4: Run tests
            test_results = await self.run_tests(requirements)

            # Step 5: Analyze results with LLM
            analysis = await self.analyze_test_results(test_results)

            return {
                "project_structure": project_structure,
                "requirements_analysis": requirements,
                "environment_setup": setup_results,
                "test_execution": test_results,
                "results_analysis": analysis,
                "overall_success": (
                    setup_results.get("success", False)
                    and test_results.get("overall_success", False)
                ),
            }

        except Exception as e:
            return {"error": str(e), "overall_success": False}

        finally:
            # Cleanup MCP clients
            if self.command_client:
                await self.command_client.disconnect()
            if self.filesystem_client:
                await self.filesystem_client.disconnect()


async def run_tests_with_llm(project_path: str) -> Dict[str, Any]:
    """Convenience function to run tests using LLM analysis."""
    runner = LLMTestRunner(project_path)
    return await runner.run_full_test_cycle()


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) != 2:
        print(
            "Usage: python -m src.orchestration.test_running.llm_test_runner <project_path>"
        )
        sys.exit(1)

    project_path = sys.argv[1]

    async def main():
        results = await run_tests_with_llm(project_path)
        print(json.dumps(results, indent=2))

    asyncio.run(main())
