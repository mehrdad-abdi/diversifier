#!/usr/bin/env python3
"""Simplified LLM-based test runner for testing the concept."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from ..config import get_config, MigrationConfig


class SimpleLLMTestRunner:
    """Simplified LLM-powered test runner."""

    def __init__(self, project_path: str, migration_config: MigrationConfig = None):
        """Initialize the simple test runner."""
        self.project_path = Path(project_path).resolve()

        # Use consistent LLM initialization pattern from config
        config = get_config()
        model_id = f"{config.llm.provider.lower()}:{config.llm.model_name}"

        self.llm = init_chat_model(
            model=model_id,
            temperature=0,  # Use low temperature for consistent test analysis
            max_tokens=config.llm.max_tokens,
            **config.llm.additional_params,
        )

        # Store migration config for test analysis configuration
        self.migration_config = migration_config or config.migration

    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze project structure using simple file operations."""
        # Find common project files from configuration
        project_files = []
        common_files = self.migration_config.common_project_files

        for filename in common_files:
            file_path = self.project_path / filename
            if file_path.exists():
                project_files.append(
                    {
                        "name": filename,
                        "type": "file" if file_path.is_file() else "directory",
                    }
                )

        # Find test directories using configured test paths
        test_dirs = []
        primary_test_path = self.migration_config.test_paths[0] if self.migration_config.test_paths else "tests/"
        test_path = primary_test_path.rstrip("/")
        dir_path = self.project_path / test_path
        if dir_path.exists() and dir_path.is_dir():
            test_dirs.append(test_path)

        # Find test files
        test_files = []
        for pattern in ["test_*.py", "*_test.py"]:
            for file_path in self.project_path.rglob(pattern):
                if file_path.is_file():
                    test_files.append(
                        {
                            "path": str(file_path.relative_to(self.project_path)),
                            "name": file_path.name,
                        }
                    )

        return {
            "project_path": str(self.project_path),
            "project_files": project_files,
            "test_directories": test_dirs,
            "test_files": test_files,
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
                if project_file["name"] == filename:
                    try:
                        file_path = self.project_path / filename
                        if file_path.stat().st_size < 10000:  # Only read small files
                            file_contents[filename] = file_path.read_text(
                                encoding="utf-8"
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

    def execute_command(self, command: str, timeout: float = 300) -> Dict[str, Any]:
        """Execute a shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                "command": command,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "error": f"Command timed out after {timeout} seconds",
                "exit_code": -1,
                "success": False,
                "stdout": "",
                "stderr": "",
            }
        except Exception as e:
            return {
                "command": command,
                "error": str(e),
                "exit_code": -1,
                "success": False,
                "stdout": "",
                "stderr": "",
            }

    def setup_test_environment(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Set up the test environment by installing dependencies."""
        setup_results = {
            "setup_commands_executed": [],
            "install_commands_executed": [],
            "success": True,
            "errors": [],
        }

        # Execute setup commands
        for command in requirements.get("setup_commands", []):
            result = self.execute_command(command)
            setup_results["setup_commands_executed"].append(result)

            if not result["success"]:
                setup_results["success"] = False
                setup_results["errors"].append(f"Setup command failed: {command}")

        # Execute install commands
        for command in requirements.get("install_commands", []):
            result = self.execute_command(command, timeout=600)  # Longer timeout
            setup_results["install_commands_executed"].append(result)

            if not result["success"]:
                setup_results["success"] = False
                setup_results["errors"].append(f"Install command failed: {command}")

        return setup_results

    def run_tests(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
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
            result = self.execute_command(command, timeout=600)

            test_results["test_commands_executed"].append(result)
            test_results["summary"]["total_commands"] += 1

            if result["success"]:
                test_results["summary"]["successful_commands"] += 1
            else:
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
            # Step 1: Analyze project structure
            project_structure = self.analyze_project_structure()

            # Step 2: Detect requirements using LLM
            requirements = await self.detect_dev_requirements(project_structure)

            # Step 3: Set up test environment
            setup_results = self.setup_test_environment(requirements)

            # Step 4: Run tests
            test_results = self.run_tests(requirements)

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


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) != 2:
        print(
            "Usage: python -m src.orchestration.test_running.simple_test_runner <project_path>"
        )
        sys.exit(1)

    project_path = sys.argv[1]

    async def main():
        runner = SimpleLLMTestRunner(project_path)
        results = await runner.run_full_test_cycle()
        print(json.dumps(results, indent=2))

    asyncio.run(main())
