"""Source code analysis for external interface discovery."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_core.tools import BaseTool, tool

from .agent import DiversificationAgent, AgentType
from .mcp_manager import MCPManager, MCPServerType


@dataclass
class APIEndpoint:
    """Represents an API endpoint discovered in source code."""

    path: str
    methods: List[str]
    handler: str
    authentication_required: bool
    file_location: str
    request_body_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None


@dataclass
class ExternalServiceIntegration:
    """Represents an external service integration found in code."""

    service_type: str  # "database", "http_client", "message_queue"
    purpose: str
    connection_pattern: str
    configuration_source: str
    file_location: str
    endpoints_or_operations: Optional[List[str]] = None


@dataclass
class ConfigurationUsage:
    """Represents configuration usage patterns in source code."""

    name: str
    purpose: str
    required: bool
    default_value: Optional[str]
    usage_locations: List[str]
    config_type: str  # "environment_variable", "config_file", "hardcoded"


@dataclass
class ExistingTestPattern:
    """Represents existing test patterns found in test files."""

    test_name: str
    test_type: str  # "unit", "integration", "api"
    endpoint_or_feature_tested: str
    file_location: str
    assertions: Optional[List[str]] = None
    mock_usage: Optional[List[str]] = None


@dataclass
class SourceCodeAnalysisResult:
    """Results of source code analysis."""

    api_endpoints: List[APIEndpoint]
    external_service_integrations: List[ExternalServiceIntegration]
    configuration_usage: List[ConfigurationUsage]
    existing_test_patterns: List[ExistingTestPattern]
    network_interfaces: Dict[str, Any]
    security_patterns: Dict[str, Any]
    testing_requirements: Dict[str, Any]
    framework_detected: str
    analysis_confidence: float  # 0.0 to 1.0


class SourceCodeAnalyzer:
    """Analyzer for project source code to discover external interfaces."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize source code analyzer.

        Args:
            project_root: Root directory of the project to analyze
            mcp_manager: MCP manager for accessing file system operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.source_code_analyzer")

        # Python source file patterns
        self.python_file_patterns = [
            "*.py",
            "**/*.py",
            "src/**/*.py",
            "app/**/*.py",
            "myapp/**/*.py",
        ]

        # Test file patterns
        self.test_file_patterns = [
            "test*.py",
            "*test.py",
            "tests/**/*.py",
            "test/**/*.py",
            "**/test_*.py",
        ]

        # Configuration file patterns
        self.config_file_patterns = [
            "settings.py",
            "config.py",
            "*.ini",
            "*.yml",
            "*.yaml",
            "*.json",
            "*.toml",
        ]

        # Framework detection patterns
        self.framework_patterns = {
            "flask": ["from flask import", "Flask(__name__)", "@app.route"],
            "django": ["django.conf", "urlpatterns", "from django"],
            "fastapi": ["from fastapi import", "FastAPI()", "@app.get", "@app.post"],
            "starlette": ["from starlette", "Starlette()", "Route("],
            "tornado": ["tornado.web", "RequestHandler", "Application("],
        }

    async def analyze_project_source_code(
        self, model_name: str = "gpt-4"
    ) -> SourceCodeAnalysisResult:
        """Analyze project source code for external interfaces.

        Args:
            model_name: LLM model to use for analysis

        Returns:
            Source code analysis results
        """
        self.logger.info("Starting source code analysis")

        # Collect source code files
        python_files = await self._collect_python_files()
        test_files = await self._collect_test_files()
        config_files = await self._collect_config_files()

        # Create analyzer agent with file system tools
        from .config import get_config

        file_tools = self._create_file_system_tools()

        # Get base LLM config and override model if specified
        llm_config = get_config().llm
        if model_name != llm_config.model_name:
            # Create a copy with the specified model name
            from dataclasses import replace

            llm_config = replace(llm_config, model_name=model_name)

        analyzer_agent = DiversificationAgent(
            agent_type=AgentType.SOURCE_CODE_ANALYZER,
            llm_config=llm_config,
            tools=file_tools,
        )

        # Analyze source code for external interfaces
        source_analysis = self._analyze_source_code_files(
            analyzer_agent, python_files, config_files
        )

        # Analyze existing test patterns
        test_analysis = self._analyze_test_files(analyzer_agent, test_files)

        # Combine and structure results
        result = self._combine_analysis_results(source_analysis, test_analysis)

        self.logger.info(
            f"Source code analysis complete. Found {len(result.api_endpoints)} API endpoints"
        )

        return result

    async def _collect_python_files(self) -> List[Path]:
        """Collect Python source files matching patterns."""
        python_files = []

        if not self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
            self.logger.warning("Filesystem MCP server not available")
            return []

        try:
            for pattern in self.python_file_patterns:
                files_result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "list_files",
                    {
                        "directory_path": str(self.project_root),
                        "pattern": pattern,
                        "recursive": True,
                    },
                )

                if files_result and "result" in files_result:
                    file_data = json.loads(files_result["result"][0]["text"])
                    found_files = file_data.get("files", [])
                    python_files.extend([Path(f) for f in found_files])

        except Exception as e:
            self.logger.error(f"Error collecting Python files: {e}")

        # Remove test files from python_files (they'll be handled separately)
        filtered_files = []
        for file_path in python_files:
            if not any(
                test_pattern in str(file_path).lower()
                for test_pattern in ["test", "spec"]
            ):
                filtered_files.append(file_path)

        return list(set(filtered_files))

    async def _collect_test_files(self) -> List[Path]:
        """Collect test files matching patterns."""
        test_files = []

        if not self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
            return []

        try:
            for pattern in self.test_file_patterns:
                files_result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "list_files",
                    {
                        "directory_path": str(self.project_root),
                        "pattern": pattern,
                        "recursive": True,
                    },
                )

                if files_result and "result" in files_result:
                    file_data = json.loads(files_result["result"][0]["text"])
                    found_files = file_data.get("files", [])
                    test_files.extend([Path(f) for f in found_files])

        except Exception as e:
            self.logger.error(f"Error collecting test files: {e}")

        return list(set(test_files))

    async def _collect_config_files(self) -> List[Path]:
        """Collect configuration files for analysis."""
        config_files = []

        if not self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
            return []

        try:
            for pattern in self.config_file_patterns:
                files_result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "list_files",
                    {
                        "directory_path": str(self.project_root),
                        "pattern": pattern,
                        "recursive": True,
                    },
                )

                if files_result and "result" in files_result:
                    file_data = json.loads(files_result["result"][0]["text"])
                    found_files = file_data.get("files", [])
                    config_files.extend([Path(f) for f in found_files])

        except Exception as e:
            self.logger.error(f"Error collecting configuration files: {e}")

        return list(set(config_files))

    def _create_file_system_tools(self) -> List[BaseTool]:
        """Create file system tools for the analyzer agent."""

        @tool
        async def read_source_file(file_path: str) -> str:
            """Read a source code file and return its contents.

            Args:
                file_path: Path to the source file to read

            Returns:
                File contents as string
            """
            try:
                result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM, "read_file", {"file_path": file_path}
                )

                if result and "result" in result:
                    return result["result"][0]["text"]
                else:
                    return f"Error reading file: {file_path}"

            except Exception as e:
                return f"Error reading file {file_path}: {str(e)}"

        @tool
        async def search_code_patterns(directory_path: str, pattern: str) -> str:
            """Search for specific code patterns in files.

            Args:
                directory_path: Directory to search in
                pattern: Regex pattern to search for

            Returns:
                JSON string with search results
            """
            try:
                result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "search_files",
                    {
                        "directory_path": directory_path,
                        "pattern": pattern,
                        "file_extensions": [".py"],
                    },
                )

                if result and "result" in result:
                    return result["result"][0]["text"]
                else:
                    return json.dumps(
                        {"error": f"Failed to search patterns in {directory_path}"}
                    )

            except Exception as e:
                return json.dumps({"error": f"Error searching patterns: {str(e)}"})

        @tool
        async def list_project_files(directory_path: str, pattern: str = "*.py") -> str:
            """List files in a project directory.

            Args:
                directory_path: Directory to list files from
                pattern: Glob pattern to filter files

            Returns:
                JSON string with file listing results
            """
            try:
                result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "list_files",
                    {
                        "directory_path": directory_path,
                        "pattern": pattern,
                        "recursive": True,
                    },
                )

                if result and "result" in result:
                    return result["result"][0]["text"]
                else:
                    return json.dumps(
                        {"error": f"Failed to list files in {directory_path}"}
                    )

            except Exception as e:
                return json.dumps({"error": f"Error listing files: {str(e)}"})

        return [read_source_file, search_code_patterns, list_project_files]

    def _analyze_source_code_files(
        self,
        agent: DiversificationAgent,
        python_files: List[Path],
        config_files: List[Path],
    ) -> Dict[str, Any]:
        """Analyze source code files using the LLM agent."""

        # Load source code analyzer prompt
        prompt_path = Path(__file__).parent / "prompts" / "source_code_analyzer.txt"
        if prompt_path.exists():
            source_prompt = prompt_path.read_text()
        else:
            source_prompt = "Analyze the following source code to identify external interfaces and APIs."

        # Limit files for analysis to avoid overwhelming the LLM
        files_to_analyze = python_files[:15]  # Analyze up to 15 source files
        config_to_analyze = config_files[:5]  # Analyze up to 5 config files

        analysis_prompt = f"""
        {source_prompt}
        
        ## Task
        Analyze the following Python source code files and configuration files to identify:
        1. API endpoints and HTTP handlers
        2. External service integrations and dependencies
        3. Configuration usage patterns and environment variables
        4. Network interfaces and communication protocols
        5. Security and authentication patterns
        
        ## Source Code Files to Analyze
        {[str(f) for f in files_to_analyze]}
        
        ## Configuration Files to Analyze  
        {[str(f) for f in config_to_analyze]}
        
        Use the read_source_file tool to examine each file and extract external interface information.
        Use the search_code_patterns tool to find specific patterns across multiple files.
        
        Focus on externally observable behavior and interfaces that acceptance tests can validate.
        
        Provide your analysis in the specified JSON format.
        """

        try:
            result = agent.invoke(analysis_prompt)

            # Extract response content
            if "output" in result:
                response_text = result["output"]
            elif "messages" in result:
                last_message = result["messages"][-1]
                response_text = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
            else:
                response_text = str(result)

            # Try to extract JSON from the response
            try:
                import re

                json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    return json.loads(response_text)
            except json.JSONDecodeError:
                return {
                    "raw_analysis": response_text,
                    "api_endpoints": {"http_endpoints": []},
                    "external_service_integrations": {},
                    "configuration_patterns": {},
                    "parsing_error": "Could not parse JSON from LLM response",
                }

        except Exception as e:
            self.logger.error(f"Error analyzing source code files: {e}")
            return {
                "error": str(e),
                "api_endpoints": {"http_endpoints": []},
                "external_service_integrations": {},
                "configuration_patterns": {},
            }

    def _analyze_test_files(
        self, agent: DiversificationAgent, test_files: List[Path]
    ) -> Dict[str, Any]:
        """Analyze existing test files to understand testing patterns."""

        if not test_files:
            return {"existing_test_patterns": {}}

        test_analysis_prompt = f"""
        Analyze the following test files to understand existing testing patterns, coverage, and approaches:
        
        ## Test Files to Analyze
        {[str(f) for f in test_files[:10]]}  
        
        Focus on:
        1. API endpoint testing patterns
        2. External service mocking strategies
        3. Test data setup and fixtures
        4. Integration test approaches
        5. Test coverage gaps
        
        Use the read_source_file tool to examine test files.
        
        Provide analysis of existing test patterns in JSON format.
        """

        try:
            result = agent.invoke(test_analysis_prompt)

            # Extract and parse response
            if "output" in result:
                response_text = result["output"]
            elif "messages" in result:
                last_message = result["messages"][-1]
                response_text = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
            else:
                response_text = str(result)

            try:
                import re

                json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    return json.loads(response_text)
            except json.JSONDecodeError:
                return {
                    "raw_analysis": response_text,
                    "existing_test_patterns": {},
                    "parsing_error": "Could not parse JSON from LLM response",
                }

        except Exception as e:
            self.logger.error(f"Error analyzing test files: {e}")
            return {"error": str(e), "existing_test_patterns": {}}

    def _combine_analysis_results(
        self, source_analysis: Dict[str, Any], test_analysis: Dict[str, Any]
    ) -> SourceCodeAnalysisResult:
        """Combine source code and test analysis results."""

        # Extract API endpoints
        api_endpoints = []
        endpoints_data = source_analysis.get("api_endpoints", {}).get(
            "http_endpoints", []
        )

        for endpoint_data in endpoints_data:
            endpoint = APIEndpoint(
                path=endpoint_data.get("path", "unknown"),
                methods=endpoint_data.get("methods", []),
                handler=endpoint_data.get("handler", ""),
                authentication_required=endpoint_data.get(
                    "authentication_required", False
                ),
                file_location=endpoint_data.get("file_location", ""),
                request_body_schema=endpoint_data.get("request_body_schema"),
                response_schema=endpoint_data.get("response_schema"),
            )
            api_endpoints.append(endpoint)

        # Extract external service integrations
        external_integrations = []
        integrations_data = source_analysis.get("external_service_integrations", {})

        for service_type, services in integrations_data.items():
            if isinstance(services, list):
                for service_data in services:
                    integration = ExternalServiceIntegration(
                        service_type=service_type,
                        purpose=service_data.get("purpose", ""),
                        connection_pattern=service_data.get("connection_pattern", ""),
                        configuration_source=service_data.get("connection_config", ""),
                        file_location=service_data.get("file_location", ""),
                        endpoints_or_operations=service_data.get(
                            "endpoints_called", []
                        ),
                    )
                    external_integrations.append(integration)

        # Extract configuration usage
        config_usage = []
        config_data = source_analysis.get("configuration_patterns", {})
        env_vars = config_data.get("environment_variables", [])

        for var_data in env_vars:
            config = ConfigurationUsage(
                name=var_data.get("name", ""),
                purpose=var_data.get("purpose", ""),
                required=var_data.get("required", False),
                default_value=var_data.get("default_value"),
                usage_locations=var_data.get("usage_locations", []),
                config_type="environment_variable",
            )
            config_usage.append(config)

        # Extract existing test patterns
        existing_tests = []
        test_patterns = test_analysis.get("existing_test_patterns", {})
        api_tests = test_patterns.get("api_test_examples", [])

        for test_data in api_tests:
            test_pattern = ExistingTestPattern(
                test_name=test_data.get("test_name", ""),
                test_type="api",
                endpoint_or_feature_tested=test_data.get("endpoint_tested", ""),
                file_location=test_data.get("file_location", ""),
                assertions=test_data.get("assertions", []),
                mock_usage=test_data.get("mock_usage", []),
            )
            existing_tests.append(test_pattern)

        # Detect framework
        framework_detected = source_analysis.get("analysis_metadata", {}).get(
            "framework_detected", "unknown"
        )

        # Calculate confidence
        confidence = self._calculate_analysis_confidence(source_analysis, test_analysis)

        return SourceCodeAnalysisResult(
            api_endpoints=api_endpoints,
            external_service_integrations=external_integrations,
            configuration_usage=config_usage,
            existing_test_patterns=existing_tests,
            network_interfaces=source_analysis.get("network_interfaces", {}),
            security_patterns=source_analysis.get("security_considerations", {}),
            testing_requirements=source_analysis.get("testing_requirements", {}),
            framework_detected=framework_detected,
            analysis_confidence=confidence,
        )

    def _calculate_analysis_confidence(
        self, source_analysis: Dict[str, Any], test_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the analysis results."""
        confidence_factors = []

        # Check if we found API endpoints
        endpoints = source_analysis.get("api_endpoints", {}).get("http_endpoints", [])
        if endpoints:
            confidence_factors.append(0.3)

        # Check if we found external integrations
        integrations = source_analysis.get("external_service_integrations", {})
        if any(integrations.values()):
            confidence_factors.append(0.25)

        # Check if we found configuration patterns
        config_patterns = source_analysis.get("configuration_patterns", {})
        if config_patterns.get("environment_variables") or config_patterns.get(
            "config_files"
        ):
            confidence_factors.append(0.2)

        # Check if we found existing tests
        test_patterns = test_analysis.get("existing_test_patterns", {})
        if test_patterns.get("api_test_examples") or test_patterns.get("test_types"):
            confidence_factors.append(0.15)

        # Check if we detected a framework
        framework = source_analysis.get("analysis_metadata", {}).get(
            "framework_detected"
        )
        if framework and framework != "unknown":
            confidence_factors.append(0.1)

        return sum(confidence_factors)

    async def export_analysis_results(
        self, result: SourceCodeAnalysisResult, output_path: Optional[str] = None
    ) -> str:
        """Export analysis results to JSON file.

        Args:
            result: Source code analysis results to export
            output_path: Optional path for output file

        Returns:
            Path to the exported file
        """
        if output_path is None:
            output_path = str(self.project_root / "source_code_analysis_results.json")

        # Convert dataclasses to dictionaries for JSON serialization
        export_data = {
            "api_endpoints": [
                {
                    "path": endpoint.path,
                    "methods": endpoint.methods,
                    "handler": endpoint.handler,
                    "authentication_required": endpoint.authentication_required,
                    "file_location": endpoint.file_location,
                    "request_body_schema": endpoint.request_body_schema,
                    "response_schema": endpoint.response_schema,
                }
                for endpoint in result.api_endpoints
            ],
            "external_service_integrations": [
                {
                    "service_type": integration.service_type,
                    "purpose": integration.purpose,
                    "connection_pattern": integration.connection_pattern,
                    "configuration_source": integration.configuration_source,
                    "file_location": integration.file_location,
                    "endpoints_or_operations": integration.endpoints_or_operations,
                }
                for integration in result.external_service_integrations
            ],
            "configuration_usage": [
                {
                    "name": config.name,
                    "purpose": config.purpose,
                    "required": config.required,
                    "default_value": config.default_value,
                    "usage_locations": config.usage_locations,
                    "config_type": config.config_type,
                }
                for config in result.configuration_usage
            ],
            "existing_test_patterns": [
                {
                    "test_name": test.test_name,
                    "test_type": test.test_type,
                    "endpoint_or_feature_tested": test.endpoint_or_feature_tested,
                    "file_location": test.file_location,
                    "assertions": test.assertions,
                    "mock_usage": test.mock_usage,
                }
                for test in result.existing_test_patterns
            ],
            "network_interfaces": result.network_interfaces,
            "security_patterns": result.security_patterns,
            "testing_requirements": result.testing_requirements,
            "framework_detected": result.framework_detected,
            "analysis_confidence": result.analysis_confidence,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write to file using MCP server if available
        if self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
            try:
                await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "write_file",
                    {
                        "file_path": output_path,
                        "content": json.dumps(export_data, indent=2),
                        "create_backup": False,
                    },
                )
                self.logger.info(
                    f"Exported source code analysis results to {output_path}"
                )
            except Exception as e:
                self.logger.error(f"Error exporting results: {e}")
        else:
            # Fallback to direct file write
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Exported source code analysis results to {output_path}")

        return output_path
