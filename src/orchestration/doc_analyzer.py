"""Documentation analysis for external interface discovery."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, replace, is_dataclass

from langchain_core.tools import BaseTool, tool

from .agent import DiversificationAgent, AgentType
from .mcp_manager import MCPManager, MCPServerType
from .config import get_config


@dataclass
class ExternalInterface:
    """Represents an external interface discovered in documentation."""

    type: str  # "http_api", "websocket", "database", "message_queue"
    name: str
    description: str
    port: Optional[int] = None
    protocol: Optional[str] = None
    authentication: Optional[str] = None
    required_for_testing: bool = True


@dataclass
class DockerServiceInfo:
    """Docker service configuration information."""

    name: str
    container_name: Optional[str]
    exposed_ports: List[int]
    dependencies: List[str]
    environment_variables: List[str]
    health_check: Optional[str] = None


@dataclass
class DocumentationAnalysisResult:
    """Results of documentation analysis."""

    external_interfaces: List[ExternalInterface]
    docker_services: List[DockerServiceInfo]
    network_configuration: Dict[str, Any]
    testing_requirements: Dict[str, Any]
    deployment_patterns: Dict[str, Any]
    analysis_confidence: float  # 0.0 to 1.0


class DocumentationAnalyzer:
    """Analyzer for project documentation to discover external interfaces."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize documentation analyzer.

        Args:
            project_root: Root directory of the project to analyze
            mcp_manager: MCP manager for accessing file system operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.doc_analyzer")

        # Common documentation file patterns
        self.doc_patterns = [
            "README*",
            "readme*",
            "*.md",
            "docs/**/*.md",
            "docs/**/*.rst",
            "API.md",
            "DEPLOYMENT.md",
            "docker-compose.yml",
            "docker-compose.yaml",
            "Dockerfile*",
            ".env.example",
            "openapi.yml",
            "openapi.yaml",
            "swagger.yml",
            "swagger.yaml",
            "*.json",  # API specs
        ]

        # Configuration file patterns
        self.config_patterns = [
            "pyproject.toml",
            "requirements*.txt",
            "package.json",
            "Pipfile",
            "poetry.lock",
            "setup.py",
            "setup.cfg",
            "tox.ini",
        ]

    async def analyze_project_documentation(
        self, model_name: str = "gpt-4"
    ) -> DocumentationAnalysisResult:
        """Analyze project documentation for external interfaces.

        Args:
            model_name: LLM model to use for analysis

        Returns:
            Documentation analysis results
        """
        self.logger.info("Starting documentation analysis")

        # Collect documentation files
        doc_files = await self._collect_documentation_files()
        config_files = await self._collect_configuration_files()

        # Create analyzer agent with file system tools
        file_tools = self._create_file_system_tools()

        # Get base LLM config and override model if specified
        llm_config = get_config().llm
        if model_name != llm_config.model_name:
            # Create a copy with the specified model name
            if is_dataclass(llm_config):
                llm_config = replace(llm_config, model_name=model_name)
            else:
                # For tests with Mock objects
                llm_config.model_name = model_name

        analyzer_agent = DiversificationAgent(
            agent_type=AgentType.DOC_ANALYZER,
            llm_config=llm_config,
            tools=file_tools,
        )

        # Analyze documentation files
        doc_analysis = self._analyze_documentation_files(analyzer_agent, doc_files)

        # Analyze Docker/deployment configuration
        docker_analysis = self._analyze_docker_configuration(
            analyzer_agent, config_files
        )

        # Combine and structure results
        result = self._combine_analysis_results(doc_analysis, docker_analysis)

        self.logger.info(
            f"Documentation analysis complete. Found {len(result.external_interfaces)} external interfaces"
        )

        return result

    async def _collect_documentation_files(self) -> List[Path]:
        """Collect documentation files matching patterns."""
        doc_files = []

        if not self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
            self.logger.warning("Filesystem MCP server not available")
            return []

        try:
            # Find documentation files using patterns
            for pattern in self.doc_patterns:
                # Use glob pattern matching to find files
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
                    doc_files.extend([Path(f) for f in found_files])

        except Exception as e:
            self.logger.error(f"Error collecting documentation files: {e}")

        # Remove duplicates and return
        return list(set(doc_files))

    async def _collect_configuration_files(self) -> List[Path]:
        """Collect configuration files for deployment analysis."""
        config_files = []

        if not self.mcp_manager.is_server_available(MCPServerType.FILESYSTEM):
            return []

        try:
            for pattern in self.config_patterns:
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
        async def read_documentation_file(file_path: str) -> str:
            """Read a documentation file and return its contents.

            Args:
                file_path: Path to the documentation file to read

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
        async def list_project_files(directory_path: str, pattern: str = "*") -> str:
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

        return [read_documentation_file, list_project_files]

    def _analyze_documentation_files(
        self, agent: DiversificationAgent, doc_files: List[Path]
    ) -> Dict[str, Any]:
        """Analyze documentation files using the LLM agent."""

        # Load documentation analyzer prompt
        prompt_path = Path(__file__).parent / "prompts" / "doc_analyzer.txt"
        if prompt_path.exists():
            doc_prompt = prompt_path.read_text()
        else:
            doc_prompt = "Analyze the following documentation to identify external interfaces and APIs."

        # Limit to first 10 files for analysis to avoid overwhelming the LLM
        files_to_analyze = doc_files[:10]

        analysis_prompt = f"""
        {doc_prompt}
        
        ## Task
        Analyze the following project documentation files to identify external interfaces, APIs, and communication patterns.
        
        Focus on discovering:
        1. HTTP/REST API endpoints and their specifications
        2. Database connections and schemas
        3. Message queue integrations
        4. External service dependencies
        5. Network ports and protocols
        6. Authentication and security requirements
        
        ## Documentation Files to Analyze
        {[str(f) for f in files_to_analyze]}
        
        Use the read_documentation_file tool to examine each file and extract external interface information.
        
        Provide your analysis in the specified JSON format.
        """

        try:
            result = agent.invoke(analysis_prompt)

            # Extract response content
            if "output" in result:
                response_text = result["output"]
            elif "messages" in result:
                # Extract from LangGraph response format
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
                # Look for JSON content in the response
                json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    # Try to parse the entire response as JSON
                    return json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, return structured response
                return {
                    "raw_analysis": response_text,
                    "external_interfaces": [],
                    "network_configuration": {},
                    "parsing_error": "Could not parse JSON from LLM response",
                }

        except Exception as e:
            self.logger.error(f"Error analyzing documentation files: {e}")
            return {
                "error": str(e),
                "external_interfaces": [],
                "network_configuration": {},
            }

    def _analyze_docker_configuration(
        self, agent: DiversificationAgent, config_files: List[Path]
    ) -> Dict[str, Any]:
        """Analyze Docker and deployment configuration files."""

        # Load Docker service discovery prompt
        prompt_path = Path(__file__).parent / "prompts" / "docker_service_discovery.txt"
        if prompt_path.exists():
            docker_prompt = prompt_path.read_text()
        else:
            docker_prompt = "Analyze Docker configuration files for service discovery and networking."

        analysis_prompt = f"""
        {docker_prompt}
        
        ## Task
        Analyze the project's Docker and deployment configuration files to understand:
        1. Container networking and service discovery
        2. Inter-service communication patterns
        3. Port mappings and exposures
        4. Service dependencies and startup order
        5. Environment variable requirements
        6. Testing infrastructure needs
        
        ## Configuration Files to Analyze
        {[str(f) for f in config_files]}
        
        Use the read_documentation_file and list_project_files tools to examine configuration files.
        
        Provide your analysis in the specified JSON format focusing on Docker service architecture.
        """

        try:
            result = agent.invoke(analysis_prompt)

            # Extract and parse response similar to documentation analysis
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

            # Extract JSON from response
            try:
                json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    return json.loads(response_text)
            except json.JSONDecodeError:
                return {
                    "raw_analysis": response_text,
                    "service_architecture": {},
                    "network_configuration": {},
                    "parsing_error": "Could not parse JSON from LLM response",
                }

        except Exception as e:
            self.logger.error(f"Error analyzing Docker configuration: {e}")
            return {
                "error": str(e),
                "service_architecture": {},
                "network_configuration": {},
            }

    def _combine_analysis_results(
        self, doc_analysis: Dict[str, Any], docker_analysis: Dict[str, Any]
    ) -> DocumentationAnalysisResult:
        """Combine documentation and Docker analysis results."""

        # Extract external interfaces
        external_interfaces = []
        doc_interfaces = doc_analysis.get("external_interfaces", {})

        # Process HTTP endpoints
        for endpoint in doc_interfaces.get("http_endpoints", []):
            interface = ExternalInterface(
                type="http_api",
                name=endpoint.get("path", "unknown"),
                description=endpoint.get("description", ""),
                port=endpoint.get("port"),
                protocol="HTTP",
                authentication=endpoint.get("authentication"),
            )
            external_interfaces.append(interface)

        # Process databases
        for db in doc_interfaces.get("databases", []):
            interface = ExternalInterface(
                type="database",
                name=db.get("name", "unknown"),
                description=f"{db.get('type', 'unknown')} database",
                port=db.get("default_port"),
                protocol=db.get("type"),
            )
            external_interfaces.append(interface)

        # Process message queues
        for queue in doc_interfaces.get("message_queues", []):
            interface = ExternalInterface(
                type="message_queue",
                name=queue.get("usage", "unknown"),
                description=f"{queue.get('type', 'unknown')} message queue",
                port=queue.get("default_port"),
                protocol=queue.get("type"),
            )
            external_interfaces.append(interface)

        # Extract Docker services
        docker_services = []
        service_arch = docker_analysis.get("service_architecture", {})
        primary_service = service_arch.get("primary_service")

        if primary_service:
            service = DockerServiceInfo(
                name=primary_service.get("name", "unknown"),
                container_name=primary_service.get("container_name"),
                exposed_ports=primary_service.get("exposed_ports", []),
                dependencies=primary_service.get("dependencies", []),
                environment_variables=[],  # Will be populated from analysis
                health_check=primary_service.get("health_check"),
            )
            docker_services.append(service)

        # Extract network configuration
        network_config = {
            "doc_analysis": doc_analysis.get("network_configuration", {}),
            "docker_analysis": docker_analysis.get("network_configuration", {}),
        }

        # Extract testing requirements
        testing_requirements = {
            "doc_considerations": doc_analysis.get("testing_considerations", {}),
            "docker_setup": docker_analysis.get("testing_setup", {}),
        }

        # Extract deployment patterns
        deployment_patterns = {
            "docker_requirements": doc_analysis.get("docker_requirements", {}),
            "deployment_config": docker_analysis.get("deployment_patterns", {}),
        }

        # Calculate confidence based on analysis quality
        confidence = self._calculate_analysis_confidence(doc_analysis, docker_analysis)

        return DocumentationAnalysisResult(
            external_interfaces=external_interfaces,
            docker_services=docker_services,
            network_configuration=network_config,
            testing_requirements=testing_requirements,
            deployment_patterns=deployment_patterns,
            analysis_confidence=confidence,
        )

    def _calculate_analysis_confidence(
        self, doc_analysis: Dict[str, Any], docker_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the analysis results."""
        confidence_factors = []

        # Check if we found external interfaces
        interfaces = doc_analysis.get("external_interfaces", {})
        if (
            interfaces.get("http_endpoints")
            or interfaces.get("databases")
            or interfaces.get("message_queues")
        ):
            confidence_factors.append(0.3)

        # Check if we have Docker configuration
        if docker_analysis.get("service_architecture", {}).get("primary_service"):
            confidence_factors.append(0.3)

        # Check if we have network configuration
        if doc_analysis.get("network_configuration") or docker_analysis.get(
            "network_configuration"
        ):
            confidence_factors.append(0.2)

        # Check if we have testing setup information
        if doc_analysis.get("testing_considerations") or docker_analysis.get(
            "testing_setup"
        ):
            confidence_factors.append(0.2)

        return sum(confidence_factors)

    async def export_analysis_results(
        self, result: DocumentationAnalysisResult, output_path: Optional[str] = None
    ) -> str:
        """Export analysis results to JSON file.

        Args:
            result: Documentation analysis results to export
            output_path: Optional path for output file

        Returns:
            Path to the exported file
        """
        if output_path is None:
            output_path = str(self.project_root / "doc_analysis_results.json")

        # Convert dataclasses to dictionaries for JSON serialization
        export_data = {
            "external_interfaces": [
                {
                    "type": interface.type,
                    "name": interface.name,
                    "description": interface.description,
                    "port": interface.port,
                    "protocol": interface.protocol,
                    "authentication": interface.authentication,
                    "required_for_testing": interface.required_for_testing,
                }
                for interface in result.external_interfaces
            ],
            "docker_services": [
                {
                    "name": service.name,
                    "container_name": service.container_name,
                    "exposed_ports": service.exposed_ports,
                    "dependencies": service.dependencies,
                    "environment_variables": service.environment_variables,
                    "health_check": service.health_check,
                }
                for service in result.docker_services
            ],
            "network_configuration": result.network_configuration,
            "testing_requirements": result.testing_requirements,
            "deployment_patterns": result.deployment_patterns,
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
                self.logger.info(f"Exported analysis results to {output_path}")
            except Exception as e:
                self.logger.error(f"Error exporting results: {e}")
        else:
            # Fallback to direct file write
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Exported analysis results to {output_path}")

        return output_path
