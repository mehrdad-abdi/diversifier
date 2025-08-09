"""Acceptance test generation for black-box Docker-based testing."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_core.tools import BaseTool, tool

from .agent import DiversificationAgent, AgentType
from .mcp_manager import MCPManager, MCPServerType
from .doc_analyzer import DocumentationAnalysisResult
from .source_code_analyzer import SourceCodeAnalysisResult


@dataclass
class AcceptanceTestSuite:
    """Represents a generated acceptance test suite."""

    name: str
    description: str
    test_file_content: str
    docker_compose_content: Optional[str] = None
    dockerfile_content: Optional[str] = None
    requirements_content: Optional[str] = None


@dataclass
class AcceptanceTestScenario:
    """Represents a specific test scenario."""

    category: (
        str  # "http_api", "service_lifecycle", "network", "config", "error", "docker"
    )
    name: str
    description: str
    test_method_name: str
    test_code: str
    dependencies: List[str]
    docker_services: List[str]


@dataclass
class AcceptanceTestGenerationResult:
    """Results of acceptance test generation."""

    test_suites: List[AcceptanceTestSuite]
    test_scenarios: List[AcceptanceTestScenario]
    docker_configuration: Dict[str, Any]
    test_dependencies: List[str]
    coverage_analysis: Dict[str, Any]
    generation_confidence: float  # 0.0 to 1.0


class AcceptanceTestGenerator:
    """Generator for comprehensive black-box acceptance tests."""

    def __init__(self, project_root: str, mcp_manager: MCPManager):
        """Initialize acceptance test generator.

        Args:
            project_root: Root directory of the project to generate tests for
            mcp_manager: MCP manager for accessing file system operations
        """
        self.project_root = Path(project_root)
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.acceptance_test_generator")

        # Test framework configurations
        self.supported_frameworks = {
            "flask": {
                "default_port": 5000,
                "health_endpoint": "/health",
                "api_prefix": "/api",
                "test_patterns": ["test_*.py", "*_test.py"],
            },
            "django": {
                "default_port": 8000,
                "health_endpoint": "/health/",
                "api_prefix": "/api/",
                "test_patterns": ["test_*.py", "*_test.py"],
            },
            "fastapi": {
                "default_port": 8000,
                "health_endpoint": "/health",
                "api_prefix": "/api",
                "test_patterns": ["test_*.py", "*_test.py"],
            },
        }

        # Docker configuration templates
        self.docker_templates = {
            "test_dockerfile": """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY tests/ ./tests/
COPY conftest.py .

CMD ["python", "-m", "pytest", "tests/", "-v"]
""",
            "docker_compose_template": """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - DATABASE_URL=postgresql://postgres:password@db:5432/testdb
    depends_on:
      - db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  test:
    build:
      context: .
      dockerfile: Dockerfile.test
    depends_on:
      - app
    environment:
      - BASE_URL=http://app:8000
    volumes:
      - ./test-results:/app/test-results

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=testdb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
""",
        }

    async def generate_acceptance_tests(
        self,
        doc_analysis: DocumentationAnalysisResult,
        source_analysis: SourceCodeAnalysisResult,
        model_name: str = "gpt-4",
    ) -> AcceptanceTestGenerationResult:
        """Generate comprehensive acceptance tests based on analysis results.

        Args:
            doc_analysis: Documentation analysis results
            source_analysis: Source code analysis results
            model_name: LLM model to use for test generation

        Returns:
            Acceptance test generation results
        """
        self.logger.info("Starting acceptance test generation")

        # Create test generator agent with file system tools
        file_tools = self._create_file_system_tools()
        generator_agent = DiversificationAgent(
            agent_type=AgentType.ACCEPTANCE_TEST_GENERATOR,
            model_name=model_name,
            temperature=0.2,  # Slightly higher for creative test generation
            tools=file_tools,
        )

        # Generate different categories of tests
        http_tests = await self._generate_http_api_tests(
            generator_agent, source_analysis
        )
        lifecycle_tests = await self._generate_service_lifecycle_tests(
            generator_agent, doc_analysis, source_analysis
        )
        network_tests = await self._generate_network_communication_tests(
            generator_agent, source_analysis
        )
        config_tests = await self._generate_configuration_tests(
            generator_agent, source_analysis
        )
        error_tests = await self._generate_error_handling_tests(
            generator_agent, source_analysis
        )
        docker_tests = await self._generate_docker_orchestration_tests(
            generator_agent, doc_analysis
        )

        # Combine all test scenarios
        all_scenarios = []
        all_scenarios.extend(http_tests.get("scenarios", []))
        all_scenarios.extend(lifecycle_tests.get("scenarios", []))
        all_scenarios.extend(network_tests.get("scenarios", []))
        all_scenarios.extend(config_tests.get("scenarios", []))
        all_scenarios.extend(error_tests.get("scenarios", []))
        all_scenarios.extend(docker_tests.get("scenarios", []))

        # Generate consolidated test suites
        test_suites = self._create_test_suites(all_scenarios, source_analysis)

        # Generate Docker configuration
        docker_config = self._generate_docker_configuration(
            doc_analysis, source_analysis
        )

        # Analyze test coverage
        coverage_analysis = self._analyze_test_coverage(all_scenarios, source_analysis)

        # Calculate generation confidence
        confidence = self._calculate_generation_confidence(
            all_scenarios, doc_analysis, source_analysis
        )

        result = AcceptanceTestGenerationResult(
            test_suites=test_suites,
            test_scenarios=all_scenarios,
            docker_configuration=docker_config,
            test_dependencies=self._get_test_dependencies(),
            coverage_analysis=coverage_analysis,
            generation_confidence=confidence,
        )

        self.logger.info(
            f"Generated {len(test_suites)} test suites with {len(all_scenarios)} scenarios"
        )

        return result

    def _create_file_system_tools(self) -> List[BaseTool]:
        """Create file system tools for the generator agent."""

        @tool
        async def write_test_file(file_path: str, content: str) -> str:
            """Write a test file to the project.

            Args:
                file_path: Path where to write the test file
                content: Test file content

            Returns:
                Success message
            """
            try:
                result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM,
                    "write_file",
                    {"file_path": file_path, "content": content},
                )

                if result and "success" in str(result).lower():
                    return f"Successfully wrote test file to {file_path}"
                else:
                    return f"Error writing test file to {file_path}"

            except Exception as e:
                return f"Error writing test file {file_path}: {str(e)}"

        @tool
        async def read_existing_test(file_path: str) -> str:
            """Read an existing test file for reference.

            Args:
                file_path: Path to the test file to read

            Returns:
                Test file content
            """
            try:
                result = await self.mcp_manager.call_tool(
                    MCPServerType.FILESYSTEM, "read_file", {"file_path": file_path}
                )

                if result and "result" in result:
                    return result["result"][0]["text"]
                else:
                    return f"Could not read test file: {file_path}"

            except Exception as e:
                return f"Error reading test file {file_path}: {str(e)}"

        return [write_test_file, read_existing_test]

    async def _generate_http_api_tests(
        self, agent: DiversificationAgent, source_analysis: SourceCodeAnalysisResult
    ) -> Dict[str, Any]:
        """Generate HTTP/API endpoint tests."""
        endpoints = source_analysis.api_endpoints
        if not endpoints:
            return {"scenarios": []}

        prompt = f"""
        Generate comprehensive HTTP/API endpoint tests for the following discovered endpoints:

        ## API Endpoints Discovered:
        {json.dumps([{
            'path': ep.path,
            'methods': ep.methods,
            'handler': ep.handler,
            'auth_required': ep.authentication_required,
            'location': ep.file_location
        } for ep in endpoints], indent=2)}

        ## Framework Detected:
        {source_analysis.framework_detected}

        Generate black-box acceptance tests that:
        1. Test each endpoint with valid requests
        2. Test authentication/authorization if required
        3. Test different HTTP methods (GET, POST, PUT, DELETE, etc.)
        4. Test request/response formats (JSON, form data)
        5. Test query parameters and path parameters
        6. Test content-type handling
        7. Test response status codes and structures

        Focus on externally observable behavior only. Generate complete pytest test methods.

        Provide results as JSON with test scenarios.
        """

        try:
            result = agent.invoke(prompt)
            response_text = self._extract_response_text(result)
            return self._parse_test_response(response_text, "http_api")
        except Exception as e:
            self.logger.error(f"Error generating HTTP tests: {e}")
            return {"scenarios": []}

    async def _generate_service_lifecycle_tests(
        self,
        agent: DiversificationAgent,
        doc_analysis: DocumentationAnalysisResult,
        source_analysis: SourceCodeAnalysisResult,
    ) -> Dict[str, Any]:
        """Generate service lifecycle and health check tests."""
        prompt = f"""
        Generate service lifecycle tests for application startup, health checks, and shutdown:

        ## Service Configuration:
        Framework: {source_analysis.framework_detected}
        Docker Services: {len(doc_analysis.docker_services)} found
        Network Interfaces: {source_analysis.network_interfaces}

        Generate tests for:
        1. Application container startup and readiness
        2. Health check endpoint validation
        3. Service dependency readiness (database, cache, etc.)
        4. Graceful shutdown handling
        5. Container resource monitoring
        6. Service recovery after failure

        Focus on external observable behavior through network interfaces.
        Generate complete pytest test methods that work in Docker environment.

        Provide results as JSON with test scenarios.
        """

        try:
            result = agent.invoke(prompt)
            response_text = self._extract_response_text(result)
            return self._parse_test_response(response_text, "service_lifecycle")
        except Exception as e:
            self.logger.error(f"Error generating lifecycle tests: {e}")
            return {"scenarios": []}

    async def _generate_network_communication_tests(
        self, agent: DiversificationAgent, source_analysis: SourceCodeAnalysisResult
    ) -> Dict[str, Any]:
        """Generate network communication and integration tests."""
        integrations = source_analysis.external_service_integrations
        if not integrations:
            return {"scenarios": []}

        prompt = f"""
        Generate network communication and integration tests:

        ## External Service Integrations:
        {json.dumps([{
            'service_type': integration.service_type,
            'purpose': integration.purpose,
            'connection_pattern': integration.connection_pattern,
            'config_source': integration.configuration_source
        } for integration in integrations], indent=2)}

        Generate tests for:
        1. Database connectivity and data persistence
        2. External API integrations and responses
        3. Message queue communication patterns
        4. WebSocket connections and real-time features
        5. Inter-service communication
        6. Network timeout and resilience handling

        Focus on black-box validation through network interfaces only.
        Generate complete pytest test methods for Docker environment.

        Provide results as JSON with test scenarios.
        """

        try:
            result = agent.invoke(prompt)
            response_text = self._extract_response_text(result)
            return self._parse_test_response(response_text, "network")
        except Exception as e:
            self.logger.error(f"Error generating network tests: {e}")
            return {"scenarios": []}

    async def _generate_configuration_tests(
        self, agent: DiversificationAgent, source_analysis: SourceCodeAnalysisResult
    ) -> Dict[str, Any]:
        """Generate configuration and environment variable tests."""
        config_usage = source_analysis.configuration_usage
        if not config_usage:
            return {"scenarios": []}

        prompt = f"""
        Generate configuration and environment variable tests:

        ## Configuration Usage:
        {json.dumps([{
            'name': config.name,
            'purpose': config.purpose,
            'required': config.required,
            'default_value': config.default_value,
            'config_type': config.config_type
        } for config in config_usage], indent=2)}

        Generate tests for:
        1. Required environment variables validation
        2. Configuration with different values
        3. Default value handling
        4. Invalid configuration handling
        5. Configuration reload behavior
        6. Environment-specific settings

        Focus on externally observable configuration behavior.
        Generate complete pytest test methods for Docker testing.

        Provide results as JSON with test scenarios.
        """

        try:
            result = agent.invoke(prompt)
            response_text = self._extract_response_text(result)
            return self._parse_test_response(response_text, "config")
        except Exception as e:
            self.logger.error(f"Error generating config tests: {e}")
            return {"scenarios": []}

    async def _generate_error_handling_tests(
        self, agent: DiversificationAgent, source_analysis: SourceCodeAnalysisResult
    ) -> Dict[str, Any]:
        """Generate error handling and edge case tests."""
        endpoints = source_analysis.api_endpoints
        integrations = source_analysis.external_service_integrations

        prompt = f"""
        Generate error handling and edge case tests:

        ## API Endpoints: {len(endpoints)}
        ## External Services: {len(integrations)}
        ## Framework: {source_analysis.framework_detected}

        Generate tests for:
        1. HTTP 4xx error responses (400, 401, 403, 404, 422)
        2. HTTP 5xx error responses (500, 502, 503)
        3. Invalid request data and malformed input
        4. Network timeout and connection failures
        5. Large request/response handling
        6. Concurrent request processing
        7. Rate limiting and throttling behavior
        8. External service failure handling

        Focus on black-box error condition validation.
        Generate complete pytest test methods.

        Provide results as JSON with test scenarios.
        """

        try:
            result = agent.invoke(prompt)
            response_text = self._extract_response_text(result)
            return self._parse_test_response(response_text, "error")
        except Exception as e:
            self.logger.error(f"Error generating error tests: {e}")
            return {"scenarios": []}

    async def _generate_docker_orchestration_tests(
        self,
        agent: DiversificationAgent,
        doc_analysis: DocumentationAnalysisResult,
    ) -> Dict[str, Any]:
        """Generate Docker container interaction and orchestration tests."""
        docker_services = doc_analysis.docker_services
        if not docker_services:
            return {"scenarios": []}

        prompt = f"""
        Generate Docker container orchestration tests:

        ## Docker Services:
        {json.dumps([{
            'name': service.name,
            'exposed_ports': service.exposed_ports,
            'dependencies': service.dependencies,
            'health_check': service.health_check
        } for service in docker_services], indent=2)}

        Generate tests for:
        1. Container startup order and dependencies
        2. Service health checks and readiness probes
        3. Container resource usage monitoring
        4. Inter-container communication
        5. Container log output validation
        6. Volume mounting and data persistence
        7. Network connectivity between services
        8. Container failure and recovery

        Focus on Docker orchestration behavior validation.
        Generate complete pytest test methods using subprocess and docker commands.

        Provide results as JSON with test scenarios.
        """

        try:
            result = agent.invoke(prompt)
            response_text = self._extract_response_text(result)
            return self._parse_test_response(response_text, "docker")
        except Exception as e:
            self.logger.error(f"Error generating Docker tests: {e}")
            return {"scenarios": []}

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

    def _parse_test_response(self, response_text: str, category: str) -> Dict[str, Any]:
        """Parse LLM response into structured test scenarios."""
        try:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response_text)

            # Convert to AcceptanceTestScenario objects
            scenarios = []
            test_scenarios = data.get("test_scenarios", [])

            for scenario_data in test_scenarios:
                scenario = AcceptanceTestScenario(
                    category=category,
                    name=scenario_data.get("name", "Unknown Test"),
                    description=scenario_data.get("description", ""),
                    test_method_name=scenario_data.get("test_method", "test_unknown"),
                    test_code=scenario_data.get("test_code", "pass"),
                    dependencies=scenario_data.get("dependencies", []),
                    docker_services=scenario_data.get("docker_services", []),
                )
                scenarios.append(scenario)

            return {"scenarios": scenarios}

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to parse test response for {category}: {e}")
            return {"scenarios": []}

    def _create_test_suites(
        self,
        scenarios: List[AcceptanceTestScenario],
        source_analysis: SourceCodeAnalysisResult,
    ) -> List[AcceptanceTestSuite]:
        """Create organized test suites from scenarios."""
        # Group scenarios by category
        categories: Dict[str, List[AcceptanceTestScenario]] = {}
        for scenario in scenarios:
            if scenario.category not in categories:
                categories[scenario.category] = []
            categories[scenario.category].append(scenario)

        test_suites = []

        for category, category_scenarios in categories.items():
            # Generate test file content
            test_content = self._generate_test_file_content(
                category, category_scenarios, source_analysis
            )

            # Create test suite
            suite = AcceptanceTestSuite(
                name=f"{category}_acceptance_tests",
                description=f"Acceptance tests for {category.replace('_', ' ')} functionality",
                test_file_content=test_content,
            )
            test_suites.append(suite)

        # Add Docker configuration files
        main_suite = AcceptanceTestSuite(
            name="docker_configuration",
            description="Docker configuration for acceptance testing",
            test_file_content="",  # No test code, just config
            docker_compose_content=self._generate_docker_compose_content(
                source_analysis
            ),
            dockerfile_content=self.docker_templates["test_dockerfile"],
            requirements_content=self._generate_requirements_content(),
        )
        test_suites.append(main_suite)

        return test_suites

    def _generate_test_file_content(
        self,
        category: str,
        scenarios: List[AcceptanceTestScenario],
        source_analysis: SourceCodeAnalysisResult,
    ) -> str:
        """Generate complete test file content for a category."""
        class_name = f"Test{category.title().replace('_', '')}"

        imports = """import pytest
import requests
import json
import time
import subprocess
import websocket
import concurrent.futures
from typing import Dict, Any

# Test configuration
BASE_URL = "http://app:8000"
TEST_TIMEOUT = 30
"""

        test_methods = []
        for scenario in scenarios:
            test_methods.append(
                f"""
    def {scenario.test_method_name}(self):
        \"\"\"{scenario.description}\"\"\"
{scenario.test_code}
"""
            )

        helper_functions = """
# Helper functions
def get_app_container_name() -> str:
    return "app_container"

def get_app_host() -> str:
    return "localhost"

def get_app_port() -> int:
    return 8000

def wait_for_service_ready(url: str, timeout: int = 30) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    
    pytest.fail(f"Service at {url} failed to become ready within {timeout} seconds")
"""

        return f'''"""
{class_name} - Acceptance tests for {category.replace('_', ' ')} functionality.

Generated by Diversifier Acceptance Test Generator.
These tests validate external interfaces and behavior through black-box testing.
All tests run in Docker containers and communicate via network interfaces only.
"""

{imports}

class {class_name}:
    """Test {category.replace('_', ' ')} functionality."""
{''.join(test_methods)}

{helper_functions}
'''

    def _generate_docker_compose_content(
        self, source_analysis: SourceCodeAnalysisResult
    ) -> str:
        """Generate Docker Compose configuration for testing."""
        # Customize based on analysis
        framework = source_analysis.framework_detected.lower()
        default_port = self.supported_frameworks.get(framework, {}).get(
            "default_port", 8000
        )

        # Basic template with customizations
        compose_content = self.docker_templates["docker_compose_template"]

        # Replace port if different
        if default_port != 8000:
            compose_content = compose_content.replace(
                "8000:8000", f"{default_port}:{default_port}"
            )
            compose_content = compose_content.replace(
                "http://app:8000", f"http://app:{default_port}"
            )

        return compose_content

    def _generate_requirements_content(self) -> str:
        """Generate test requirements.txt content."""
        return """pytest>=7.0.0
requests>=2.28.0
websocket-client>=1.4.0
docker>=6.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-xdist>=3.0.0
"""

    def _get_test_dependencies(self) -> List[str]:
        """Get list of test dependencies."""
        return [
            "pytest",
            "requests",
            "websocket-client",
            "docker",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-xdist",
        ]

    def _generate_docker_configuration(
        self,
        doc_analysis: DocumentationAnalysisResult,
        source_analysis: SourceCodeAnalysisResult,
    ) -> Dict[str, Any]:
        """Generate Docker configuration for testing."""
        return {
            "test_dockerfile": self.docker_templates["test_dockerfile"],
            "docker_compose": self._generate_docker_compose_content(source_analysis),
            "network_config": {
                "test_network": "diversifier_test_network",
                "app_service": "app",
                "test_service": "test",
            },
            "environment_variables": {
                "BASE_URL": "http://app:8000",
                "TEST_TIMEOUT": "30",
                "PYTHONPATH": "/app",
            },
        }

    def _analyze_test_coverage(
        self,
        scenarios: List[AcceptanceTestScenario],
        source_analysis: SourceCodeAnalysisResult,
    ) -> Dict[str, Any]:
        """Analyze test coverage across different areas."""
        coverage: Dict[str, Any] = {
            "total_scenarios": len(scenarios),
            "categories_covered": len(set(scenario.category for scenario in scenarios)),
            "api_endpoints_covered": 0,
            "external_services_covered": 0,
            "configuration_covered": 0,
            "error_scenarios": 0,
        }

        # Count coverage by category
        category_counts: Dict[str, int] = {}
        for scenario in scenarios:
            category_counts[scenario.category] = (
                category_counts.get(scenario.category, 0) + 1
            )

        coverage["category_breakdown"] = category_counts

        # Estimate coverage percentages
        total_endpoints = len(source_analysis.api_endpoints)
        total_services = len(source_analysis.external_service_integrations)
        total_config = len(source_analysis.configuration_usage)

        if total_endpoints > 0:
            coverage["api_coverage_percentage"] = min(
                100, (category_counts.get("http_api", 0) / total_endpoints) * 100
            )

        if total_services > 0:
            coverage["service_coverage_percentage"] = min(
                100, (category_counts.get("network", 0) / total_services) * 100
            )

        if total_config > 0:
            coverage["config_coverage_percentage"] = min(
                100, (category_counts.get("config", 0) / total_config) * 100
            )

        return coverage

    def _calculate_generation_confidence(
        self,
        scenarios: List[AcceptanceTestScenario],
        doc_analysis: DocumentationAnalysisResult,
        source_analysis: SourceCodeAnalysisResult,
    ) -> float:
        """Calculate confidence score for test generation."""
        confidence_factors = []

        # Base confidence on number of scenarios generated
        if len(scenarios) >= 10:
            confidence_factors.append(0.25)
        elif len(scenarios) >= 5:
            confidence_factors.append(0.15)
        elif len(scenarios) >= 1:
            confidence_factors.append(0.1)

        # Confidence based on API endpoint coverage
        if source_analysis.api_endpoints and any(
            s.category == "http_api" for s in scenarios
        ):
            confidence_factors.append(0.2)

        # Confidence based on service integration coverage
        if source_analysis.external_service_integrations and any(
            s.category == "network" for s in scenarios
        ):
            confidence_factors.append(0.15)

        # Confidence based on configuration coverage
        if source_analysis.configuration_usage and any(
            s.category == "config" for s in scenarios
        ):
            confidence_factors.append(0.15)

        # Confidence based on Docker setup
        if doc_analysis.docker_services and any(
            s.category == "docker" for s in scenarios
        ):
            confidence_factors.append(0.1)

        # Confidence based on error handling
        if any(s.category == "error" for s in scenarios):
            confidence_factors.append(0.1)

        # Confidence based on framework detection
        if source_analysis.framework_detected != "unknown":
            confidence_factors.append(0.05)

        return sum(confidence_factors)

    async def export_test_suites(
        self, result: AcceptanceTestGenerationResult, output_dir: Optional[str] = None
    ) -> str:
        """Export generated test suites to files.

        Args:
            result: Test generation results to export
            output_dir: Optional directory for output files

        Returns:
            Path to the output directory
        """
        if output_dir is None:
            output_dir = str(self.project_root / "acceptance_tests")

        # Create output directory
        output_path = Path(output_dir)

        try:
            # Export each test suite
            for suite in result.test_suites:
                if suite.test_file_content:
                    test_file_path = output_path / f"{suite.name}.py"
                    await self._write_file(str(test_file_path), suite.test_file_content)

                if suite.docker_compose_content:
                    compose_file_path = output_path / "docker-compose.test.yml"
                    await self._write_file(
                        str(compose_file_path), suite.docker_compose_content
                    )

                if suite.dockerfile_content:
                    dockerfile_path = output_path / "Dockerfile.test"
                    await self._write_file(
                        str(dockerfile_path), suite.dockerfile_content
                    )

                if suite.requirements_content:
                    requirements_path = output_path / "requirements-test.txt"
                    await self._write_file(
                        str(requirements_path), suite.requirements_content
                    )

            # Export test generation metadata
            metadata = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_suites": len(result.test_suites),
                "total_scenarios": len(result.test_scenarios),
                "coverage_analysis": result.coverage_analysis,
                "generation_confidence": result.generation_confidence,
                "test_dependencies": result.test_dependencies,
                "docker_configuration": result.docker_configuration,
            }

            metadata_path = output_path / "test_generation_metadata.json"
            await self._write_file(str(metadata_path), json.dumps(metadata, indent=2))

            self.logger.info(f"Exported acceptance tests to {output_dir}")

        except Exception as e:
            self.logger.error(f"Error exporting test suites: {e}")

        return str(output_path)

    async def _write_file(self, file_path: str, content: str) -> None:
        """Write content to a file using MCP server or fallback."""
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
