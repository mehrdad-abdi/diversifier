"""API mapping analysis for library migration."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from ..mcp_manager import MCPManager, MCPServerType
from ..config import LLMConfig


@dataclass
class DocumentationSource:
    """Documentation source information for a library."""

    library_name: str
    docs_url: str
    content: str


@dataclass
class APIMappingResult:
    """Result of API mapping analysis."""

    source_library: str
    target_library: str
    used_apis: list[str]
    api_mappings: Dict[str, str]
    unmapped_apis: list[str]
    source_docs: Optional[DocumentationSource]
    target_docs: Optional[DocumentationSource]
    success: bool
    error: Optional[str] = None


class APIMapper:
    """Handles API mapping between source and target libraries."""

    def __init__(
        self,
        source_library: str,
        target_library: str,
        llm_config: LLMConfig,
        mcp_manager: MCPManager,
        prompts_dir: Optional[str] = None,
    ):
        """Initialize API mapper.

        Args:
            source_library: Library to migrate from
            target_library: Library to migrate to
            llm_config: LLM configuration
            mcp_manager: MCP manager for web access
            prompts_dir: Directory containing prompt templates
        """
        self.source_library = source_library
        self.target_library = target_library
        self.llm_config = llm_config
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger("diversifier.api_mapper")

        # Set prompts directory
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            # Default to prompts folder in project root
            self.prompts_dir = Path(__file__).parent.parent.parent.parent / "prompts"

    async def find_api_mappings(self, used_apis: Set[str]) -> APIMappingResult:
        """Find API mappings between source and target libraries.

        Args:
            used_apis: Set of API names used in the project

        Returns:
            APIMappingResult with mapping information
        """
        try:
            self.logger.info(
                f"Finding API mappings: {self.source_library} -> {self.target_library}"
            )

            # Check if Web MCP server is available
            if not self.mcp_manager.is_server_available(MCPServerType.WEB):
                return APIMappingResult(
                    source_library=self.source_library,
                    target_library=self.target_library,
                    used_apis=list(used_apis),
                    api_mappings={},
                    unmapped_apis=list(used_apis),
                    source_docs=None,
                    target_docs=None,
                    success=False,
                    error="Web MCP server not available for API mapping",
                )

            if not used_apis:
                return APIMappingResult(
                    source_library=self.source_library,
                    target_library=self.target_library,
                    used_apis=[],
                    api_mappings={},
                    unmapped_apis=[],
                    source_docs=None,
                    target_docs=None,
                    success=True,
                )

            self.logger.info(
                f"Found {len(used_apis)} API usages to map: {list(used_apis)[:5]}..."
            )

            # Get documentation for both libraries
            source_docs = await self._fetch_library_documentation(self.source_library)
            target_docs = await self._fetch_library_documentation(self.target_library)

            # Use LLM to analyze API mappings
            api_mappings = await self._analyze_api_mappings_with_llm(
                used_apis, source_docs, target_docs
            )

            if not api_mappings:
                return APIMappingResult(
                    source_library=self.source_library,
                    target_library=self.target_library,
                    used_apis=list(used_apis),
                    api_mappings={},
                    unmapped_apis=list(used_apis),
                    source_docs=source_docs,
                    target_docs=target_docs,
                    success=False,
                    error=f"Failed to find API mappings between {self.source_library} and {self.target_library}",
                )

            unmapped_apis = [api for api in used_apis if api not in api_mappings]

            return APIMappingResult(
                source_library=self.source_library,
                target_library=self.target_library,
                used_apis=list(used_apis),
                api_mappings=api_mappings,
                unmapped_apis=unmapped_apis,
                source_docs=source_docs,
                target_docs=target_docs,
                success=True,
            )

        except Exception as e:
            self.logger.error(f"API mapping failed: {e}")
            return APIMappingResult(
                source_library=self.source_library,
                target_library=self.target_library,
                used_apis=list(used_apis),
                api_mappings={},
                unmapped_apis=list(used_apis),
                source_docs=None,
                target_docs=None,
                success=False,
                error=f"API mapping failed: {str(e)}",
            )

    async def _fetch_library_documentation(
        self, library_name: str
    ) -> Optional[DocumentationSource]:
        """Fetch documentation for a library.

        Args:
            library_name: Name of the library

        Returns:
            DocumentationSource or None if failed
        """
        try:
            web_connection = self.mcp_manager.get_connection(MCPServerType.WEB)
            if not web_connection:
                self.logger.warning(
                    "No web connection available for documentation fetching"
                )
                return None

            # Fetch PyPI information
            pypi_info = web_connection.call_tool(
                "get_pypi_info", {"package_name": library_name}
            )
            docs_url = self._extract_docs_url(pypi_info, library_name)

            if not docs_url:
                self.logger.warning(f"No documentation URL found for {library_name}")
                return DocumentationSource(
                    library_name=library_name,
                    docs_url="",
                    content="No documentation available",
                )

            # Fetch documentation content
            docs_result = web_connection.call_tool(
                "fetch_documentation", {"docs_url": docs_url}
            )
            docs_content = ""

            if docs_result and isinstance(docs_result, dict):
                result_data = docs_result.get("result", [])
                if (
                    result_data
                    and isinstance(result_data, list)
                    and len(result_data) > 0
                ):
                    docs_content = result_data[0].get("text", "")
                    if len(docs_content) > 10000:  # Limit content size
                        docs_content = docs_content[:10000] + "...[truncated]"

            return DocumentationSource(
                library_name=library_name,
                docs_url=docs_url,
                content=docs_content or "No documentation content available",
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch documentation for {library_name}: {e}")
            return None

    def _extract_docs_url(
        self, pypi_info: Optional[Dict[str, Any]], library_name: str
    ) -> str:
        """Extract documentation URL from PyPI information.

        Args:
            pypi_info: PyPI package information
            library_name: Name of the library

        Returns:
            Documentation URL or empty string
        """
        if not pypi_info or "info" not in pypi_info:
            return ""

        info = pypi_info["info"]

        # Try different fields for documentation URL
        for field in ["docs_url", "home_page", "project_urls"]:
            if field in info and info[field]:
                if field == "project_urls" and isinstance(info[field], dict):
                    # Look for documentation-related URLs
                    for key, url in info[field].items():
                        if any(
                            doc_keyword in key.lower()
                            for doc_keyword in ["doc", "manual", "guide"]
                        ):
                            return url
                    # Fallback to homepage
                    if "homepage" in info[field]:
                        return info[field]["homepage"]
                else:
                    return info[field]

        # Fallback to common documentation patterns
        return f"https://{library_name}.readthedocs.io/"

    async def _analyze_api_mappings_with_llm(
        self,
        used_apis: Set[str],
        source_docs: Optional[DocumentationSource],
        target_docs: Optional[DocumentationSource],
    ) -> Dict[str, str]:
        """Use LLM to analyze and map APIs between libraries.

        Args:
            used_apis: Set of APIs to map
            source_docs: Source library documentation
            target_docs: Target library documentation

        Returns:
            Dictionary mapping source APIs to target APIs
        """
        try:
            # Initialize chat model
            model_id = (
                f"{self.llm_config.provider.lower()}:{self.llm_config.model_name}"
            )
            kwargs: Dict[str, Any] = {
                "temperature": 0.1
            }  # Low temperature for consistent results
            kwargs.update(self.llm_config.additional_params)

            model = init_chat_model(model=model_id, **kwargs)

            # Load prompts from files
            system_prompt = self._load_prompt_template("api_mapping_system.txt")
            human_prompt = self._load_prompt_template("api_mapping_human.txt")

            # Format system message
            system_message = SystemMessage(
                content=system_prompt.format(
                    source_library=self.source_library,
                    target_library=self.target_library,
                    source_docs=(
                        source_docs.content[:2000]
                        if source_docs and source_docs.content
                        else "No documentation available"
                    ),
                    target_docs=(
                        target_docs.content[:2000]
                        if target_docs and target_docs.content
                        else "No documentation available"
                    ),
                )
            )

            # Format human message
            apis_list = list(used_apis)[
                :20
            ]  # Limit to first 20 APIs to avoid token limits
            human_message = HumanMessage(
                content=human_prompt.format(
                    source_library=self.source_library,
                    target_library=self.target_library,
                    apis_list=apis_list,
                )
            )

            # Get response from LLM
            response = model.invoke([system_message, human_message])
            response_text = response.content.strip()

            # Parse JSON response
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            api_mappings = json.loads(response_text)

            # Validate mappings
            validated_mappings = {}
            for source_api, target_api in api_mappings.items():
                if source_api in used_apis:
                    validated_mappings[source_api] = target_api

            self.logger.info(f"Successfully mapped {len(validated_mappings)} APIs")
            return validated_mappings

        except Exception as e:
            self.logger.error(f"LLM API mapping analysis failed: {e}")
            return {}

    def _load_prompt_template(self, filename: str) -> str:
        """Load prompt template from file.

        Args:
            filename: Name of the prompt file

        Returns:
            Prompt template content
        """
        try:
            prompt_file = self.prompts_dir / filename
            if prompt_file.exists():
                return prompt_file.read_text()
            else:
                self.logger.warning(f"Prompt file not found: {prompt_file}")
                return ""
        except Exception as e:
            self.logger.error(f"Failed to load prompt template {filename}: {e}")
            return ""
