"""API mapping analysis for library migration."""

import logging
import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Optional RAG dependencies - import only if available
try:
    from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    # Create placeholder classes for type checking
    class OpenAIEmbeddings: # type: ignore
        pass
    class FAISS: # type: ignore
        pass 
    class RecursiveCharacterTextSplitter: # type: ignore
        pass
    class Document: # type: ignore
        pass

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


class APIMigrationGuide(BaseModel):
    """Migration guide for a specific API."""

    source_api: str = Field(description="Original API from source library")
    target_api: str = Field(description="Equivalent API in target library")
    confidence: str = Field(
        description="Confidence level: HIGH, MEDIUM, LOW, or NO_EQUIVALENT"
    )
    source_example: Optional[str] = Field(
        default=None, description="Code example showing how the source API is used"
    )
    target_example: Optional[str] = Field(
        default=None,
        description="Code example showing equivalent usage in target library",
    )
    migration_notes: Optional[str] = Field(
        default=None,
        description="Important considerations, parameter changes, or behavior differences",
    )
    additional_imports: Optional[str] = Field(
        default=None, description="Additional imports needed for the target library"
    )


class APIMappingsResult(BaseModel):
    """Structured output model for comprehensive API migration analysis."""

    migration_guides: list[APIMigrationGuide] = Field(
        description="Detailed migration guides for each API"
    )
    general_migration_notes: Optional[str] = Field(
        default=None,
        description="General notes about migrating between these libraries",
    )
    compatibility_issues: Optional[str] = Field(
        default=None,
        description="Known compatibility issues or breaking changes to be aware of",
    )


class DocumentationRAG:
    """RAG system for semantic documentation search."""

    def __init__(self, llm_config: LLMConfig):
        """Initialize RAG system."""
        self.llm_config = llm_config
        self.embeddings = None
        self.vector_stores: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.rag_enabled = RAG_AVAILABLE

        if self.rag_enabled:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""],
            )
        else:
            self.logger.warning(
                "RAG dependencies not available - using fallback text search"
            )
            self.text_splitter = None  # type: ignore

    def _get_embeddings(self):
        """Get embeddings model (lazy initialization)."""
        if not self.rag_enabled:
            return None

        if self.embeddings is None:
            # Use OpenAI embeddings for now - could be made configurable
            self.embeddings = OpenAIEmbeddings()  # type: ignore
        return self.embeddings

    def index_documentation(self, library_name: str, content: str) -> None:
        """Index documentation content for semantic search."""
        if not self.rag_enabled:
            # Store content for simple text search fallback
            self.vector_stores[library_name] = content
            return

        try:
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)

            # Create documents
            documents = [
                Document(  # type: ignore
                    page_content=chunk,
                    metadata={"library": library_name, "chunk_id": i},
                )
                for i, chunk in enumerate(chunks)
            ]

            # Create vector store
            embeddings = self._get_embeddings()
            vector_store = FAISS.from_documents(documents, embeddings)  # type: ignore
            self.vector_stores[library_name] = vector_store

            self.logger.info(f"Indexed {len(documents)} chunks for {library_name}")

        except Exception as e:
            self.logger.error(f"Failed to index documentation for {library_name}: {e}")
            # Fallback to simple storage
            self.vector_stores[library_name] = content

    def search_documentation(
        self, library_name: str, query: str, top_k: int = 3
    ) -> List[str]:
        """Search documentation for relevant content."""
        if library_name not in self.vector_stores:
            return []

        if not self.rag_enabled:
            # Simple text search fallback
            content = self.vector_stores[library_name]
            if query.lower() in content.lower():
                # Return relevant chunks (simple approach)
                lines = content.split("\n")
                relevant_lines = [
                    line for line in lines if query.lower() in line.lower()
                ][: top_k * 5]
                return [
                    "\n".join(relevant_lines[i : i + 5])
                    for i in range(0, len(relevant_lines), 5)
                ][:top_k]
            return []

        try:
            vector_store = self.vector_stores[library_name]
            docs = vector_store.similarity_search(query, k=top_k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            self.logger.error(f"Failed to search documentation for {library_name}: {e}")
            return []


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

        # Initialize RAG system
        self.rag = DocumentationRAG(llm_config)
        self._last_migration_guides = None

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
            source_docs = await self._fetch_enhanced_documentation(self.source_library)
            target_docs = await self._fetch_enhanced_documentation(self.target_library)

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

    def _fetch_additional_documentation(
        self, library_name: str, base_docs_url: str
    ) -> List[str]:
        """Fetch additional documentation pages via HTTP requests."""
        additional_content = []

        try:
            # Common documentation patterns to try
            common_paths = [
                "/api/",
                "/reference/",
                "/tutorial/",
                "/quickstart/",
                "/examples/",
                "/migration/",
                "/changelog/",
                "/guide/",
            ]

            base_url = base_docs_url.rstrip("/")
            session = requests.Session()
            session.headers.update(
                {"User-Agent": "DiversifierBot/1.0 (Library Migration Tool)"}
            )

            for path in common_paths:
                try:
                    url = f"{base_url}{path}"
                    response = session.get(url, timeout=10)
                    if response.status_code == 200:
                        # Simple HTML text extraction
                        text_content = re.sub(r"<[^>]+>", " ", response.text)
                        text_content = re.sub(r"\s+", " ", text_content).strip()
                        if len(text_content) > 500:  # Only keep substantial content
                            additional_content.append(text_content[:5000])  # Limit size
                            self.logger.info(f"Fetched additional docs from {url}")

                except requests.RequestException as e:
                    self.logger.debug(f"Failed to fetch {url}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error fetching additional documentation: {e}")

        return additional_content

    async def _fetch_enhanced_documentation(
        self, library_name: str
    ) -> Optional[DocumentationSource]:
        """Fetch comprehensive documentation using multiple sources."""
        # First get base documentation via MCP
        base_docs = await self._fetch_library_documentation(library_name)
        if not base_docs:
            return None

        # Fetch additional documentation via HTTP
        additional_content = []
        if base_docs.docs_url:
            additional_content = self._fetch_additional_documentation(
                library_name, base_docs.docs_url
            )

        # Combine all content
        full_content = base_docs.content
        if additional_content:
            full_content += "\n\n--- ADDITIONAL DOCUMENTATION ---\n\n"
            full_content += "\n\n".join(additional_content)

        # Index in RAG system for semantic search
        self.rag.index_documentation(library_name, full_content)

        return DocumentationSource(
            library_name=library_name, docs_url=base_docs.docs_url, content=full_content
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
            structured_model = model.with_structured_output(APIMappingsResult)

            # Load prompts from files
            system_prompt = self._load_prompt_template("api_mapping_system.txt")
            human_prompt = self._load_prompt_template("api_mapping_human.txt")

            # Use RAG to get relevant documentation for the specific APIs
            apis_list = list(used_apis)[
                :20
            ]  # Limit to first 20 APIs to avoid token limits

            # Get relevant documentation chunks for each API via RAG search
            source_relevant_docs = []
            target_relevant_docs = []

            for api in apis_list:
                # Search for API-specific documentation
                source_chunks = self.rag.search_documentation(
                    self.source_library, f"{api} function method usage example", top_k=2
                )
                target_chunks = self.rag.search_documentation(
                    self.target_library,
                    f"{api} function method usage example equivalent",
                    top_k=2,
                )
                source_relevant_docs.extend(source_chunks)
                target_relevant_docs.extend(target_chunks)

            # Combine and deduplicate relevant documentation
            source_docs_text = "\n\n".join(list(dict.fromkeys(source_relevant_docs)))[
                :3000
            ]
            target_docs_text = "\n\n".join(list(dict.fromkeys(target_relevant_docs)))[
                :3000
            ]

            if not source_docs_text and source_docs:
                source_docs_text = source_docs.content[:2000]
            if not target_docs_text and target_docs:
                target_docs_text = target_docs.content[:2000]

            # Format system message with RAG-retrieved documentation
            system_message = SystemMessage(
                content=system_prompt.format(
                    source_library=self.source_library,
                    target_library=self.target_library,
                    source_docs=source_docs_text or "No documentation available",
                    target_docs=target_docs_text or "No documentation available",
                )
            )

            # Format human message
            human_message = HumanMessage(
                content=human_prompt.format(
                    source_library=self.source_library,
                    target_library=self.target_library,
                    apis_list=apis_list,
                )
            )

            # Get structured response from LLM
            structured_response = structured_model.invoke(
                [system_message, human_message]
            )

            # Extract simple mappings for backward compatibility
            # TODO: Update callers to use full migration guides
            validated_mappings = {}
            for guide in structured_response.migration_guides:
                if (
                    guide.source_api in used_apis
                    and guide.target_api != "NO_EQUIVALENT"
                ):
                    validated_mappings[guide.source_api] = guide.target_api

            self.logger.info(f"Successfully mapped {len(validated_mappings)} APIs")
            # Store detailed migration guides for future use
            self._last_migration_guides = structured_response
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
