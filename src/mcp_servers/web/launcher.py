"""Web MCP Client using mcp-server-fetch for web content fetching."""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from ..base_client import BaseMCPClient


class WebMCPClient(BaseMCPClient):
    """Client for web content fetching using mcp-server-fetch via uvx."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Web MCP client.

        Args:
            project_root: Root directory (not used for web operations but kept for consistency).
        """
        super().__init__(project_root, "Web MCP Client")

    def _get_server_script_path(self) -> Path:
        """Get path to server script (not used for external MCP server)."""
        return Path(__file__)  # Placeholder, not actually used

    def start_server(self) -> bool:
        """Start the mcp-server-fetch via uvx.

        Returns:
            True if server started successfully, False otherwise.
        """
        try:
            # Check if uvx is available
            if not shutil.which("uvx"):
                print("uvx not found. Please install uv first.")
                return False

            # Start mcp-server-fetch via uvx with stdio communication
            self.process = subprocess.Popen(
                ["uvx", "mcp-server-fetch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )

            return True

        except Exception as e:
            print(f"Failed to start {self.server_name}: {e}")
            return False

    def fetch_url(self, url: str, start_index: int = 0) -> Optional[str]:
        """Fetch content from a URL using the MCP server.

        Args:
            url: URL to fetch
            start_index: Starting index for content extraction (for chunking)

        Returns:
            Fetched content as markdown or None if error
        """
        params: Dict[str, Any] = {"url": url}
        if start_index > 0:
            params["start_index"] = start_index

        result = self.call_tool("fetch", params)
        if result and "result" in result and result["result"]:
            content = result["result"]
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text")
            elif isinstance(content, dict):
                return content.get("text")
        return None

    def get_pypi_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Fetch PyPI information for a Python package.

        Args:
            package_name: Name of the Python package

        Returns:
            PyPI package information or None if error
        """
        pypi_url = f"https://pypi.org/pypi/{package_name}/json"
        content = self.fetch_url(pypi_url)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        return None

    def fetch_documentation(self, docs_url: str) -> Optional[str]:
        """Fetch documentation from a URL.

        Args:
            docs_url: URL to the documentation

        Returns:
            Documentation content as markdown or None if error
        """
        return self.fetch_url(docs_url)

    def __enter__(self):
        """Context manager entry."""
        if self.start_server():
            return self
        else:
            raise RuntimeError("Failed to start Web MCP Client")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()
