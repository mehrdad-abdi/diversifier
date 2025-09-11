"""Tests for Web MCP Client."""

import json
import pytest
from unittest.mock import patch, MagicMock, ANY
from src.mcp_servers.web.launcher import WebMCPClient


class TestWebMCPClient:
    """Test cases for WebMCPClient."""

    def setup_method(self):
        """Set up test environment."""
        self.client = WebMCPClient()

    def test_init(self):
        """Test WebMCPClient initialization."""
        assert self.client.server_name == "Web MCP Client"
        assert self.client.project_root is not None
        assert self.client.process is None
        assert self.client.initialized is False

    @patch("src.mcp_servers.web.launcher.shutil.which")
    @patch("src.mcp_servers.web.launcher.subprocess.Popen")
    def test_start_server_success(self, mock_popen, mock_which):
        """Test successful server startup."""
        # Mock uvx availability
        mock_which.return_value = "/path/to/uvx"

        # Mock successful process creation
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        result = self.client.start_server()

        assert result is True
        assert self.client.process == mock_process
        mock_popen.assert_called_once_with(
            ["uvx", "mcp-server-fetch"],
            stdin=ANY,
            stdout=ANY,
            stderr=ANY,
            text=True,
            bufsize=0,
        )

    @patch("src.mcp_servers.web.launcher.shutil.which")
    def test_start_server_uvx_not_found(self, mock_which):
        """Test server startup when uvx is not available."""
        mock_which.return_value = None

        result = self.client.start_server()

        assert result is False
        assert self.client.process is None

    @patch("src.mcp_servers.web.launcher.shutil.which")
    @patch("src.mcp_servers.web.launcher.subprocess.Popen")
    def test_start_server_exception(self, mock_popen, mock_which):
        """Test server startup when subprocess creation fails."""
        mock_which.return_value = "/path/to/uvx"
        mock_popen.side_effect = Exception("Process creation failed")

        result = self.client.start_server()

        assert result is False
        assert self.client.process is None

    @patch.object(WebMCPClient, "call_tool")
    def test_fetch_url(self, mock_call_tool):
        """Test URL fetching functionality."""
        # Mock successful tool call
        mock_call_tool.return_value = {
            "result": [{"text": "# Sample Content\n\nThis is test content."}]
        }

        result = self.client.fetch_url("https://example.com")

        assert result == "# Sample Content\n\nThis is test content."
        mock_call_tool.assert_called_once_with("fetch", {"url": "https://example.com"})

    @patch.object(WebMCPClient, "call_tool")
    def test_fetch_url_with_start_index(self, mock_call_tool):
        """Test URL fetching with start index."""
        mock_call_tool.return_value = {"result": [{"text": "Partial content..."}]}

        result = self.client.fetch_url("https://example.com", start_index=1000)

        assert result == "Partial content..."
        mock_call_tool.assert_called_once_with(
            "fetch", {"url": "https://example.com", "start_index": 1000}
        )

    @patch.object(WebMCPClient, "call_tool")
    def test_fetch_url_failure(self, mock_call_tool):
        """Test URL fetching failure."""
        mock_call_tool.return_value = None

        result = self.client.fetch_url("https://invalid.url")

        assert result is None

    @patch.object(WebMCPClient, "fetch_url")
    def test_get_pypi_info(self, mock_fetch_url):
        """Test PyPI information fetching."""
        # Mock PyPI JSON response
        mock_pypi_data = {
            "info": {
                "name": "requests",
                "version": "2.31.0",
                "summary": "Python HTTP for Humans.",
            }
        }
        mock_fetch_url.return_value = json.dumps(mock_pypi_data)

        result = self.client.get_pypi_info("requests")

        assert result == mock_pypi_data
        mock_fetch_url.assert_called_once_with("https://pypi.org/pypi/requests/json")

    @patch.object(WebMCPClient, "fetch_url")
    def test_get_pypi_info_invalid_json(self, mock_fetch_url):
        """Test PyPI info with invalid JSON response."""
        mock_fetch_url.return_value = "Invalid JSON content"

        result = self.client.get_pypi_info("nonexistent")

        assert result is None

    @patch.object(WebMCPClient, "fetch_url")
    def test_fetch_documentation(self, mock_fetch_url):
        """Test documentation fetching."""
        mock_fetch_url.return_value = "# Documentation\n\nThis is documentation."

        result = self.client.fetch_documentation("https://docs.example.com")

        assert result == "# Documentation\n\nThis is documentation."
        mock_fetch_url.assert_called_once_with("https://docs.example.com")

    @patch.object(WebMCPClient, "start_server")
    def test_context_manager_success(self, mock_start_server):
        """Test context manager with successful server startup."""
        mock_start_server.return_value = True

        with self.client as client:
            assert client == self.client
            mock_start_server.assert_called_once()

    @patch.object(WebMCPClient, "start_server")
    def test_context_manager_failure(self, mock_start_server):
        """Test context manager with failed server startup."""
        mock_start_server.return_value = False

        with pytest.raises(RuntimeError, match="Failed to start Web MCP Client"):
            with self.client:
                pass


class TestWebMCPClientIntegration:
    """Integration tests for WebMCPClient."""

    def setup_method(self):
        """Set up integration test environment."""
        self.client = WebMCPClient()

    @pytest.mark.skipif(True, reason="Integration test requiring external network")
    def test_real_pypi_request(self):
        """Test real PyPI API request."""
        # This test would require actual network access and uvx installation
        # Skip by default but can be enabled for manual testing
        with self.client:
            pypi_info = self.client.get_pypi_info("requests")
            assert pypi_info is not None
            assert "info" in pypi_info
            assert pypi_info["info"]["name"] == "requests"
