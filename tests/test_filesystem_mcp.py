"""Tests for File System MCP Server."""

import json
import subprocess
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.mcp_servers.filesystem.server import FileSystemMCPServer
from src.mcp_servers.filesystem.launcher import FileSystemMCPClient


class TestFileSystemMCPServer:
    """Test cases for FileSystemMCPServer."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.server = FileSystemMCPServer(str(self.project_root))

        # Create test files
        (self.project_root / "test.py").write_text(
            "import os\nfrom pathlib import Path\n"
        )
        (self.project_root / "subdir").mkdir()
        (self.project_root / "subdir" / "module.py").write_text(
            "import json\nimport requests\n"
        )

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_path_validation_within_bounds(self):
        """Test that paths within project boundaries are validated correctly."""
        valid_path = self.server._validate_path("test.py")
        expected_path = (self.project_root / "test.py").resolve()
        assert valid_path == expected_path

        valid_subdir_path = self.server._validate_path("subdir/module.py")
        expected_subdir_path = (self.project_root / "subdir" / "module.py").resolve()
        assert valid_subdir_path == expected_subdir_path

    def test_path_validation_outside_bounds(self):
        """Test that paths outside project boundaries are rejected."""
        with pytest.raises(ValueError, match="outside project boundaries"):
            self.server._validate_path("../outside.py")

        with pytest.raises(ValueError, match="outside project boundaries"):
            self.server._validate_path("/tmp/absolute.py")

    @pytest.mark.asyncio
    async def test_read_file_success(self):
        """Test successful file reading."""
        result = await self.server._read_file("test.py")
        assert len(result) == 1
        assert "import os" in result[0].text
        assert "from pathlib import Path" in result[0].text

    @pytest.mark.asyncio
    async def test_read_file_not_found(self):
        """Test reading non-existent file."""
        result = await self.server._read_file("nonexistent.py")
        assert len(result) == 1
        assert "File not found" in result[0].text

    @pytest.mark.asyncio
    async def test_write_file_success(self):
        """Test successful file writing."""
        new_content = "# New test file\nimport sys\n"
        result = await self.server._write_file("new_file.py", new_content, False)

        assert len(result) == 1
        assert "Successfully wrote" in result[0].text

        # Verify file was created
        new_file = self.project_root / "new_file.py"
        assert new_file.exists()
        assert new_file.read_text() == new_content

    @pytest.mark.asyncio
    async def test_write_file_with_backup(self):
        """Test file writing with backup creation."""
        original_content = "import os\nfrom pathlib import Path\n"
        new_content = "# Modified content\nimport sys\n"

        result = await self.server._write_file("test.py", new_content, True)

        assert len(result) == 1
        assert "Successfully wrote" in result[0].text
        assert "backup created" in result[0].text

        # Verify original file was modified
        test_file = self.project_root / "test.py"
        assert test_file.read_text() == new_content

        # Verify backup was created
        backup_file = self.project_root / "test.py.bak"
        assert backup_file.exists()
        assert backup_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_list_files_non_recursive(self):
        """Test listing files in directory (non-recursive)."""
        result = await self.server._list_files(".", "*", False)

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert data["recursive"] is False
        assert "test.py" in data["files"]
        assert "subdir/module.py" not in data["files"]

    @pytest.mark.asyncio
    async def test_list_files_recursive(self):
        """Test listing files in directory (recursive)."""
        result = await self.server._list_files(".", "*.py", True)

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert data["recursive"] is True
        assert "test.py" in data["files"]
        assert "subdir/module.py" in data["files"]

    @pytest.mark.asyncio
    async def test_list_files_directory_not_found(self):
        """Test listing files in non-existent directory."""
        result = await self.server._list_files("nonexistent", "*", False)

        assert len(result) == 1
        assert "Directory not found" in result[0].text

    @pytest.mark.asyncio
    async def test_analyze_python_imports_success(self):
        """Test successful Python import analysis."""
        result = await self.server._analyze_python_imports("test.py")

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "direct_imports" in data
        assert "from_imports" in data
        assert "all_modules" in data

        # Check that expected imports are found
        all_modules = data["all_modules"]
        assert "os" in all_modules
        assert "pathlib" in all_modules

    @pytest.mark.asyncio
    async def test_analyze_python_imports_not_python(self):
        """Test analyzing non-Python file."""
        # Create a non-Python file
        (self.project_root / "config.txt").write_text("some config")

        result = await self.server._analyze_python_imports("config.txt")

        assert len(result) == 1
        assert "not a Python file" in result[0].text

    @pytest.mark.asyncio
    async def test_analyze_python_imports_syntax_error(self):
        """Test analyzing Python file with syntax errors."""
        # Create Python file with syntax error
        (self.project_root / "broken.py").write_text(
            "import os\nif True\n  print('missing colon')"
        )

        result = await self.server._analyze_python_imports("broken.py")

        assert len(result) == 1
        assert "Syntax error" in result[0].text

    @pytest.mark.asyncio
    async def test_find_python_files_no_filter(self):
        """Test finding all Python files without filter."""
        result = await self.server._find_python_files()

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert data["total_python_files"] == 2
        file_paths = [f["path"] for f in data["files"]]
        assert "test.py" in file_paths
        assert "subdir/module.py" in file_paths

    @pytest.mark.asyncio
    async def test_find_python_files_with_filter(self):
        """Test finding Python files that import specific library."""
        result = await self.server._find_python_files("requests")

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert data["filter_library"] == "requests"
        assert len(data["matching_files"]) == 1
        assert data["matching_files"][0]["path"] == "subdir/module.py"

    @pytest.mark.asyncio
    async def test_get_project_structure(self):
        """Test getting project structure analysis."""
        result = await self.server._get_project_structure()

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "python_files" in data
        assert "directories" in data
        assert "summary" in data

        assert "test.py" in data["python_files"]
        assert "subdir/module.py" in data["python_files"]
        assert "subdir" in data["directories"]

        summary = data["summary"]
        assert summary["python_files_count"] == 2
        assert summary["directories_count"] == 1


class TestFileSystemMCPClient:
    """Test cases for FileSystemMCPClient."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = self.temp_dir.name

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @patch("subprocess.Popen")
    def test_start_server_success(self, mock_popen):
        """Test successful server startup."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        client = FileSystemMCPClient(self.project_root)
        result = client.start_server()

        assert result is True
        assert client.process == mock_process
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_start_server_failure(self, mock_popen):
        """Test server startup failure."""
        mock_popen.side_effect = Exception("Failed to start")

        client = FileSystemMCPClient(self.project_root)
        result = client.start_server()

        assert result is False
        assert client.process is None

    def test_stop_server(self):
        """Test stopping server process."""
        mock_process = MagicMock()

        client = FileSystemMCPClient(self.project_root)
        client.process = mock_process
        client.stop_server()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        assert client.process is None

    def test_stop_server_timeout(self):
        """Test stopping server with timeout."""
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)

        client = FileSystemMCPClient(self.project_root)
        client.process = mock_process
        client.stop_server()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert client.process is None

    def test_send_request_no_process(self):
        """Test sending request when no process is running."""
        client = FileSystemMCPClient(self.project_root)
        result = client.send_request("test_method")

        assert result is None

    def test_context_manager_success(self):
        """Test context manager with successful server start."""
        with (
            patch.object(
                FileSystemMCPClient, "start_server", return_value=True
            ) as mock_start,
            patch.object(FileSystemMCPClient, "stop_server") as mock_stop,
        ):

            with FileSystemMCPClient(self.project_root) as client:
                assert client is not None

            mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_context_manager_failure(self):
        """Test context manager with server start failure."""
        with patch.object(FileSystemMCPClient, "start_server", return_value=False):
            with pytest.raises(RuntimeError, match="Failed to start"):
                with FileSystemMCPClient(self.project_root):
                    pass


if __name__ == "__main__":
    pytest.main([__file__])
