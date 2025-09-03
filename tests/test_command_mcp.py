#!/usr/bin/env python3
"""Tests for Command MCP Server working_directory functionality."""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.mcp_servers.command.server import CommandMCPServer


class TestCommandMCPServer:
    """Test Command MCP Server functionality."""

    def test_init_with_project_root(self):
        """Test server initialization with project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = CommandMCPServer(project_root=tmpdir)
            assert server.project_root == Path(tmpdir).resolve()

    def test_init_without_project_root(self):
        """Test server initialization without project root defaults to cwd."""
        server = CommandMCPServer()
        assert server.project_root == Path.cwd().resolve()

    @pytest.mark.asyncio
    async def test_execute_command_with_working_directory(self):
        """Test command execution with working_directory parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory within the project root
            project_root = Path(tmpdir)
            subdir = project_root / "subproject"
            subdir.mkdir()

            # Create a test file in the subdirectory
            test_file = subdir / "test.txt"
            test_file.write_text("test content")

            server = CommandMCPServer(project_root=str(project_root))

            # Execute command with working_directory set to the subdirectory
            result = await server._execute_command(
                command="ls test.txt",
                working_directory="subproject",
                timeout=30,
                capture_output=True,
            )

            assert len(result) == 1
            content = result[0].text
            assert "test.txt" in content
            assert '"working_directory": "subproject"' in content
            assert '"exit_code": 0' in content
            assert '"success": true' in content

    @pytest.mark.asyncio
    async def test_execute_command_without_working_directory(self):
        """Test command execution without working_directory uses project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create a test file in the project root
            test_file = project_root / "test.txt"
            test_file.write_text("test content")

            server = CommandMCPServer(project_root=str(project_root))

            # Execute command without working_directory
            result = await server._execute_command(
                command="ls test.txt",
                working_directory=None,
                timeout=30,
                capture_output=True,
            )

            assert len(result) == 1
            content = result[0].text
            assert "test.txt" in content
            assert '"working_directory": "."' in content
            assert '"exit_code": 0' in content
            assert '"success": true' in content

    @pytest.mark.asyncio
    async def test_execute_command_invalid_working_directory(self):
        """Test command execution with invalid working_directory raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = CommandMCPServer(project_root=tmpdir)

            # Try to use a non-existent working directory
            with pytest.raises(ValueError, match="Working directory .* does not exist"):
                await server._execute_command(
                    command="ls",
                    working_directory="nonexistent",
                    timeout=30,
                    capture_output=True,
                )

    @pytest.mark.asyncio
    async def test_execute_command_working_directory_outside_project_root(self):
        """Test command execution with working_directory outside project root raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = CommandMCPServer(project_root=tmpdir)

            # Try to use a working directory outside the project root
            with pytest.raises(
                ValueError, match="Path .* is outside project boundaries"
            ):
                await server._execute_command(
                    command="ls",
                    working_directory="../outside",
                    timeout=30,
                    capture_output=True,
                )
