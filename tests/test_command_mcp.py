#!/usr/bin/env python3
"""Tests for Command MCP Server working_directory functionality."""

import tempfile
import pytest
from pathlib import Path

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

    @pytest.mark.asyncio
    async def test_execute_pwd_command_with_working_directory(self):
        """Test pwd command to verify actual working directory is set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            subdir = project_root / "subproject"
            subdir.mkdir()

            server = CommandMCPServer(project_root=str(project_root))

            # Test pwd without working_directory (should be project_root)
            result = await server._execute_command(
                command="pwd", working_directory=None, timeout=30, capture_output=True
            )

            assert len(result) == 1
            content = result[0].text
            assert str(project_root) in content
            assert '"working_directory": "."' in content
            assert '"exit_code": 0' in content

            # Test pwd with working_directory set to subdirectory
            result = await server._execute_command(
                command="pwd",
                working_directory="subproject",
                timeout=30,
                capture_output=True,
            )

            assert len(result) == 1
            content = result[0].text
            assert str(subdir) in content
            assert '"working_directory": "subproject"' in content
            assert '"exit_code": 0' in content

    @pytest.mark.asyncio
    async def test_execute_pwd_command_different_project_roots(self):
        """Test pwd command with different project root scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a nested directory structure
            base_dir = Path(tmpdir)
            project_dir = base_dir / "project"
            project_dir.mkdir()

            # Test: Server initialized with project_dir as root
            server = CommandMCPServer(project_root=str(project_dir))
            result = await server._execute_command(
                command="pwd", working_directory=None, timeout=30, capture_output=True
            )

            assert len(result) == 1
            content = result[0].text
            # pwd output should show the actual project directory
            assert str(project_dir) in content
            assert '"working_directory": "."' in content

    @pytest.mark.asyncio
    async def test_llm_test_runner_scenario(self):
        """Test scenario that mimics LLM test runner behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate the scenario:
            # - diversifier project is in tmpdir/diversifier
            # - target project is in tmpdir/pilot-project
            base_dir = Path(tmpdir)
            diversifier_dir = base_dir / "diversifier"
            target_project_dir = base_dir / "pilot-project"

            diversifier_dir.mkdir()
            target_project_dir.mkdir()

            # Create a marker file in target project to identify it
            (target_project_dir / "target_marker.txt").write_text("target project")

            # Initialize command server with target project as root (like coordinator does)
            server = CommandMCPServer(project_root=str(target_project_dir))

            # Test 1: Without working_directory (original LLM test runner behavior)
            result_original = await server._execute_command(
                command="pwd && ls -la",
                working_directory=None,
                timeout=30,
                capture_output=True,
            )

            # Test 2: With working_directory="." (my fix)
            result_with_fix = await server._execute_command(
                command="pwd && ls -la",
                working_directory=".",
                timeout=30,
                capture_output=True,
            )

            # Both should execute in the target project directory
            assert len(result_original) == 1
            assert len(result_with_fix) == 1

            original_output = result_original[0].text
            fix_output = result_with_fix[0].text

            # Both outputs should contain the target project path
            assert str(target_project_dir) in original_output
            assert str(target_project_dir) in fix_output

            # Both should see the target_marker.txt file
            assert "target_marker.txt" in original_output
            assert "target_marker.txt" in fix_output

            # Both should show successful execution
            assert '"exit_code": 0' in original_output
            assert '"exit_code": 0' in fix_output

            # Working directory should be reported correctly
            assert '"working_directory": "."' in original_output
            assert '"working_directory": "."' in fix_output

    @pytest.mark.asyncio
    async def test_mcp_server_wrong_project_root_scenario(self):
        """Test what happens when MCP server is initialized with wrong project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            diversifier_dir = base_dir / "diversifier"
            target_project_dir = base_dir / "pilot-project"

            diversifier_dir.mkdir()
            target_project_dir.mkdir()

            # Create marker files to identify directories
            (diversifier_dir / "diversifier_marker.txt").write_text(
                "diversifier project"
            )
            (target_project_dir / "target_marker.txt").write_text("target project")

            # WRONG: Initialize command server with diversifier directory as root
            # This simulates the bug scenario
            server = CommandMCPServer(project_root=str(diversifier_dir))

            # Test 1: Without working_directory - should run in diversifier directory
            result = await server._execute_command(
                command="pwd && ls -la",
                working_directory=None,
                timeout=30,
                capture_output=True,
            )

            assert len(result) == 1
            output = result[0].text

            # Should be in diversifier directory, not target project
            assert str(diversifier_dir) in output
            assert "diversifier_marker.txt" in output
            assert (
                "target_marker.txt" not in output
            )  # Should NOT see target project files

            # Test 2: Try to access target project with relative path
            try:
                # Calculate relative path from diversifier to target project
                rel_path = target_project_dir.relative_to(diversifier_dir)
                result2 = await server._execute_command(
                    command="pwd && ls -la",
                    working_directory=str(rel_path),
                    timeout=30,
                    capture_output=True,
                )

                assert len(result2) == 1
                output2 = result2[0].text

                # Should now be in target project directory
                assert str(target_project_dir) in output2
                assert "target_marker.txt" in output2

            except (ValueError, OSError):
                # This might fail if target project is outside the MCP server's boundaries
                pass
