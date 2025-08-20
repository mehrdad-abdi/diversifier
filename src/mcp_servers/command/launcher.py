#!/usr/bin/env python3
"""Launcher for the Command MCP Server."""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def launch_command_server(project_root: Optional[str] = None) -> subprocess.Popen:
    """Launch the Command MCP Server as a subprocess.

    Args:
        project_root: Root directory for command operations

    Returns:
        subprocess.Popen: The running server process
    """
    server_script = Path(__file__).parent / "server.py"

    cmd = [sys.executable, str(server_script)]
    if project_root:
        cmd.append(project_root)

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


if __name__ == "__main__":
    project_root = sys.argv[1] if len(sys.argv) > 1 else None
    process = launch_command_server(project_root)

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
