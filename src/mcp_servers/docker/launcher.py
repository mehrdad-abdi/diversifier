"""Docker MCP Server launcher with subprocess management."""

import argparse
import subprocess
from pathlib import Path
from typing import Optional
from ..base_client import BaseMCPClient


class DockerMCPLauncher(BaseMCPClient):
    """Launcher for Docker MCP Server with stdio communication."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Docker MCP Server launcher.

        Args:
            project_root: Root directory for Docker operations.
                         If None, uses current working directory.
        """
        super().__init__(project_root, "Docker MCP Server")

    def _get_server_script_path(self) -> Path:
        """Get the path to the Docker server script."""
        return Path(__file__).parent / "server.py"

    def start(self) -> subprocess.Popen:
        """Start the Docker MCP Server as a subprocess (legacy method).

        Returns:
            The subprocess.Popen object for the server process.
        """
        if self.start_server() and self.process:
            return self.process
        raise RuntimeError("Failed to start Docker MCP Server")

    def stop(self) -> None:
        """Stop the Docker MCP Server process (legacy method)."""
        self.stop_server()

    def is_running(self) -> bool:
        """Check if the Docker MCP Server process is running."""
        return self.process is not None and self.process.poll() is None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """Main entry point for launching Docker MCP Server."""
    parser = argparse.ArgumentParser(description="Launch Docker MCP Server")
    parser.add_argument("--project-root", help="Root directory for Docker operations")

    args = parser.parse_args()

    launcher = DockerMCPLauncher(project_root=args.project_root)

    try:
        process = launcher.start()
        print(f"Docker MCP Server started with PID: {process.pid}")

        # Keep launcher running until interrupted
        process.wait()

    except KeyboardInterrupt:
        print("\nStopping Docker MCP Server...")
        launcher.stop()
    except Exception as e:
        print(f"Error: {e}")
        launcher.stop()


if __name__ == "__main__":
    main()
