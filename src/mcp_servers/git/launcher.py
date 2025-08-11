"""Git MCP Server launcher and lifecycle management."""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from ..base_client import BaseMCPClient


class GitMCPClient(BaseMCPClient):
    """Client for managing Git MCP Server lifecycle and communication."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Git MCP client.

        Args:
            project_root: Root directory to constrain Git operations to.
        """
        super().__init__(project_root, "Git MCP Server")

    def _get_server_script_path(self) -> Path:
        """Get the path to the Git server script."""
        return Path(__file__).parent / "server.py"

    def init_repository(self, repo_path: str = ".") -> Optional[Dict[str, Any]]:
        """Initialize a Git repository.

        Args:
            repo_path: Path to initialize repository in

        Returns:
            Operation result
        """
        result = self.call_tool("init_repository", {"repo_path": repo_path})
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def clone_repository(self, url: str, destination: str) -> Optional[Dict[str, Any]]:
        """Clone a Git repository.

        Args:
            url: Repository URL to clone
            destination: Destination directory for clone

        Returns:
            Operation result
        """
        result = self.call_tool(
            "clone_repository", {"url": url, "destination": destination}
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def get_status(self, repo_path: str = ".") -> Optional[Dict[str, Any]]:
        """Get Git repository status.

        Args:
            repo_path: Path to repository

        Returns:
            Repository status
        """
        result = self.call_tool("get_status", {"repo_path": repo_path})
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def create_branch(
        self, branch_name: str, repo_path: str = ".", switch: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Create a new branch.

        Args:
            branch_name: Name of the new branch
            repo_path: Path to repository
            switch: Whether to switch to the new branch

        Returns:
            Operation result
        """
        result = self.call_tool(
            "create_branch",
            {"repo_path": repo_path, "branch_name": branch_name, "switch": switch},
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def switch_branch(
        self, branch_name: str, repo_path: str = "."
    ) -> Optional[Dict[str, Any]]:
        """Switch to an existing branch.

        Args:
            branch_name: Name of the branch to switch to
            repo_path: Path to repository

        Returns:
            Operation result
        """
        result = self.call_tool(
            "switch_branch", {"repo_path": repo_path, "branch_name": branch_name}
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def list_branches(
        self, repo_path: str = ".", include_remote: bool = True
    ) -> Optional[Dict[str, Any]]:
        """List all branches.

        Args:
            repo_path: Path to repository
            include_remote: Include remote branches

        Returns:
            Branch list
        """
        result = self.call_tool(
            "list_branches",
            {"repo_path": repo_path, "include_remote": include_remote},
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def add_files(
        self, file_patterns: List[str], repo_path: str = "."
    ) -> Optional[Dict[str, Any]]:
        """Stage files for commit.

        Args:
            file_patterns: File patterns to add
            repo_path: Path to repository

        Returns:
            Operation result
        """
        result = self.call_tool(
            "add_files", {"repo_path": repo_path, "file_patterns": file_patterns}
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def commit_changes(
        self,
        message: str,
        repo_path: str = ".",
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Commit staged changes.

        Args:
            message: Commit message
            repo_path: Path to repository
            author_name: Author name (optional)
            author_email: Author email (optional)

        Returns:
            Operation result
        """
        args = {"repo_path": repo_path, "message": message}
        if author_name:
            args["author_name"] = author_name
        if author_email:
            args["author_email"] = author_email

        result = self.call_tool("commit_changes", args)
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def get_diff(
        self,
        repo_path: str = ".",
        staged: bool = False,
        commit_range: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get diff of changes.

        Args:
            repo_path: Path to repository
            staged: Get staged changes diff
            commit_range: Commit range (e.g., 'HEAD~1..HEAD')
            file_paths: Specific files to get diff for

        Returns:
            Diff result
        """
        args = {"repo_path": repo_path, "staged": staged}
        if commit_range:
            args["commit_range"] = commit_range
        if file_paths:
            args["file_paths"] = file_paths

        result = self.call_tool("get_diff", args)
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def get_log(
        self,
        repo_path: str = ".",
        max_count: int = 10,
        branch: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get commit history.

        Args:
            repo_path: Path to repository
            max_count: Maximum number of commits to retrieve
            branch: Branch to get log from
            file_path: Get log for specific file

        Returns:
            Commit log
        """
        args = {"repo_path": repo_path, "max_count": max_count}
        if branch:
            args["branch"] = branch
        if file_path:
            args["file_path"] = file_path

        result = self.call_tool("get_log", args)
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def get_changed_files(
        self,
        repo_path: str = ".",
        base_commit: str = "HEAD~1",
        target_commit: str = "HEAD",
    ) -> Optional[Dict[str, Any]]:
        """Get list of changed files between commits.

        Args:
            repo_path: Path to repository
            base_commit: Base commit to compare from
            target_commit: Target commit to compare to

        Returns:
            Changed files list
        """
        result = self.call_tool(
            "get_changed_files",
            {
                "repo_path": repo_path,
                "base_commit": base_commit,
                "target_commit": target_commit,
            },
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def create_temp_branch(
        self,
        repo_path: str = ".",
        base_branch: str = "main",
        prefix: str = "diversifier-temp",
    ) -> Optional[Dict[str, Any]]:
        """Create a temporary branch for migration work.

        Args:
            repo_path: Path to repository
            base_branch: Base branch to create temp branch from
            prefix: Prefix for temp branch name

        Returns:
            Operation result
        """
        result = self.call_tool(
            "create_temp_branch",
            {
                "repo_path": repo_path,
                "base_branch": base_branch,
                "prefix": prefix,
            },
        )
        if result and "result" in result and result["result"]:
            content = result["result"][0].get("text")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass
        return None

    def __enter__(self):
        """Context manager entry."""
        if self.start_server():
            return self
        else:
            raise RuntimeError("Failed to start Git MCP Server")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()


# Example usage
async def example_usage():
    """Example of how to use the Git MCP Client."""

    with GitMCPClient() as client:
        # Get repository status
        status = client.get_status()
        if status:
            print(f"Current branch: {status.get('branch')}")
            print(f"Is dirty: {status.get('is_dirty')}")

        # List branches
        branches = client.list_branches()
        if branches:
            print(f"Local branches: {branches.get('local', [])}")

        # Get recent commit history
        log = client.get_log(max_count=5)
        if log:
            print(f"Recent commits: {len(log.get('commits', []))}")

        # Get diff of changes
        diff = client.get_diff()
        if diff:
            print(f"Diff lines: {diff.get('lines', 0)}")

        # Create a temporary branch
        temp_branch = client.create_temp_branch()
        if temp_branch:
            print(f"Created temp branch: {temp_branch.get('temp_branch')}")


if __name__ == "__main__":
    asyncio.run(example_usage())
