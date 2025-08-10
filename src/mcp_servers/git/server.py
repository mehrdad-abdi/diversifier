#!/usr/bin/env python3
"""Git MCP Server with stdio transport."""

import json
from pathlib import Path

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    TextContent,
    Tool,
)

import git


class GitMCPServer:
    """MCP Server for Git operations with security constraints."""

    def __init__(self, project_root: str | None = None):
        """Initialize the Git MCP Server.

        Args:
            project_root: Root directory to constrain Git operations to.
                         If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.project_root = self.project_root.resolve()

        # Initialize MCP server
        self.server = Server("git-server")

        # Register tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="init_repository",
                    description="Initialize a Git repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to initialize repository in",
                            }
                        },
                        "required": ["repo_path"],
                    },
                ),
                Tool(
                    name="clone_repository",
                    description="Clone a Git repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Repository URL to clone",
                            },
                            "destination": {
                                "type": "string",
                                "description": "Destination directory for clone",
                            },
                        },
                        "required": ["url", "destination"],
                    },
                ),
                Tool(
                    name="get_status",
                    description="Get Git repository status",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            }
                        },
                    },
                ),
                Tool(
                    name="create_branch",
                    description="Create and optionally switch to a new branch",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "branch_name": {
                                "type": "string",
                                "description": "Name of the new branch",
                            },
                            "switch": {
                                "type": "boolean",
                                "description": "Whether to switch to the new branch",
                                "default": True,
                            },
                        },
                        "required": ["branch_name"],
                    },
                ),
                Tool(
                    name="switch_branch",
                    description="Switch to an existing branch",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "branch_name": {
                                "type": "string",
                                "description": "Name of the branch to switch to",
                            },
                        },
                        "required": ["branch_name"],
                    },
                ),
                Tool(
                    name="list_branches",
                    description="List all branches in the repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "include_remote": {
                                "type": "boolean",
                                "description": "Include remote branches",
                                "default": True,
                            },
                        },
                    },
                ),
                Tool(
                    name="add_files",
                    description="Stage files for commit",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "file_patterns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "File patterns to add (e.g., ['*.py', 'src/'])",
                            },
                        },
                        "required": ["file_patterns"],
                    },
                ),
                Tool(
                    name="commit_changes",
                    description="Commit staged changes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "message": {
                                "type": "string",
                                "description": "Commit message",
                            },
                            "author_name": {
                                "type": "string",
                                "description": "Author name (optional)",
                            },
                            "author_email": {
                                "type": "string",
                                "description": "Author email (optional)",
                            },
                        },
                        "required": ["message"],
                    },
                ),
                Tool(
                    name="get_diff",
                    description="Get diff of changes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "staged": {
                                "type": "boolean",
                                "description": "Get staged changes diff",
                                "default": False,
                            },
                            "commit_range": {
                                "type": "string",
                                "description": "Commit range (e.g., 'HEAD~1..HEAD')",
                            },
                            "file_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific files to get diff for",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_log",
                    description="Get commit history",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "max_count": {
                                "type": "integer",
                                "description": "Maximum number of commits to retrieve",
                                "default": 10,
                            },
                            "branch": {
                                "type": "string",
                                "description": "Branch to get log from",
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Get log for specific file",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_changed_files",
                    description="Get list of changed files between commits",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "base_commit": {
                                "type": "string",
                                "description": "Base commit to compare from",
                                "default": "HEAD~1",
                            },
                            "target_commit": {
                                "type": "string",
                                "description": "Target commit to compare to",
                                "default": "HEAD",
                            },
                        },
                    },
                ),
                Tool(
                    name="create_temp_branch",
                    description="Create a temporary branch for migration work",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to repository",
                                "default": ".",
                            },
                            "base_branch": {
                                "type": "string",
                                "description": "Base branch to create temp branch from",
                                "default": "main",
                            },
                            "prefix": {
                                "type": "string",
                                "description": "Prefix for temp branch name",
                                "default": "diversifier-temp",
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            try:
                if name == "init_repository":
                    return await self._init_repository(arguments["repo_path"])
                elif name == "clone_repository":
                    return await self._clone_repository(
                        arguments["url"], arguments["destination"]
                    )
                elif name == "get_status":
                    return await self._get_status(arguments.get("repo_path", "."))
                elif name == "create_branch":
                    return await self._create_branch(
                        arguments.get("repo_path", "."),
                        arguments["branch_name"],
                        arguments.get("switch", True),
                    )
                elif name == "switch_branch":
                    return await self._switch_branch(
                        arguments.get("repo_path", "."), arguments["branch_name"]
                    )
                elif name == "list_branches":
                    return await self._list_branches(
                        arguments.get("repo_path", "."),
                        arguments.get("include_remote", True),
                    )
                elif name == "add_files":
                    return await self._add_files(
                        arguments.get("repo_path", "."), arguments["file_patterns"]
                    )
                elif name == "commit_changes":
                    return await self._commit_changes(
                        arguments.get("repo_path", "."),
                        arguments["message"],
                        arguments.get("author_name"),
                        arguments.get("author_email"),
                    )
                elif name == "get_diff":
                    return await self._get_diff(
                        arguments.get("repo_path", "."),
                        arguments.get("staged", False),
                        arguments.get("commit_range"),
                        arguments.get("file_paths"),
                    )
                elif name == "get_log":
                    return await self._get_log(
                        arguments.get("repo_path", "."),
                        arguments.get("max_count", 10),
                        arguments.get("branch"),
                        arguments.get("file_path"),
                    )
                elif name == "get_changed_files":
                    return await self._get_changed_files(
                        arguments.get("repo_path", "."),
                        arguments.get("base_commit", "HEAD~1"),
                        arguments.get("target_commit", "HEAD"),
                    )
                elif name == "create_temp_branch":
                    return await self._create_temp_branch(
                        arguments.get("repo_path", "."),
                        arguments.get("base_branch", "main"),
                        arguments.get("prefix", "diversifier-temp"),
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _validate_path(self, repo_path: str) -> Path:
        """Validate that path is within project boundaries.

        Args:
            repo_path: Path to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is outside project boundaries
        """
        path = Path(repo_path)
        if not path.is_absolute():
            path = self.project_root / path

        path = path.resolve()

        # Ensure path is within project root
        if not str(path).startswith(str(self.project_root)):
            raise ValueError(f"Path {path} is outside project boundaries")

        return path

    def _get_repo(self, repo_path: str) -> git.Repo:
        """Get Git repository object.

        Args:
            repo_path: Path to repository

        Returns:
            GitPython Repo object

        Raises:
            ValueError: If not a valid Git repository
        """
        path = self._validate_path(repo_path)

        try:
            return git.Repo(path)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"Not a Git repository: {path}")

    async def _init_repository(self, repo_path: str) -> list[TextContent]:
        """Initialize a Git repository."""
        path = self._validate_path(repo_path)

        try:
            git.Repo.init(path)
            result = {
                "status": "success",
                "path": str(path),
                "message": f"Initialized Git repository at {path}",
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [
                TextContent(type="text", text=f"Error initializing repository: {e}")
            ]

    async def _clone_repository(self, url: str, destination: str) -> list[TextContent]:
        """Clone a Git repository."""
        dest_path = self._validate_path(destination)

        try:
            repo = git.Repo.clone_from(url, dest_path)
            result = {
                "status": "success",
                "url": url,
                "destination": str(dest_path),
                "branch": repo.active_branch.name,
                "message": f"Cloned repository from {url} to {dest_path}",
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error cloning repository: {e}")]

    async def _get_status(self, repo_path: str) -> list[TextContent]:
        """Get Git repository status."""
        repo = self._get_repo(repo_path)

        try:
            status = {
                "branch": repo.active_branch.name if repo.active_branch else "detached",
                "is_dirty": repo.is_dirty(),
                "untracked_files": repo.untracked_files,
                "modified_files": [item.a_path for item in repo.index.diff(None)],
                "staged_files": [item.a_path for item in repo.index.diff("HEAD")],
                "ahead": [],
                "behind": [],
            }

            # Calculate ahead/behind counts
            if repo.active_branch and repo.active_branch.tracking_branch():
                try:
                    tracking_branch = repo.active_branch.tracking_branch()
                    if tracking_branch:
                        ahead_commits = list(
                            repo.iter_commits(
                                f"{tracking_branch.name}..{repo.active_branch.name}"
                            )
                        )
                        behind_commits = list(
                            repo.iter_commits(
                                f"{repo.active_branch.name}..{tracking_branch.name}"
                            )
                        )
                        status["ahead_count"] = len(ahead_commits)
                        status["behind_count"] = len(behind_commits)
                    else:
                        status["ahead_count"] = 0
                        status["behind_count"] = 0
                except Exception:
                    status["ahead_count"] = 0
                    status["behind_count"] = 0
            else:
                status["ahead_count"] = 0
                status["behind_count"] = 0

            return [TextContent(type="text", text=json.dumps(status, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error getting status: {e}")]

    async def _create_branch(
        self, repo_path: str, branch_name: str, switch: bool = True
    ) -> list[TextContent]:
        """Create a new branch."""
        repo = self._get_repo(repo_path)

        try:
            new_branch = repo.create_head(branch_name)

            if switch:
                new_branch.checkout()

            result = {
                "status": "success",
                "branch_name": branch_name,
                "switched": switch,
                "current_branch": (
                    repo.active_branch.name if repo.active_branch else "detached"
                ),
                "message": f"Created branch '{branch_name}'"
                + (" and switched to it" if switch else ""),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error creating branch: {e}")]

    async def _switch_branch(
        self, repo_path: str, branch_name: str
    ) -> list[TextContent]:
        """Switch to an existing branch."""
        repo = self._get_repo(repo_path)

        try:
            branch = repo.heads[branch_name]
            branch.checkout()

            result = {
                "status": "success",
                "previous_branch": (
                    repo.active_branch.name if repo.active_branch else "detached"
                ),
                "current_branch": branch_name,
                "message": f"Switched to branch '{branch_name}'",
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error switching branch: {e}")]

    async def _list_branches(
        self, repo_path: str, include_remote: bool = True
    ) -> list[TextContent]:
        """List all branches."""
        repo = self._get_repo(repo_path)

        try:
            branches = {
                "local": [head.name for head in repo.heads],
                "current": (
                    repo.active_branch.name if repo.active_branch else "detached"
                ),
            }

            if include_remote:
                branches["remote"] = (
                    [ref.name for ref in repo.remotes.origin.refs]
                    if repo.remotes
                    else []
                )

            return [TextContent(type="text", text=json.dumps(branches, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error listing branches: {e}")]

    async def _add_files(
        self, repo_path: str, file_patterns: list[str]
    ) -> list[TextContent]:
        """Stage files for commit."""
        repo = self._get_repo(repo_path)

        try:
            added_files = []
            for pattern in file_patterns:
                # Use git add command to handle patterns properly
                repo.git.add(pattern)
                added_files.append(pattern)

            result = {
                "status": "success",
                "patterns": file_patterns,
                "message": f"Added {len(file_patterns)} file patterns to staging area",
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error adding files: {e}")]

    async def _commit_changes(
        self,
        repo_path: str,
        message: str,
        author_name: str | None = None,
        author_email: str | None = None,
    ) -> list[TextContent]:
        """Commit staged changes."""
        repo = self._get_repo(repo_path)

        try:
            # Set author if provided
            actor = None
            if author_name and author_email:
                actor = git.Actor(author_name, author_email)

            commit = repo.index.commit(message, author=actor)

            result = {
                "status": "success",
                "commit_hash": commit.hexsha,
                "message": message,
                "author": str(commit.author) if commit.author else None,
                "timestamp": commit.committed_datetime.isoformat(),
                "files_changed": len(commit.stats.files),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error committing changes: {e}")]

    async def _get_diff(
        self,
        repo_path: str,
        staged: bool = False,
        commit_range: str | None = None,
        file_paths: list[str] | None = None,
    ) -> list[TextContent]:
        """Get diff of changes."""
        repo = self._get_repo(repo_path)

        try:
            if commit_range:
                # Get diff between commits
                diff = repo.git.diff(commit_range, *file_paths if file_paths else [])
            elif staged:
                # Get staged changes
                diff = repo.git.diff("--cached", *file_paths if file_paths else [])
            else:
                # Get working directory changes
                diff = repo.git.diff(*file_paths if file_paths else [])

            result = {
                "staged": staged,
                "commit_range": commit_range,
                "file_paths": file_paths,
                "diff": diff,
                "lines": len(diff.splitlines()) if diff else 0,
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error getting diff: {e}")]

    async def _get_log(
        self,
        repo_path: str,
        max_count: int = 10,
        branch: str | None = None,
        file_path: str | None = None,
    ) -> list[TextContent]:
        """Get commit history."""
        repo = self._get_repo(repo_path)

        try:
            paths_arg = [file_path] if file_path else None

            commits = []
            if branch:
                if paths_arg:
                    commit_iter = repo.iter_commits(
                        branch, max_count=max_count, paths=paths_arg
                    )
                else:
                    commit_iter = repo.iter_commits(branch, max_count=max_count)
            else:
                if paths_arg:
                    commit_iter = repo.iter_commits(
                        max_count=max_count, paths=paths_arg
                    )
                else:
                    commit_iter = repo.iter_commits(max_count=max_count)

            for commit in commit_iter:
                commits.append(
                    {
                        "hash": commit.hexsha,
                        "short_hash": commit.hexsha[:7],
                        "message": commit.message.strip(),
                        "author": str(commit.author),
                        "timestamp": commit.committed_datetime.isoformat(),
                        "files_changed": len(commit.stats.files) if commit.stats else 0,
                    }
                )

            result = {
                "commits": commits,
                "count": len(commits),
                "branch": branch,
                "file_path": file_path,
                "max_count": max_count,
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error getting log: {e}")]

    async def _get_changed_files(
        self, repo_path: str, base_commit: str = "HEAD~1", target_commit: str = "HEAD"
    ) -> list[TextContent]:
        """Get list of changed files between commits."""
        repo = self._get_repo(repo_path)

        try:
            # Get the diff between commits
            diff = repo.commit(base_commit).diff(repo.commit(target_commit))

            changed_files: dict = {
                "added": [],
                "modified": [],
                "deleted": [],
                "renamed": [],
            }

            for diff_item in diff:
                if diff_item.change_type == "A":
                    changed_files["added"].append(diff_item.b_path)
                elif diff_item.change_type == "M":
                    changed_files["modified"].append(diff_item.a_path)
                elif diff_item.change_type == "D":
                    changed_files["deleted"].append(diff_item.a_path)
                elif diff_item.change_type == "R":
                    changed_files["renamed"].append(
                        {
                            "from": diff_item.a_path,
                            "to": diff_item.b_path,
                        }
                    )

            result = {
                "base_commit": base_commit,
                "target_commit": target_commit,
                "changed_files": changed_files,
                "total_changes": sum(
                    len(files)
                    for files in changed_files.values()
                    if isinstance(files, list)
                ),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error getting changed files: {e}")]

    async def _create_temp_branch(
        self,
        repo_path: str,
        base_branch: str = "main",
        prefix: str = "diversifier-temp",
    ) -> list[TextContent]:
        """Create a temporary branch for migration work."""
        repo = self._get_repo(repo_path)

        try:
            import time

            timestamp = int(time.time())
            temp_branch_name = f"{prefix}-{timestamp}"

            # Switch to base branch first
            base = repo.heads[base_branch]
            base.checkout()

            # Create and switch to temp branch
            temp_branch = repo.create_head(temp_branch_name)
            temp_branch.checkout()

            result = {
                "status": "success",
                "temp_branch": temp_branch_name,
                "base_branch": base_branch,
                "current_branch": repo.active_branch.name,
                "message": f"Created temporary branch '{temp_branch_name}' from '{base_branch}'",
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error creating temp branch: {e}")]

    async def run(self) -> None:
        """Run the MCP server with stdio transport."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="git-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def main():
    """Main entry point for the Git MCP Server."""
    import asyncio
    import sys

    # Get project root from command line argument if provided
    project_root = sys.argv[1] if len(sys.argv) > 1 else None

    server = GitMCPServer(project_root=project_root)

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
