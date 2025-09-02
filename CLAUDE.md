# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diversifier is a defensive cybersecurity tool that performs automated library substitution in Python projects to implement software diversity and Moving Target Defense (MTD) strategies. The tool uses LLM agents to migrate codebases between functionally equivalent libraries (e.g., `requests` → `httpx`) while maintaining functional equivalence through rigorous testing.

**Security Context**: This is a legitimate defensive security tool designed to reduce monoculture vulnerabilities in deployments. It helps security teams create diverse variants of applications to limit attack blast radius and enable rapid incident response.

## Development Commands

This project uses `uv` as the package manager. Common development tasks:

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev

# Run the main CLI tool (when implemented)
uv run diversifier <project_path> <remove_lib> <inject_lib>

# Code formatting
uv run black src/

# Linting
uv run ruff check src/

# Type checking
uv run mypy src/

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing
```

## Architecture

Diversifier uses a **Model Context Protocol (MCP) based architecture** with specialized servers:

### Core Components
- **Main Agent** (`src/`): LLM-powered migration orchestration using LangChain
- **MCP Servers**: Specialized microservices for different operations (planned implementation)

### MCP Server Architecture

Diversifier uses **local MCP servers with stdio communication** for better security, performance, and simplicity:

1. **File System MCP Server**: Python AST parsing, import analysis, code modification via local stdio
2. **Testing MCP Server**: pytest execution, result analysis, coverage reporting via local stdio  
3. **Git MCP Server**: Version control operations, branch management, change tracking via local stdio
4. **Docker MCP Server** (Optional): Container operations only when local testing is insufficient

**Architecture Benefits:**
- **Simplified Deployment**: No Docker Compose complexity required
- **Enhanced Security**: Local processes, no network exposure or container vulnerabilities
- **Better Performance**: Direct stdio communication, faster than HTTP/containerized services
- **Easier Development**: Standard Python subprocess management patterns

### Migration Workflow (7 Epics)
1. **CLI Interface & Core Architecture**: Command-line tool and LangChain orchestration
2. **MCP Server Infrastructure**: Local MCP servers with stdio communication
3. **Library-Independent Test Generation**: Create acceptance tests using LLM analysis
4. **Project Analysis & Code Migration**: LLM-driven transformation via MCP servers
5. **Post-Migration Testing & Validation**: Execute tests and validate equivalence
6. **Iterative Transformation Repair**: Debug failures and apply corrections
7. **Integration & Documentation**: End-to-end pipeline and examples

## Current Development State

The project is in early development phase with GitHub project management:
- **7 GitHub Milestones**: Created for each epic
- **35 GitHub Issues**: Individual tasks with dependencies and acceptance criteria  
- **Basic project structure**: Dependencies defined in pyproject.toml
- **Architecture planned**: MCP-based system with LLM-driven intelligence
- **Implementation pending**: Core functionality to be built according to GitHub roadmap

## Key Dependencies

- **LangChain**: LLM integration for code analysis and migration
- **Development Tools**: black (formatting), ruff (linting), mypy (type checking), pytest (testing)
- **Python**: Requires Python >=3.13

## Development Guidelines

### Code Quality Requirements
- **ALWAYS write unit tests** for every code change, new function, or feature
- **MANDATORY pre-push checks**: Run these commands before every push to the repository:
  ```bash
  uv run ruff check src/ tests/           # Linting
  uv run mypy src/ tests/      # Type checking
  uv run pytest                       # Unit tests
  uv run black src/ tests/     # Code formatting -- Important: Run as last check
  ```
- All checks must pass before pushing code to any branch
- Test coverage should be comprehensive, including edge cases and error conditions
- Use descriptive test names that explain the expected behavior being tested
- Mock external dependencies (file system, network, subprocess) in unit tests

### Git Workflow Requirements
- **ALWAYS work on a GitHub issue**: Every development task must correspond to an existing GitHub issue
- **Create dedicated dev branches**: Use descriptive branch names like `feature/issue-#-description` or `fix/issue-#-description`
- **Complete workflow**: When work is finished, create a PR → merge PR → close the associated issue
- Never work directly on main branch for development tasks

### Security Considerations
- This tool is designed for defensive security purposes only
- Focus on library migrations that maintain functional equivalence
- Ensure generated tests comprehensively validate application behavior
- All migrations must pass existing and generated test suites
- Tool should never introduce new vulnerabilities during migration

### Key Architecture Principles
- **LLM-Driven Intelligence**: Use sophisticated prompts rather than hardcoded logic
- **Local MCP Architecture**: Use stdio-based MCP servers instead of containerized services
- **MCP Server Coordination**: Leverage specialized local servers for different operations
- **Functional Equivalence Focus**: Ignore performance differences in POC phase
- **Iterative Repair**: Build in debugging and correction capabilities
- **Library Independence**: Generate tests that don't depend on specific implementations
- **Security-First Design**: Local processes minimize attack surface for defensive security tool