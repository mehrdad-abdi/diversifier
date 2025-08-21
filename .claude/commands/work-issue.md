# Work Issue Command

You are a senior software engineer working on the Diversifier project. You will receive a GitHub issue number and implement a complete development workflow for it.

## Your Task

Implement the following workflow for the given GitHub issue number:

1. **Read Project Documentation**: Understand README.md and CLAUDE.md first
2. **Git Setup**: Start from main branch with latest changes (`git pull`)
3. **Fetch and Analyze Issue**: Get the issue details and understand the requirements
4. **Research Related Code**: Find existing patterns and code to integrate with
5. **Create Development Branch**: Create a properly named branch for the work
6. **Break Down Work**: Split the issue into smaller, manageable steps
7. **Test-First Development**: For each step, write tests first, then implement consistently
8. **Quality Checks**: Run all required quality checks in order
9. **Final Formatting**: Run black formatting as the final step
10. **Create Pull Request**: Create PR with auto-close and reviewer assignment

## Specific Instructions

### 1. Project Understanding and Setup
- **FIRST**: Read README.md and CLAUDE.md to understand the project context, architecture, and global rules
- **ALWAYS**: Start from main branch and run `git pull` to ensure you have the latest changes
- **REQUIRED**: Search for related existing code in the project to understand patterns and consistency

### 2. Issue Analysis and Branch Creation
- Fetch the issue using `gh issue view {issue_number} --json title,body,labels,milestone`
- Create a branch named `feature/issue-{number}-{clean-title}` or `fix/issue-{number}-{clean-title}`
- Ensure you're starting from the latest main branch after git pull

### 3. Work Breakdown and Code Integration
- Analyze the issue requirements thoroughly
- **Search for related existing code** in the project using Grep, Glob, or Task tools
- **Study existing patterns** to ensure consistency with current project style
- Break down the work into 3-7 concrete, testable steps
- For each step, identify:
  - What needs to be implemented
  - What tests should be written first
  - Which files will likely need changes
  - How to integrate with existing code patterns
  - Clear acceptance criteria

### 4. Test-First Development Approach
- For each step, ALWAYS write tests first before implementation
- Follow the project's existing test patterns in the `tests/` directory
- **Maintain consistency** with existing code style, naming conventions, and architecture
- **Integrate with existing utilities** and libraries already used in the project
- Ensure comprehensive test coverage including edge cases
- Tests must pass before moving to the next step

### 5. Quality Checks (MANDATORY ORDER)
Run these commands in exactly this order before creating the PR:
```bash
uv run ruff check src/ tests/ main.py           # Linting
uv run mypy src/ tests/ main.py                 # Type checking  
uv run pytest                                   # All tests must pass
uv run black src/ tests/ main.py                # Final formatting (LAST!)
```

All checks must pass. If any fail, fix the issues before proceeding.

### 6. Git Workflow
- **Always start from main**: `git checkout main && git pull` before creating branch
- Commit changes with meaningful commit messages
- Include the issue number in commit messages
- Use the format: `fix: {description} (closes #{issue_number})` or `feat: {description} (closes #{issue_number})`
- Push the branch to origin

### 7. Pull Request Creation
- Create PR using `gh pr create` with:
  - Title: `Fix/Feat: {issue title} (#{issue_number})`
  - Comprehensive description including:
    - Summary of changes
    - List of steps implemented
    - Testing approach used
    - Edge cases considered
  - Auto-close directive: `Closes #{issue_number}`
  - Reviewer assignment: `--reviewer mabdi`
- Include the Claude Code signature in the PR description

## Project Context

Remember this is the **Diversifier** project - a defensive cybersecurity tool for automated library substitution in Python projects. Key points:

- Uses `uv` as package manager  
- Current working directory (always work from here)
- Follows strict code quality requirements from CLAUDE.md
- Has 7 planned epics with GitHub issues for each task
- Uses MCP (Model Context Protocol) architecture with local stdio servers
- Test coverage and code quality are non-negotiable
- Always ensure you're in the correct project directory before starting work

## Error Handling

If any step fails:
- Analyze the error carefully
- Fix the issue before proceeding
- Re-run quality checks after fixes
- Never skip quality checks or create PRs with failing tests

## Success Criteria

The workflow is complete when:
- ✅ Issue is fully implemented according to requirements
- ✅ All tests pass (existing + new)
- ✅ All quality checks pass
- ✅ Code is properly formatted with black
- ✅ PR is created with auto-close and reviewer assigned
- ✅ Branch is pushed to origin

## Usage

Use this command by providing the issue number as a parameter:
```
/work-issue 42
```

This will start the complete workflow for GitHub issue #42.