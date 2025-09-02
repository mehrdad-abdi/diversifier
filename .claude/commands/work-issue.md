# Work Issue Command

You are a senior software engineer working on the Diversifier project. You will receive a GitHub issue number via $ARGUMENTS and implement a complete development workflow for it.

**Issue Number**: $ARGUMENTS

## Your Task

Implement the following workflow for the given GitHub issue number:

1. **Read Project Documentation**: Understand README.md and CLAUDE.md first
2. **Git Setup**: Start from main branch with latest changes (`git pull`)
3. **Fetch and Analyze Issue**: Get the issue details and understand the requirements
4. **Research Related Code**: Find existing patterns and code to integrate with
5. **Create Development Branch**: Create a properly named branch for the work
6. **Ask Clarifying Questions**: Identify vague points and ask up to 5 meaningful questions
7. **Break Down Work**: Split the issue into smaller, manageable steps after clarification
8. **Test-First Development**: For each step, write tests first, then implement consistently
9. **Quality Checks**: Run all required quality checks in order
10. **Final Formatting**: Run black formatting as the final step
11. **Create Pull Request**: Create PR with auto-close and reviewer assignment

## Specific Instructions

### 1. Project Understanding and Setup
- **FIRST**: Read README.md and CLAUDE.md to understand the project context, architecture, and global rules
- **ALWAYS**: Start from main branch and run `git pull` to ensure you have the latest changes
- **REQUIRED**: Search for related existing code in the project to understand patterns and consistency

### 2. Issue Analysis and Branch Creation
- Fetch the issue using `gh issue view $ARGUMENTS --json title,body,labels,milestone`
- Create a branch named `feature/issue-$ARGUMENTS-{clean-title}` or `fix/issue-$ARGUMENTS-{clean-title}`
- Ensure you're starting from the latest main branch after git pull

### 3. Work Breakdown and Code Integration
- Analyze the issue requirements thoroughly
- **Search for related existing code** in the project using Grep, Glob, or Task tools
- **Study existing patterns** to ensure consistency with current project style
- **IMPORTANT**: Never make assumptions! If anything is unclear, ask the user for clarification
- **Identify vague points** in the issue and prepare up to 5 meaningful clarification questions
- **Only ask non-trivial questions** - avoid obvious or easily answerable questions
- **Ask questions BEFORE** breaking down the work to ensure clear understanding

**Examples of good clarification questions:**
- "Should the new feature integrate with the existing MCP server architecture or be standalone?"
- "What specific error handling behavior is expected when the library substitution fails?"
- "Should this support async/sync versions of the target library, or just one?"
- "What's the expected behavior for edge case X that isn't mentioned in the issue?"
- "Should this follow the existing pattern in file Y or create a new approach?"
- "Should I implement this with a simple approach X or more complex approach Y? (always lean toward simpler)"

**Avoid trivial questions like:**
- "What programming language should I use?" (obvious from context)
- "Should I write tests?" (already specified in workflow)
- "Where should I put the code?" (can be determined from existing patterns)

- After clarification, break down the work into 3-7 concrete, testable steps
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
- **Keep implementations SIMPLE**: Choose the most straightforward approach that works
- **Avoid unnecessary complexity**: Prefer readable, maintainable code over clever solutions
- **Refactor for simplicity**: If code becomes complex, break it into smaller, simpler parts
- Ensure comprehensive test coverage including edge cases
- Tests must pass before moving to the next step

### 5. Quality Checks (MANDATORY ORDER)
Run these commands in exactly this order before creating the PR:
```bash
uv run ruff check src/ tests/           # Linting
uv run mypy src/ tests/                 # Type checking  
uv run pytest                                   # All tests must pass
uv run black src/ tests/                # Final formatting (LAST!)
```

All checks must pass. If any fail, fix the issues before proceeding.

### 6. Git Workflow
- **Always start from main**: `git checkout main && git pull` before creating branch
- Commit changes with meaningful commit messages
- Include the issue number in commit messages
- Use the format: `fix: {description} (closes #$ARGUMENTS)` or `feat: {description} (closes #$ARGUMENTS)`
- Push the branch to origin

### 7. Pull Request Creation
- Create PR using `gh pr create` with:
  - Title: `Fix/Feat: {issue title} (#$ARGUMENTS)`
  - Comprehensive description including:
    - Summary of changes
    - List of steps implemented
    - Testing approach used
    - Edge cases considered
  - Auto-close directive: `Closes #$ARGUMENTS`
  - Reviewer assignment: `--reviewer mehrdad-abdi`
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

## Core Principles

- **No Assumptions Rule**: Never assume unclear requirements - always ask for clarification
- **Question Quality**: Ask meaningful questions that impact implementation decisions
- **Clarity First**: Get clear requirements before coding to avoid rework
- **User Guidance**: When in doubt, involve the user in the decision-making process
- **SIMPLICITY FIRST**: Always prioritize simplicity over complexity - the project MUST remain maintainable
- **Avoid Over-Engineering**: Choose the simplest solution that meets requirements
- **Maintainability**: Code should be easy to read, understand, and modify by other developers

## Error Handling

If any step fails:
- Analyze the error carefully
- Fix the issue before proceeding
- Re-run quality checks after fixes
- Never skip quality checks or create PRs with failing tests
- **If errors involve unclear requirements**: Ask the user for clarification rather than guessing

## Success Criteria

The workflow is complete when:
- ✅ Issue is fully implemented according to requirements
- ✅ All tests pass (existing + new)
- ✅ All quality checks pass
- ✅ Code is properly formatted with black
- ✅ **Code is simple, readable, and maintainable**
- ✅ **No unnecessary complexity was introduced**
- ✅ PR is created with auto-close and reviewer assigned
- ✅ Branch is pushed to origin

## Usage

Use this command by providing the issue number as a parameter:
```
/work-issue 42
```

This will start the complete workflow for GitHub issue #42, where 42 is passed as $ARGUMENTS to the command.