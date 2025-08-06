# Diversifier

> An LLM-based agent that performs automated library substitution in containerized projects while maintaining functional equivalence through rigorous testing.

## Why Diversifier?

In modern software deployment, **monoculture vulnerabilities** pose significant risks. When all instances of an application use identical dependencies, a single vulnerability can compromise the entire deployment. Diversifier addresses this through **automated software diversity** - a proven cybersecurity strategy.

### 🛡️ Security Through Diversity

**The Monoculture Problem:**
When your entire Kubernetes cluster runs identical containers with the same vulnerable library (e.g., a compromised HTTP client or JSON parser), attackers can:
- Exploit the vulnerability across all pods simultaneously
- Achieve complete service compromise with a single attack vector
- Bypass traditional security measures that assume attack diversity

**Diversifier's Solution:**
Generate functionally equivalent application variants using different libraries, enabling:
- **Reduced Attack Surface**: Different libraries have different vulnerability profiles
- **Blast Radius Limitation**: Exploits affect only a subset of your deployment
- **Rapid Incident Response**: Shift traffic to unaffected variants during security events

### 🎯 Moving Target Defense (MTD)

Diversifier implements **Moving Target Defense** principles by creating multiple valid targets:

```bash
# Before: Single target for attackers
kubectl get pods
NAME                     READY   STATUS
api-server-requests-1    1/1     Running  # All use requests library
api-server-requests-2    1/1     Running  # Same vulnerability
api-server-requests-3    1/1     Running  # Same attack vector

# After: Diversified deployment  
kubectl get pods
NAME                     READY   STATUS
api-server-requests-1    1/1     Running  # Uses requests
api-server-httpx-1       1/1     Running  # Uses httpx (different codebase)
api-server-aiohttp-1     1/1     Running  # Uses aiohttp (async)
```

**MTD Benefits:**
- **Attack Uncertainty**: Attackers cannot predict which library variant they'll encounter
- **Increased Attack Cost**: Requires multiple exploit chains for complete compromise
- **Dynamic Reconfiguration**: Adjust library distribution based on threat intelligence

### 🔀 Software Diversity & Randomization

Diversifier applies established **software diversity techniques** from systems security:

####  **Instruction Set Randomization (ISR) Inspiration**
Just as ISR randomizes instruction encodings to prevent code injection:
- **Library API Randomization**: Different libraries expose different APIs for identical functionality
- **Implementation Randomization**: Varies internal algorithms and data structures
- **Error Handling Randomization**: Different failure modes and exception patterns

#### **Address Space Layout Randomization (ASLR) Parallels**  
Similar to how ASLR randomizes memory layouts:
- **Dependency Randomization**: Varies the dependency graph structure
- **Call Stack Randomization**: Different function call patterns between libraries
- **Resource Usage Randomization**: Different memory and CPU usage patterns

### 🚀 Operational Advantages

**Zero-Downtime Security Response:**
```bash
# CVE discovered in requests library
kubectl scale deployment api-server-requests --replicas=0    # Scale down vulnerable variant
kubectl scale deployment api-server-httpx --replicas=10     # Scale up safe variant
# Service continues with zero downtime
```

**Canary Deployments for Security:**
- Deploy new library variants as canaries
- Test security posture with minimal traffic exposure  
- Gradually shift traffic based on security assessment

**Supply Chain Attack Mitigation:**
- Reduces dependency on single library maintainers
- Distributes supply chain risk across multiple projects
- Enables rapid pivot away from compromised dependencies

### 📊 Threat Model Coverage

| Attack Vector | Monoculture Risk | Diversified Defense |
|---------------|------------------|---------------------|
| **Library RCE** | 100% compromise | ~33% impact (varies by distribution) |
| **Supply Chain** | Complete takeover | Partial compromise, rapid mitigation |
| **Zero-Day Exploits** | Total exposure | Probabilistic protection |
| **Dependency Confusion** | Full deployment | Limited blast radius |
| **Insider Threats** | Single point of failure | Multiple targets required |

### 🎖️ Industry Alignment

**NIST Cybersecurity Framework:**
- **Identify**: Map library dependencies and vulnerability exposure
- **Protect**: Implement diversity-based protective controls
- **Detect**: Monitor variants for differential behavior
- **Respond**: Rapid incident response through traffic reshaping
- **Recover**: Maintain service availability during security events

**Zero Trust Architecture:**
- **Never Trust, Always Verify**: Each library variant operates under same security controls
- **Least Privilege**: Variants can have different permission sets if needed
- **Continuous Verification**: Monitor behavior across diverse implementations

### 🔬 Research Foundation

Diversifier builds on decades of academic research:

**Software Diversity Research:**
- [Instruction Set Randomization](https://people.csail.mit.edu/rinard/paper/asplos03.pdf) (MIT, 2003)
- [Address Space Layout Randomization](https://pax.grsecurity.net/docs/aslr.txt) (PaX Team, 2001)
- [N-Version Programming](https://ieeexplore.ieee.org/document/1702202) (Chen & Avizienis, 1978)

**Moving Target Defense:**
- [MTD Effectiveness Studies](https://www.dhs.gov/science-and-technology/csd-mtd) (DHS S&T)
- [Dynamic Platform Techniques](https://dl.acm.org/doi/10.1145/2435349.2435352) (ACM CCS, 2012)

**Modern Applications:**
- Container diversity in cloud deployments
- Microservice architecture resilience patterns
- DevSecOps integration strategies

When vulnerabilities are discovered in critical Python libraries, teams can quickly generate alternative versions of their applications using different libraries, allowing them to:
- **🛡️ Implement Defense in Depth**: Layer security through architectural diversity
- **⚡ Enable Rapid Response**: Reduce MTTR (Mean Time To Recovery) from hours to minutes  
- **🧪 Support Safe Testing**: Validate alternative implementations in production-like environments
- **🚀 Maintain Zero Downtime**: Deploy alternative versions alongside originals in Kubernetes
- **📈 Improve Security Posture**: Transition from reactive patching to proactive defense

## Quick Start

### Prerequisites

- Python >=3.13
- uv package manager
- Access to your Python project's source code

### 1. Install Diversifier

```bash
# Clone and install
git clone https://github.com/mehrdad-abdi/diversifier
cd diversifier
uv sync
```

### 2. Run Migration

```bash
# Example: Migrate from requests to httpx
uv run diversifier /path/to/your/project requests httpx
```

### 3. Review Results

The agent will create a migrated version of your project and provide a comprehensive report showing:
- All code changes made
- Test results comparison (functional equivalence validation)
- Migration success/failure status
- Deployment readiness confirmation

## How It Works

```mermaid
graph LR
    A[Analyze Project] --> B[Generate Tests]
    B --> C[Test Original]
    C --> D[Migrate Code]
    D --> E[Test Migration]
    E --> F{Tests Pass?}
    F -->|No| G[Debug & Fix]
    G --> E
    F -->|Yes| H[Generate Report]
```

### 1. 🔍 Project Analysis
- Scans Python files for import statements and library usage
- Analyzes `requirements.txt`, `pyproject.toml`, `setup.py`
- Identifies all functions and methods using the target library
- Maps dependencies and potential compatibility issues

### 2. 🧪 Test Generation  
- Creates library-independent acceptance tests covering core functionality
- Analyzes project documentation and existing tests for patterns
- Generates pytest-compatible test files focused on key functionality
- Tests validate external behavior without depending on specific libraries

### 3. ✅ Baseline Validation
- Runs generated acceptance tests against original project
- Captures baseline results for functional equivalence comparison
- Validates test reliability and coverage
- Establishes success criteria for migration

### 4. 🔄 Code Migration
- Uses LLM prompts to analyze and transform code
- Updates import statements and API usage patterns
- Handles library-specific transformations (async/sync, parameter mapping)
- Updates configuration files and dependencies via MCP servers

### 5. 🚀 Migration Testing
- Runs acceptance tests on transformed project
- Compares results with baseline for functional equivalence
- Identifies failures that need repair
- Determines migration success or triggers repair workflow

### 6. 🛠️ Debugging & Repair
- Uses LLM prompts to analyze test failures and diagnose issues
- Generates corrective code changes for transformation problems
- Applies fixes via MCP servers with rollback capability
- Iterates repair → test → analyze cycle until all tests pass

### 7. 📋 Results & Reporting
- Generates comprehensive migration report
- Documents all code changes made
- Provides side-by-side comparison of test results
- Confirms production readiness


## Development Strategy

### Phase 1: Python MVP (Minimum Viable Product)
**Scope**: Focus exclusively on Python projects to establish core functionality and prove the concept.

**Target Libraries**: Common Python library substitutions
- HTTP clients: `requests` → `httpx` or `aiohttp`
- Database drivers: `psycopg2` → `asyncpg` or `SQLAlchemy` → `Tortoise ORM`
- JSON processing: `json` → `orjson` or `ujson`
- Logging: `logging` → `loguru`

**Benefits of Python-first approach**:
- Clear dependency management (requirements.txt, pyproject.toml)
- Predictable import patterns (`import`, `from X import Y`)
- Rich ecosystem of functionally equivalent libraries
- Mature containerization practices
- Excellent testing framework support

### Future Phases

Make the tool multi-language based on MVP feedback.

## Core Use Case

When a vulnerability is discovered in a critical Python library (e.g., a specific HTTP client, database driver, or JSON parser), teams can quickly generate an alternative version of their application using a different library, allowing them to:
- Maintain service availability by diversifying pod deployments
- Reduce blast radius of security vulnerabilities
- Test alternative implementations before full migration

## Development Roadmap

The project development is organized into 7 epics with 35 individual tasks tracked in GitHub issues:

### Epic 1: CLI Interface & Core Architecture
- CLI entry point with argument parsing (`diversifier <project_path> <remove_lib> <inject_lib>`)
- LangChain orchestration and MCP client integration
- Configuration management and logging
- Error handling and user feedback

### Epic 2: MCP Server Infrastructure  
- Local MCP server architecture with stdio communication
- File System MCP (code analysis and modification)
- Git MCP (version control operations)
- Testing MCP (test execution and validation)
- Optional Docker MCP (isolated test environments)

### Epic 3: Library-Independent Test Generation
- LLM prompts for documentation and test analysis
- Comprehensive acceptance test generation
- Baseline test execution and validation

### Epic 4: LLM-Driven Code Migration
- Project analysis and library usage detection
- Import and API transformation prompts
- Configuration file updates
- Migration workflow orchestration

### Epic 5: Post-Migration Testing & Validation
- Test execution on transformed projects
- Result analysis and comparison
- Success/failure determination

### Epic 6: Iterative Transformation Repair
- Failure analysis and debugging prompts
- Corrective change generation
- Repair iteration loops

### Epic 7: Integration & Documentation
- End-to-end pipeline integration
- Example migrations and documentation

## MCP Server Architecture

Diversifier uses **local MCP servers with stdio communication** - a simpler, more secure approach than containerized services.

### 📁 File System MCP Server
- **Purpose**: Project file manipulation and analysis
- **Communication**: Local stdio process
- **Capabilities**: Python AST parsing, import analysis, code modification
- **Integration**: Analyzes dependencies, transforms code, updates requirements

### 🧪 Testing MCP Server  
- **Purpose**: Test execution and validation
- **Communication**: Local stdio process
- **Capabilities**: pytest execution, result parsing, coverage analysis
- **Integration**: Runs acceptance tests, compares baseline results

### 🔄 Git MCP Server
- **Purpose**: Version control operations
- **Communication**: Local stdio process
- **Capabilities**: Branch management, change tracking, commit operations
- **Integration**: Creates migration branches, tracks transformations

### 🐳 Docker MCP Server (Optional)
- **Purpose**: Isolated test environments
- **Communication**: Local stdio process
- **Capabilities**: Container lifecycle for complex test scenarios
- **Integration**: Used only when local testing is insufficient

### Architecture Benefits
- **Simplified Deployment**: No Docker Compose complexity
- **Enhanced Security**: Local processes, no network exposure
- **Better Performance**: Direct stdio communication
- **Easier Development**: Standard Python subprocess management

## License

MIT License - see [LICENSE](LICENSE) file for details.
