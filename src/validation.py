import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Tuple


def validate_python_project(project_path: Path) -> Tuple[bool, List[str]]:
    errors = []
    
    python_files = list(project_path.rglob("*.py"))
    if not python_files:
        errors.append("No Python files found in project")
    
    has_setup = (
        (project_path / "setup.py").exists() or
        (project_path / "pyproject.toml").exists() or
        (project_path / "setup.cfg").exists()
    )
    
    if not has_setup and not python_files:
        errors.append("Project does not appear to be a valid Python project (no setup files or Python files found)")
    
    return len(errors) == 0, errors


def validate_library_exists(library_name: str) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", library_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return True, f"Library '{library_name}' found on PyPI"
        else:
            return False, f"Library '{library_name}' not found on PyPI"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout checking library '{library_name}' on PyPI"
    except Exception as e:
        return False, f"Error checking library '{library_name}': {str(e)}"


def validate_library_name_format(lib_name: str) -> Tuple[bool, str]:
    if not lib_name:
        return False, "Library name cannot be empty"
    
    if len(lib_name) > 214:
        return False, "Library name too long (max 214 characters)"
    
    if not re.match(r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", lib_name, re.IGNORECASE):
        return False, "Library name must start and end with alphanumeric character, can contain letters, numbers, periods, hyphens, and underscores"
    
    if ".." in lib_name:
        return False, "Library name cannot contain consecutive periods"
        
    if lib_name.startswith("-") or lib_name.endswith("-"):
        return False, "Library name cannot start or end with hyphen"
        
    if lib_name.startswith("_") or lib_name.endswith("_"):
        return False, "Library name cannot start or end with underscore"
    
    return True, f"Library name '{lib_name}' is valid"


def find_library_usage(project_path: Path, library_name: str) -> Tuple[bool, List[str]]:
    usage_patterns = [
        rf"^import\s+{re.escape(library_name)}\b",
        rf"^from\s+{re.escape(library_name)}\b",
        rf"^import\s+{re.escape(library_name.replace('-', '_'))}\b",
        rf"^from\s+{re.escape(library_name.replace('-', '_'))}\b",
    ]
    
    found_files = []
    
    for py_file in project_path.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern in usage_patterns:
                if re.search(pattern, content, re.MULTILINE):
                    found_files.append(str(py_file.relative_to(project_path)))
                    break
                    
        except (UnicodeDecodeError, PermissionError):
            continue
    
    return len(found_files) > 0, found_files


def validate_migration_feasibility(project_path: Path, remove_lib: str, inject_lib: str) -> Tuple[bool, List[str]]:
    issues = []
    
    has_remove_lib, remove_files = find_library_usage(project_path, remove_lib)
    if not has_remove_lib:
        issues.append(f"Library '{remove_lib}' does not appear to be used in the project")
    
    has_inject_lib, inject_files = find_library_usage(project_path, inject_lib)
    if has_inject_lib:
        issues.append(f"Library '{inject_lib}' is already used in the project. Files: {', '.join(inject_files[:3])}")
    
    return len(issues) == 0, issues