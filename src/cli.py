import argparse
import sys
from pathlib import Path
from typing import Optional

from .validation import (
    validate_python_project,
    validate_library_name_format,
    validate_library_exists,
    validate_migration_feasibility
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diversifier",
        description="AI-powered tool to generate diversified project variants via dependency mutation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  diversifier /path/to/project requests httpx
  diversifier . pandas polars
  diversifier ~/myproject flask fastapi
        """,
    )
    
    parser.add_argument(
        "project_path",
        type=str,
        help="Path to the Python project directory to diversify"
    )
    
    parser.add_argument(
        "remove_lib",
        type=str,
        help="Name of the library to remove/replace"
    )
    
    parser.add_argument(
        "inject_lib",
        type=str,
        help="Name of the library to inject/substitute"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def validate_project_path(path_str: str) -> Optional[Path]:
    try:
        path = Path(path_str).resolve()
        
        if not path.exists():
            print(f"Error: Project path does not exist: {path}")
            return None
            
        if not path.is_dir():
            print(f"Error: Project path is not a directory: {path}")
            return None
            
        return path
    except Exception as e:
        print(f"Error: Invalid project path '{path_str}': {e}")
        return None


def validate_library_name(lib_name: str) -> bool:
    is_valid, message = validate_library_name_format(lib_name)
    if not is_valid:
        print(f"Error: {message}")
    return is_valid


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    
    project_path = validate_project_path(args.project_path)
    if not project_path:
        return 1
        
    if not validate_library_name(args.remove_lib):
        return 1
        
    if not validate_library_name(args.inject_lib):
        return 1
    
    if args.remove_lib == args.inject_lib:
        print("Error: remove_lib and inject_lib cannot be the same")
        return 1
    
    if args.verbose:
        print(f"Project path: {project_path}")
        print(f"Remove library: {args.remove_lib}")
        print(f"Inject library: {args.inject_lib}")
        print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("DRY RUN: Would perform library substitution:")
        print(f"  Project: {project_path}")
        print(f"  Replace '{args.remove_lib}' with '{args.inject_lib}'")
        return 0
    
    print("Diversifier CLI is ready - core migration logic to be implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())