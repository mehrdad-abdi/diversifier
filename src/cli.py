import argparse
import sys

from .validation import (
    validate_python_project,
    validate_project_path,
    validate_library_name,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diversifier",
        description="AI-powered tool to generate diversified project variants "
        "via dependency mutation.",
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
        help="Path to the Python project directory to diversify",
    )

    parser.add_argument(
        "remove_lib", type=str, help="Name of the library to remove/replace"
    )

    parser.add_argument(
        "inject_lib", type=str, help="Name of the library to inject/substitute"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    project_path = validate_project_path(args.project_path)
    if not project_path:
        return 1

    is_valid_python_project, project_errors = validate_python_project(
        project_path
    )
    if not is_valid_python_project:
        print("Error: Invalid Python project:")
        for error in project_errors:
            print(f"  - {error}")
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
