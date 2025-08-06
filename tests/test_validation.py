from unittest.mock import patch, MagicMock
import subprocess

from src.validation import (
    validate_python_project,
    validate_library_exists,
    validate_library_name_format,
    find_library_usage,
    validate_migration_feasibility,
    validate_project_path,
    validate_library_name,
)


class TestValidatePythonProject:
    def test_valid_python_project_with_py_files(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "module.py").write_text("def func(): pass")

        is_valid, errors = validate_python_project(tmp_path)
        assert is_valid is True
        assert len(errors) == 0

    def test_valid_python_project_with_pyproject_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "main.py").write_text("print('hello')")

        is_valid, errors = validate_python_project(tmp_path)
        assert is_valid is True
        assert len(errors) == 0

    def test_valid_python_project_with_setup_py(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup")
        (tmp_path / "main.py").write_text("print('hello')")

        is_valid, errors = validate_python_project(tmp_path)
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_project_no_python_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("Not a python project")

        is_valid, errors = validate_python_project(tmp_path)
        assert is_valid is False
        assert "No Python files found" in errors[0]

    def test_invalid_project_no_setup_no_python(self, tmp_path):
        is_valid, errors = validate_python_project(tmp_path)
        assert is_valid is False
        assert len(errors) > 0


class TestValidateLibraryExists:
    @patch("subprocess.run")
    def test_library_exists_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        is_valid, message = validate_library_exists("requests")
        assert is_valid is True
        assert "requests" in message
        assert "found on PyPI" in message

    @patch("subprocess.run")
    def test_library_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)

        is_valid, message = validate_library_exists("nonexistent-lib")
        assert is_valid is False
        assert "not found on PyPI" in message

    @patch("subprocess.run")
    def test_library_check_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("pip", 10)

        is_valid, message = validate_library_exists("requests")
        assert is_valid is False
        assert "Timeout" in message

    @patch("subprocess.run")
    def test_library_check_exception(self, mock_run):
        mock_run.side_effect = Exception("Connection error")

        is_valid, message = validate_library_exists("requests")
        assert is_valid is False
        assert "Error checking" in message


class TestValidateLibraryNameFormat:
    def test_valid_library_names(self):
        valid_names = [
            "requests",
            "django-rest-framework",
            "my_package",
            "test.pkg",
            "Package123",
            "a",
            "A1",
        ]

        for name in valid_names:
            is_valid, message = validate_library_name_format(name)
            assert is_valid is True, f"'{name}' should be valid: {message}"

    def test_invalid_library_names(self):
        invalid_cases = [
            ("", "cannot be empty"),
            ("package..name", "consecutive periods"),
            ("-package", "hyphen"),
            ("package-", "hyphen"),
            ("_package", "underscore"),
            ("package_", "underscore"),
            ("package@name", "alphanumeric"),
            ("a" * 215, "too long"),
        ]

        for name, expected_error in invalid_cases:
            is_valid, message = validate_library_name_format(name)
            assert is_valid is False, f"'{name}' should be invalid"
            assert expected_error in message.lower()


class TestFindLibraryUsage:
    def test_find_import_statement(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("import requests\nprint('hello')")

        found, files = find_library_usage(tmp_path, "requests")
        assert found is True
        assert "test.py" in files[0]

    def test_find_from_import(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("from requests import get\nprint('hello')")

        found, files = find_library_usage(tmp_path, "requests")
        assert found is True
        assert "test.py" in files[0]

    def test_find_hyphenated_package(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("import django_rest_framework")

        found, files = find_library_usage(tmp_path, "django-rest-framework")
        assert found is True
        assert "test.py" in files[0]

    def test_no_usage_found(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello world')")

        found, files = find_library_usage(tmp_path, "requests")
        assert found is False
        assert len(files) == 0

    def test_multiple_files_with_usage(self, tmp_path):
        (tmp_path / "file1.py").write_text("import requests")
        (tmp_path / "file2.py").write_text("from requests import get")
        (tmp_path / "file3.py").write_text("print('no import')")

        found, files = find_library_usage(tmp_path, "requests")
        assert found is True
        assert len(files) == 2

    def test_unicode_decode_error_handling(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_bytes(b"\xff\xfe")  # Invalid UTF-8

        found, files = find_library_usage(tmp_path, "requests")
        assert found is False
        assert len(files) == 0


class TestValidateMigrationFeasibility:
    def test_feasible_migration(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("import requests\nprint('hello')")

        is_feasible, issues = validate_migration_feasibility(
            tmp_path, "requests", "httpx"
        )
        assert is_feasible is True
        assert len(issues) == 0

    def test_remove_library_not_used(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("import os\nprint('hello')")

        is_feasible, issues = validate_migration_feasibility(
            tmp_path, "requests", "httpx"
        )
        assert is_feasible is False
        assert any("does not appear to be used" in issue for issue in issues)

    def test_inject_library_already_used(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("import requests\nimport httpx")

        is_feasible, issues = validate_migration_feasibility(
            tmp_path, "requests", "httpx"
        )
        assert is_feasible is False
        assert any("already used in the project" in issue for issue in issues)

    def test_both_libraries_problematic(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("import httpx\nprint('no requests')")

        is_feasible, issues = validate_migration_feasibility(
            tmp_path, "requests", "httpx"
        )
        assert is_feasible is False
        assert len(issues) == 2


class TestValidateProjectPath:
    def test_valid_directory_path(self, tmp_path):
        result = validate_project_path(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_nonexistent_path(self):
        with patch("builtins.print"):
            result = validate_project_path("/nonexistent/path")
            assert result is None

    def test_file_instead_of_directory(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch("builtins.print"):
            result = validate_project_path(str(test_file))
            assert result is None

    def test_relative_path_resolution(self):
        with patch("builtins.print"):
            result = validate_project_path(".")
            assert result is not None
            assert result.is_absolute()


class TestValidateLibraryName:
    def test_valid_library_names(self):
        valid_names = [
            "requests",
            "django-rest-framework",
            "my_package",
            "test.pkg",
            "Package123",
        ]

        for name in valid_names:
            with patch("builtins.print"):
                assert validate_library_name(name) is True

    def test_invalid_library_names(self):
        invalid_names = [
            "",
            "package..name",
            "-package",
            "package-",
            "_package",
            "package_",
        ]

        for name in invalid_names:
            with patch("builtins.print"):
                assert validate_library_name(name) is False
