import pytest
import sys
from unittest.mock import patch
from io import StringIO

from src.cli import (
    main,
    create_parser,
)


class TestCreateParser:
    def test_parser_creation(self):
        parser = create_parser()
        assert parser.prog == "diversifier"

    def test_parser_required_arguments(self):
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])

        args = parser.parse_args([".", "requests", "httpx"])
        assert args.project_path == "."
        assert args.remove_lib == "requests"
        assert args.inject_lib == "httpx"
        assert args.dry_run is False
        assert args.verbose is False

    def test_parser_optional_flags(self):
        parser = create_parser()

        args = parser.parse_args(
            [".", "requests", "httpx", "--dry-run", "--verbose"]
        )
        assert args.dry_run is True
        assert args.verbose is True

    def test_parser_help_output(self):
        parser = create_parser()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit):
                parser.parse_args(["--help"])

            help_output = fake_out.getvalue()
            assert "diversifier" in help_output
            assert "project_path" in help_output
            assert "remove_lib" in help_output
            assert "inject_lib" in help_output


class TestMainFunction:
    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_dry_run_success(self, mock_print, mock_validate_python):
        mock_validate_python.return_value = (True, [])

        test_args = [
            "diversifier",
            ".",
            "requests",
            "httpx",
            "--dry-run",
            "--verbose",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 0
        mock_validate_python.assert_called_once()

    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_invalid_project(self, mock_print, mock_validate_python):
        mock_validate_python.return_value = (
            False,
            ["No Python files found"],
        )

        test_args = ["diversifier", ".", "requests", "httpx"]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 1
        mock_print.assert_called()

    @patch("builtins.print")
    def test_main_nonexistent_path(self, mock_print):
        test_args = ["diversifier", "/nonexistent", "requests", "httpx"]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 1

    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_same_libraries(self, mock_print, mock_validate_python):
        mock_validate_python.return_value = (True, [])

        test_args = ["diversifier", ".", "requests", "requests"]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 1

    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_invalid_library_name(self, mock_print, mock_validate_python):
        mock_validate_python.return_value = (True, [])

        test_args = ["diversifier", ".", "requests", ""]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 1

    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_verbose_output(self, mock_print, mock_validate_python):
        mock_validate_python.return_value = (True, [])

        test_args = [
            "diversifier",
            ".",
            "requests",
            "httpx",
            "--verbose",
            "--dry-run",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 0

        print_calls = [call.args[0] for call in mock_print.call_args_list]
        verbose_output = any("Project path:" in call for call in print_calls)
        assert verbose_output
