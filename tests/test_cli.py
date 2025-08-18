import pytest
import sys
from unittest.mock import patch, Mock, AsyncMock
from io import StringIO

from src.cli import (
    main,
    create_parser,
)
from src.orchestration.config import DiversifierConfig, LLMConfig


class TestCreateParser:
    def test_parser_creation(self):
        parser = create_parser()
        assert parser.prog == "diversifier"

    def test_parser_required_arguments(self):
        parser = create_parser()

        # Test that --config is required for normal operation
        args = parser.parse_args(["--create-config", "/test/config.toml"])
        assert args.create_config == "/test/config.toml"

        # Test normal operation with config
        args = parser.parse_args(
            ["--config", "/test/config.toml", ".", "requests", "httpx"]
        )
        assert args.config == "/test/config.toml"
        assert args.project_path == "."
        assert args.remove_lib == "requests"
        assert args.inject_lib == "httpx"
        assert args.dry_run is False
        assert args.verbose is False

    def test_parser_optional_flags(self):
        parser = create_parser()

        args = parser.parse_args(
            [
                "--config",
                "/test/config.toml",
                ".",
                "requests",
                "httpx",
                "--dry-run",
                "--verbose",
            ]
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
    @patch("src.cli.get_config")
    @patch("src.cli.DiversificationCoordinator")
    @patch("src.cli.validate_library_name")
    @patch("src.cli.validate_project_path")
    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_dry_run_success(
        self,
        mock_print,
        mock_validate_python,
        mock_validate_path,
        mock_validate_lib,
        mock_coordinator_class,
        mock_get_config,
    ):

        # Mock LLM config and migration config
        mock_llm_config = Mock(spec=LLMConfig)
        mock_migration_config = Mock()
        mock_migration_config.test_path = "tests/"
        mock_config = Mock(spec=DiversifierConfig)
        mock_config.llm = mock_llm_config
        mock_config.migration = mock_migration_config
        mock_get_config.return_value = mock_config

        mock_validate_python.return_value = (True, [])
        mock_validate_path.return_value = "/fake/path"
        mock_validate_lib.return_value = True
        mock_coordinator = Mock()
        # Mock the async execute_workflow method
        mock_coordinator.execute_workflow = AsyncMock(return_value=True)
        mock_coordinator_class.return_value = mock_coordinator

        test_args = [
            "diversifier",
            "--config",
            "/test/config.toml",
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

    @patch("src.cli.get_config")
    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_invalid_project(
        self, mock_print, mock_validate_python, mock_get_config
    ):

        # Mock LLM config and migration config
        mock_llm_config = Mock(spec=LLMConfig)
        mock_migration_config = Mock()
        mock_migration_config.test_path = "tests/"
        mock_config = Mock(spec=DiversifierConfig)
        mock_config.llm = mock_llm_config
        mock_config.migration = mock_migration_config
        mock_get_config.return_value = mock_config

        mock_validate_python.return_value = (
            False,
            ["No Python files found"],
        )

        test_args = [
            "diversifier",
            "--config",
            "/test/config.toml",
            ".",
            "requests",
            "httpx",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 1
        mock_print.assert_called()

    @patch("builtins.print")
    def test_main_nonexistent_path(self, mock_print):
        test_args = [
            "diversifier",
            "--config",
            "/test/config.toml",
            "/nonexistent",
            "requests",
            "httpx",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 1

    @patch("src.cli.get_config")
    @patch("src.cli.DiversificationCoordinator")
    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_same_libraries(
        self, mock_print, mock_validate_python, mock_coordinator_class, mock_get_config
    ):

        # Mock LLM config and migration config
        mock_llm_config = Mock(spec=LLMConfig)
        mock_migration_config = Mock()
        mock_migration_config.test_path = "tests/"
        mock_config = Mock(spec=DiversifierConfig)
        mock_config.llm = mock_llm_config
        mock_config.migration = mock_migration_config
        mock_get_config.return_value = mock_config

        mock_validate_python.return_value = (True, [])

        test_args = [
            "diversifier",
            "--config",
            "/test/config.toml",
            ".",
            "requests",
            "requests",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 1

    @patch("src.cli.get_config")
    @patch("src.cli.DiversificationCoordinator")
    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_invalid_library_name(
        self, mock_print, mock_validate_python, mock_coordinator_class, mock_get_config
    ):

        # Mock LLM config and migration config
        mock_llm_config = Mock(spec=LLMConfig)
        mock_migration_config = Mock()
        mock_migration_config.test_path = "tests/"
        mock_config = Mock(spec=DiversifierConfig)
        mock_config.llm = mock_llm_config
        mock_config.migration = mock_migration_config
        mock_get_config.return_value = mock_config

        mock_validate_python.return_value = (True, [])

        test_args = [
            "diversifier",
            "--config",
            "/test/config.toml",
            ".",
            "requests",
            "",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()

        assert result == 1

    @patch("src.cli.get_config")
    @patch("src.cli.DiversificationCoordinator")
    @patch("src.cli.validate_library_name")
    @patch("src.cli.validate_project_path")
    @patch("src.cli.validate_python_project")
    @patch("builtins.print")
    def test_main_verbose_output(
        self,
        mock_print,
        mock_validate_python,
        mock_validate_path,
        mock_validate_lib,
        mock_coordinator_class,
        mock_get_config,
    ):

        # Mock LLM config and migration config
        mock_llm_config = Mock(spec=LLMConfig)
        mock_migration_config = Mock()
        mock_migration_config.test_path = "tests/"
        mock_config = Mock(spec=DiversifierConfig)
        mock_config.llm = mock_llm_config
        mock_config.migration = mock_migration_config
        mock_get_config.return_value = mock_config

        mock_validate_python.return_value = (True, [])
        mock_validate_path.return_value = "/fake/path"
        mock_validate_lib.return_value = True
        mock_coordinator = Mock()
        # Mock the async execute_workflow method
        mock_coordinator.execute_workflow = AsyncMock(return_value=True)
        mock_coordinator_class.return_value = mock_coordinator

        test_args = [
            "diversifier",
            "--config",
            "/test/config.toml",
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
