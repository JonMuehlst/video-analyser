import pytest
import os
import sys
from unittest.mock import patch, MagicMock

from smolavision.cli.main import main
from smolavision.cli.commands import run_analysis, check_dependencies, setup_ollama


class TestCLI:
    """Integration tests for the command line interface."""

    @patch('smolavision.cli.commands.create_pipeline')
    @patch('smolavision.cli.commands.validate_config')
    @patch('smolavision.cli.commands.load_config')
    @patch('smolavision.cli.parser.parse_args')
    def test_run_analysis_command(self, mock_parse_args, mock_load_config, 
                                 mock_validate_config, mock_create_pipeline,
                                 temp_output_dir):
        """Test the run_analysis command."""
        # Mock arguments
        mock_args = MagicMock()
        mock_args.video = "test_video.mp4"
        mock_args.config = None
        mock_parse_args.return_value = mock_args
        
        # Mock config validation
        mock_validate_config.return_value = (True, [])
        
        # Mock config
        mock_config = {
            "output_dir": temp_output_dir,
            "video": {},
            "model": {},
            "analysis": {}
        }
        mock_load_config.return_value = mock_config
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "summary_text": "Video summary",
            "analyses": ["Analysis 1", "Analysis 2"],
            "summary_path": os.path.join(temp_output_dir, "summary.txt"),
            "full_analysis_path": os.path.join(temp_output_dir, "full_analysis.txt"),
            "output_dir": temp_output_dir
        }
        mock_create_pipeline.return_value = mock_pipeline
        
        # Run the command
        with patch('smolavision.cli.commands.print_results'):
            exit_code = run_analysis(mock_args)
        
        # Verify results
        assert exit_code == 0
        mock_load_config.assert_called_once()
        mock_validate_config.assert_called_once()
        mock_create_pipeline.assert_called_once()
        mock_pipeline.run.assert_called_once_with("test_video.mp4")

    @patch('smolavision.cli.commands.check_dependencies')
    @patch('smolavision.cli.parser.parse_args')
    def test_check_dependencies_command(self, mock_parse_args, mock_check_deps):
        """Test the check_dependencies command."""
        # Mock arguments
        mock_args = MagicMock()
        mock_args.command = "check"
        mock_parse_args.return_value = mock_args
        
        # Mock dependencies check
        mock_check_deps.return_value = []  # No missing dependencies
        
        # Run the command
        with patch('builtins.print'):
            exit_code = check_dependencies(mock_args)
        
        # Verify results
        assert exit_code == 0
        mock_check_deps.assert_called_once()

    @patch('smolavision.cli.commands.setup_ollama_models')
    @patch('smolavision.cli.parser.parse_args')
    def test_setup_ollama_command(self, mock_parse_args, mock_setup_ollama):
        """Test the setup_ollama command."""
        # Mock arguments
        mock_args = MagicMock()
        mock_args.command = "setup-ollama"
        mock_args.models = "llama3,llava"
        mock_args.ollama_base_url = "http://localhost:11434"
        mock_parse_args.return_value = mock_args
        
        # Mock setup
        mock_setup_ollama.return_value = True
        
        # Run the command
        with patch('builtins.print'):
            exit_code = setup_ollama(mock_args)
        
        # Verify results
        assert exit_code == 0
        mock_setup_ollama.assert_called_once_with(
            models=["llama3", "llava"],
            base_url="http://localhost:11434"
        )

    @patch('smolavision.cli.main.run_analysis')
    @patch('smolavision.cli.parser.parse_args')
    def test_main_function(self, mock_parse_args, mock_run_analysis):
        """Test the main function."""
        # Mock arguments
        mock_args = MagicMock()
        mock_args.command = None  # Default to run_analysis
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        # Mock run_analysis
        mock_run_analysis.return_value = 0
        
        # Run main
        with patch('smolavision.cli.main.setup_logging'):
            exit_code = main()
        
        # Verify results
        assert exit_code == 0
        mock_run_analysis.assert_called_once()
