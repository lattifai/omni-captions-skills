"""Tests for CLI entry point and help commands."""

import subprocess
import sys

import pytest


class TestCLIEntrypoint:
    """Test CLI entry point is correctly installed."""

    def test_omnicaptions_help(self):
        """Test `omnicaptions --help` works."""
        result = subprocess.run(
            [sys.executable, "-m", "omnicaptions", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "omnicaptions" in result.stdout.lower()
        assert "transcribe" in result.stdout
        assert "convert" in result.stdout
        assert "translate" in result.stdout
        assert "download" in result.stdout
        assert "LaiCut" in result.stdout

    def test_transcribe_help(self):
        """Test `omnicaptions transcribe --help` works."""
        result = subprocess.run(
            [sys.executable, "-m", "omnicaptions", "transcribe", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--api-key" in result.stdout or "-k" in result.stdout
        assert "--model" in result.stdout or "-m" in result.stdout
        assert "--language" in result.stdout or "-l" in result.stdout
        assert "--translate" in result.stdout or "-t" in result.stdout
        assert "--bilingual" in result.stdout

    def test_convert_help(self):
        """Test `omnicaptions convert --help` works."""
        result = subprocess.run(
            [sys.executable, "-m", "omnicaptions", "convert", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--from" in result.stdout or "-f" in result.stdout
        assert "--to" in result.stdout or "-t" in result.stdout
        assert "--output" in result.stdout or "-o" in result.stdout

    def test_translate_help(self):
        """Test `omnicaptions translate --help` works."""
        result = subprocess.run(
            [sys.executable, "-m", "omnicaptions", "translate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--language" in result.stdout or "-l" in result.stdout
        assert "--bilingual" in result.stdout
        assert "--api-key" in result.stdout or "-k" in result.stdout

    def test_download_help(self):
        """Test `omnicaptions download --help` works."""
        result = subprocess.run(
            [sys.executable, "-m", "omnicaptions", "download", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--quality" in result.stdout or "-q" in result.stdout
        assert "--output" in result.stdout or "-o" in result.stdout

    def test_laicut_help(self):
        """Test `omnicaptions LaiCut --help` works."""
        result = subprocess.run(
            [sys.executable, "-m", "omnicaptions", "LaiCut", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "audio" in result.stdout
        assert "caption" in result.stdout
        assert "--api-key" in result.stdout or "-k" in result.stdout

    def test_no_command_shows_help(self):
        """Test running without command shows error."""
        result = subprocess.run(
            [sys.executable, "-m", "omnicaptions"],
            capture_output=True,
            text=True,
        )
        # Should fail without command
        assert result.returncode != 0
        # Should mention available commands
        assert "transcribe" in result.stderr or "convert" in result.stderr

    def test_invalid_command_error(self):
        """Test invalid command shows error."""
        result = subprocess.run(
            [sys.executable, "-m", "omnicaptions", "nonexistent"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestCLIScript:
    """Test CLI script entry point (installed as `omnicaptions` command)."""

    def test_script_entrypoint_exists(self):
        """Test that the script entry point is defined in pyproject.toml."""
        from importlib.metadata import entry_points

        eps = entry_points()
        if hasattr(eps, "select"):
            # Python 3.10+
            scripts = eps.select(group="console_scripts")
        else:
            # Python 3.9
            scripts = eps.get("console_scripts", [])

        script_names = [ep.name for ep in scripts]
        assert "omnicaptions" in script_names

    def test_main_callable(self):
        """Test main function is callable."""
        from omnicaptions.cli import main

        assert callable(main)
