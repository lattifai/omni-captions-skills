"""Tests for omnicaptions.services.transcribe module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnicaptions.services.base import ErrorCode, ToolResult
from omnicaptions.services.transcribe import (
    get_transcription_prompt,
    transcribe_media,
)


# ---------------------------------------------------------------------------
# get_transcription_prompt
# ---------------------------------------------------------------------------


def test_get_transcription_prompt_default():
    """Default prompt should load from the prompt file."""
    prompt = get_transcription_prompt()
    # The file exists and contains real content
    assert len(prompt) > 50
    assert isinstance(prompt, str)


def test_get_transcription_prompt_custom():
    """Custom prompt should override the default."""
    custom = "My custom transcription instructions."
    prompt = get_transcription_prompt(custom_prompt=custom)
    assert prompt == custom


def test_get_transcription_prompt_with_language():
    """Language hint should be appended to the prompt."""
    prompt = get_transcription_prompt(language="Japanese")
    assert "Japanese" in prompt
    assert "Use Japanese language for transcription" in prompt


def test_get_transcription_prompt_custom_with_language():
    """Custom prompt + language should combine correctly."""
    prompt = get_transcription_prompt(language="French", custom_prompt="Base prompt.")
    assert prompt.startswith("Base prompt.")
    assert "French" in prompt


# ---------------------------------------------------------------------------
# transcribe_media — API key checks
# ---------------------------------------------------------------------------


@patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value=None)
def test_transcribe_missing_api_key(mock_key):
    """Missing API key should return MISSING_API_KEY error."""
    result = transcribe_media("some_file.mp3")
    assert isinstance(result, ToolResult)
    assert result.status == "error"
    assert result.error_code == ErrorCode.MISSING_API_KEY
    assert "aistudio.google.com" in result.message


# ---------------------------------------------------------------------------
# transcribe_media — file validation
# ---------------------------------------------------------------------------


@patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key")
def test_transcribe_file_not_found(mock_key):
    """Non-existent local file should return INVALID_INPUT error."""
    result = transcribe_media("/nonexistent/path/audio.wav")
    assert result.status == "error"
    assert result.error_code == ErrorCode.INVALID_INPUT
    assert "File not found" in result.message


@patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key")
@patch("omnicaptions.services.transcribe._run_transcription")
def test_transcribe_url_input_accepted(mock_run, mock_key):
    """URL input should skip file-existence check (not fail with INVALID_INPUT)."""
    mock_run.return_value = "Transcribed text from URL."

    result = transcribe_media("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert result.status == "success"
    assert result.error_code is None
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# transcribe_media — transcription failure
# ---------------------------------------------------------------------------


@patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key")
@patch("omnicaptions.services.transcribe._run_transcription")
def test_transcribe_exception_returns_error(mock_run, mock_key):
    """Exception during transcription should return TRANSCRIPTION_FAILED."""
    mock_run.side_effect = RuntimeError("API exploded")

    result = transcribe_media("https://example.com/video.mp4")

    assert result.status == "error"
    assert result.error_code == ErrorCode.TRANSCRIPTION_FAILED
    assert "API exploded" in result.message


# ---------------------------------------------------------------------------
# transcribe_media — success path
# ---------------------------------------------------------------------------


@patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key")
@patch("omnicaptions.services.transcribe._run_transcription")
def test_transcribe_success_auto_output(mock_run, mock_key, tmp_path):
    """Successful transcription should auto-generate output path and truncate text."""
    long_text = "A" * 1000
    mock_run.return_value = long_text

    # Use a URL so file-existence check is skipped
    result = transcribe_media(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )

    assert result.status == "success"
    assert result.data["char_count"] == 1000
    # Text in data should be truncated to 500
    assert len(result.data["text"]) == 500
    assert "dQw4w9WgXcQ_GeminiUnd.md" in result.data["output_path"]

    # Clean up
    Path(result.data["output_path"]).unlink(missing_ok=True)


@patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key")
@patch("omnicaptions.services.transcribe._run_transcription")
def test_transcribe_success_explicit_output(mock_run, mock_key, tmp_path):
    """Explicit output path should be used as-is."""
    mock_run.return_value = "Hello world"
    out = str(tmp_path / "result.md")

    result = transcribe_media("https://example.com/audio.mp3", output=out)

    assert result.status == "success"
    assert result.data["output_path"] == out
    assert Path(out).read_text() == "Hello world"
