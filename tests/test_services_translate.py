"""Tests for omnicaptions.services.translate module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnicaptions.services.base import ErrorCode, ToolResult
from omnicaptions.services.translate import (
    LANGUAGE_NAMES,
    _build_translate_prompt,
    translate_caption,
)


# ---------------------------------------------------------------------------
# API key / file validation
# ---------------------------------------------------------------------------


@patch("omnicaptions.services.translate.get_gemini_api_key", return_value=None)
def test_translate_missing_api_key(mock_key):
    """Missing API key should return MISSING_API_KEY error."""
    result = translate_caption("some_file.srt", target_lang="zh")
    assert isinstance(result, ToolResult)
    assert result.status == "error"
    assert result.error_code == ErrorCode.MISSING_API_KEY
    assert "aistudio.google.com" in result.message


@patch("omnicaptions.services.translate.get_gemini_api_key", return_value="fake-key")
def test_translate_file_not_found(mock_key):
    """Non-existent file should return INVALID_INPUT error."""
    result = translate_caption("/nonexistent/path/file.srt", target_lang="ja")
    assert result.status == "error"
    assert result.error_code == ErrorCode.INVALID_INPUT
    assert "File not found" in result.message


# ---------------------------------------------------------------------------
# _build_translate_prompt
# ---------------------------------------------------------------------------


def test_build_translate_prompt_monolingual():
    """Monolingual prompt should request JSON array of translated strings."""
    prompt = _build_translate_prompt(
        texts=["Hello world", "Good morning"],
        target_lang="zh",
        bilingual=False,
    )
    assert "Chinese (Simplified)" in prompt
    assert "JSON array of translated strings" in prompt
    assert '"to_translate"' in prompt
    # Verify the texts appear in the JSON payload
    assert "Hello world" in prompt
    assert "Good morning" in prompt
    # Should NOT mention "original" and "translated" keys
    assert '"original" and "translated"' not in prompt


def test_build_translate_prompt_bilingual():
    """Bilingual prompt should request JSON array with original+translated keys."""
    prompt = _build_translate_prompt(
        texts=["Hello"],
        target_lang="ja",
        bilingual=True,
    )
    assert "Japanese" in prompt
    assert '"original" and "translated"' in prompt


def test_build_translate_prompt_with_context():
    """Context lines should appear in the prompt JSON under context keys."""
    prompt = _build_translate_prompt(
        texts=["Main line"],
        target_lang="en",
        bilingual=False,
        context_before=["Before line 1", "Before line 2"],
        context_after=["After line 1"],
    )
    assert "context_before" in prompt
    assert "context_after" in prompt
    assert "Before line 1" in prompt
    assert "After line 1" in prompt

    # Without context, those keys should be absent
    prompt_no_ctx = _build_translate_prompt(
        texts=["Main line"],
        target_lang="en",
        bilingual=False,
    )
    assert "context_before" not in prompt_no_ctx
    assert "context_after" not in prompt_no_ctx


# ---------------------------------------------------------------------------
# translate_caption — success path
# ---------------------------------------------------------------------------


@patch("omnicaptions.services.translate.get_gemini_api_key", return_value="fake-key")
@patch("omnicaptions.services.translate._translate_batch_async")
def test_translate_returns_tool_result(mock_batch, mock_key, tmp_path):
    """Successful translation should return ToolResult with correct data."""
    # Create a temp SRT file from the SA1.vtt content
    srt_content = (
        "1\n"
        "00:00:00,480 --> 00:00:03,980\n"
        "She had your dark suit in greasy wash water all year.\n"
    )
    srt_file = tmp_path / "SA1.srt"
    srt_file.write_text(srt_content, encoding="utf-8")

    # Mock the async batch translate to return a translated string
    import asyncio

    async def fake_batch(*args, **kwargs):
        return ["Elle avait votre costume sombre dans l'eau de lavage grasse toute l'annee."]

    mock_batch.side_effect = fake_batch

    result = translate_caption(
        input_file=str(srt_file),
        target_lang="fr",
        verbose=False,
    )

    assert result.status == "success"
    assert result.error_code is None
    assert result.data["segment_count"] == 1
    assert result.data["target_lang"] == "fr"
    assert result.data["bilingual"] is False
    assert "output_path" in result.data

    # Verify auto-generated output path pattern
    out = Path(result.data["output_path"])
    assert "Gemini_French" in out.name
    assert out.suffix == ".srt"

    # Verify file was written
    assert out.exists()


# ---------------------------------------------------------------------------
# LANGUAGE_NAMES
# ---------------------------------------------------------------------------


def test_language_names_has_common_langs():
    """LANGUAGE_NAMES should include commonly used language codes."""
    expected = {"zh", "en", "ja", "ko", "es", "fr", "de"}
    assert expected.issubset(set(LANGUAGE_NAMES.keys()))
    # Verify values are human-readable
    assert LANGUAGE_NAMES["zh"] == "Chinese (Simplified)"
    assert LANGUAGE_NAMES["en"] == "English"
