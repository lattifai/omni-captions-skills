"""Tests for omnicaptions.services.download module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from omnicaptions.services.base import ErrorCode, ToolResult
from omnicaptions.services.download import (
    DownloadResult,
    download_media,
    is_video_platform_url,
    is_youtube_video_id,
)

# ---------------------------------------------------------------------------
# DownloadResult
# ---------------------------------------------------------------------------


def test_download_result_fields():
    """DownloadResult defaults should all be None/empty."""
    r = DownloadResult()
    assert r.audio_path is None
    assert r.video_path is None
    assert r.caption_path is None
    assert r.video_id == ""
    assert r.title == ""
    assert r.width is None
    assert r.height is None


def test_download_result_with_values():
    """DownloadResult should store provided values."""
    r = DownloadResult(
        audio_path=Path("/tmp/a.m4a"),
        video_id="abc123",
        title="Test",
        width=1920,
        height=1080,
    )
    assert r.audio_path == Path("/tmp/a.m4a")
    assert r.video_id == "abc123"
    assert r.width == 1920


# ---------------------------------------------------------------------------
# is_youtube_video_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("dQw4w9WgXcQ", True),  # classic 11-char ID
        ("abc-_123AbC", True),  # hyphens and underscores
        ("short", False),  # too short
        ("toolongvalue!!", False),  # too long / invalid chars
        ("https://youtube.com/watch?v=abc", False),  # full URL
        ("", False),
    ],
)
def test_is_youtube_video_id(value, expected):
    assert is_youtube_video_id(value) == expected


# ---------------------------------------------------------------------------
# is_video_platform_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("https://youtu.be/abc", True),
        ("https://www.bilibili.com/video/BV1xx", True),
        ("https://vimeo.com/123456", True),
        ("https://x.com/user/status/123", True),
        ("https://twitter.com/user/status/123", True),
        ("https://youtube.com/shorts/abc", True),
        ("https://example.com/page", False),
        ("not-a-url", False),
        ("", False),
    ],
)
def test_is_video_platform_url(url, expected):
    assert is_video_platform_url(url) == expected


# ---------------------------------------------------------------------------
# download_media
# ---------------------------------------------------------------------------


def test_download_media_invalid_url_empty():
    """Empty URL should return INVALID_URL error."""
    result = download_media("")
    assert isinstance(result, ToolResult)
    assert result.status == "error"
    assert result.error_code == ErrorCode.INVALID_URL


def test_download_media_invalid_url_whitespace():
    """Whitespace-only URL should return INVALID_URL error."""
    result = download_media("   ")
    assert isinstance(result, ToolResult)
    assert result.status == "error"
    assert result.error_code == ErrorCode.INVALID_URL


@patch("omnicaptions.services.download._download_with_ytdlp")
@patch("omnicaptions.services.download.resolve_video_input")
def test_download_media_returns_tool_result(mock_resolve, mock_ytdlp):
    """Successful download should return ToolResult with expected data keys."""
    mock_resolve.return_value = "https://www.youtube.com/watch?v=test123"
    mock_ytdlp.return_value = DownloadResult(
        audio_path=Path("/tmp/test123.m4a"),
        video_path=None,
        caption_path=Path("/tmp/test123.en.vtt"),
        video_id="test123",
        title="Test Video",
        width=1920,
        height=1080,
    )

    result = download_media("https://www.youtube.com/watch?v=test123", output_dir="/tmp")

    assert isinstance(result, ToolResult)
    assert result.status == "success"
    assert result.data["video_id"] == "test123"
    assert result.data["title"] == "Test Video"
    assert result.data["audio_path"] == "/tmp/test123.m4a"
    assert result.data["caption_path"] == "/tmp/test123.en.vtt"
    assert result.data["width"] == 1920
    assert result.data["height"] == 1080


@patch("omnicaptions.services.download._download_with_ytdlp")
@patch("omnicaptions.services.download.resolve_video_input")
def test_download_media_download_failed(mock_resolve, mock_ytdlp):
    """Exception in _download_with_ytdlp should return DOWNLOAD_FAILED."""
    mock_resolve.return_value = "https://www.youtube.com/watch?v=fail"
    mock_ytdlp.side_effect = RuntimeError("yt-dlp crashed")

    result = download_media("https://www.youtube.com/watch?v=fail")

    assert result.status == "error"
    assert result.error_code == ErrorCode.DOWNLOAD_FAILED
    assert "yt-dlp crashed" in result.message


@patch("omnicaptions.services.download.resolve_video_input")
def test_download_media_resolve_value_error(mock_resolve):
    """ValueError from resolve_video_input should return INVALID_URL."""
    mock_resolve.side_effect = ValueError("Video not found")

    result = download_media("badVideoId11")

    assert result.status == "error"
    assert result.error_code == ErrorCode.INVALID_URL
    assert "Video not found" in result.message
