"""Download service — extract and download media from video platforms.

Provides standalone functions (no class dependency) for downloading
audio/video and captions via yt-dlp.
"""

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional

from omnicaptions.services.base import ErrorCode, ToolResult

logger = logging.getLogger(__name__)


class DownloadResult(NamedTuple):
    """Result from yt-dlp download."""

    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    caption_path: Optional[Path] = None
    video_id: str = ""
    title: str = ""
    width: Optional[int] = None
    height: Optional[int] = None


# Video quality presets
VIDEO_QUALITY_FORMATS = {
    "best": "bestvideo+bestaudio/best",
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
    "audio": "bestaudio/best",
}

# Video platform URL patterns
VIDEO_PLATFORM_PATTERNS = [
    r"youtube\.com/watch",
    r"youtube\.com/shorts",
    r"youtu\.be/",
    r"bilibili\.com/video",
    r"vimeo\.com/",
    r"twitter\.com/.*/status",
    r"x\.com/.*/status",
]

# YouTube video ID pattern: 11 alphanumeric chars with hyphens and underscores
YOUTUBE_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def is_youtube_video_id(value: str) -> bool:
    """Check if a string looks like a bare YouTube video ID."""
    return bool(YOUTUBE_VIDEO_ID_RE.match(value))


def resolve_video_input(url_or_id: str) -> str:
    """Resolve a video URL or bare YouTube video ID to a full URL.

    If the input is a bare YouTube video ID, validates it via yt-dlp
    extract_info and converts to a full URL. Otherwise returns as-is.

    Raises:
        ValueError: If the video ID does not exist or is unavailable.
    """
    if not is_youtube_video_id(url_or_id):
        return url_or_id

    import yt_dlp

    full_url = f"https://www.youtube.com/watch?v={url_or_id}"
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(full_url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise ValueError(f"YouTube video '{url_or_id}' not found or unavailable: {e}") from e
        if not info:
            raise ValueError(f"YouTube video '{url_or_id}' not found or unavailable")

    logger.info("Resolved video ID '%s' -> %s (title: %s)", url_or_id, full_url, info.get("title", ""))
    return full_url


def is_video_platform_url(url: str) -> bool:
    """Check if URL is from a supported video platform."""
    for pattern in VIDEO_PLATFORM_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def _detect_audio_language(info: dict) -> Optional[str]:
    """Detect original audio language from video info.

    Checks multiple sources:
    1. Audio track language metadata
    2. Format language tags
    3. Requested format language
    """
    # Check audio formats for language
    formats = info.get("formats", [])
    for fmt in formats:
        if fmt.get("acodec") and fmt.get("acodec") != "none":
            lang = fmt.get("language")
            if lang:
                return lang

    # Check requested formats
    requested = info.get("requested_formats", [])
    for fmt in requested:
        lang = fmt.get("language")
        if lang:
            return lang

    return None


def _download_with_ytdlp(
    url: str,
    output_dir: Optional[Path] = None,
    quality: str = "audio",
    verbose: bool = False,
) -> DownloadResult:
    """Download audio/video and captions using yt-dlp.

    Downloads:
    - Audio/Video: Based on quality setting
    - Captions: Manual (user-uploaded) first, then auto-generated, original language

    Args:
        url: Video URL
        output_dir: Output directory (uses temp dir if None)
        quality: Quality preset - "audio", "best", "1080p", "720p", "480p", "360p"
        verbose: Whether to enable verbose logging

    Returns:
        DownloadResult with audio_path/video_path, caption_path, video_id, title
    """
    import yt_dlp

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "gemini-caption"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get format string for quality
    format_str = VIDEO_QUALITY_FORMATS.get(quality, VIDEO_QUALITY_FORMATS["audio"])
    is_video = quality != "audio"

    # Base options
    ydl_opts = {
        "format": format_str,
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": not verbose,
        "no_warnings": not verbose,
        # Captions: download if available
        "writesubtitles": True,
        "writeautomaticsub": True,  # Also try auto-generated
        "subtitlesformat": "vtt/srt/best",
        # Ignore caption download errors (e.g., 429 rate limit)
        "ignoreerrors": "only_download",
    }

    # Merge video+audio into single file for video downloads
    if is_video:
        ydl_opts["merge_output_format"] = "mp4"

    if verbose:
        logger.info(f"Downloading with yt-dlp ({quality}): {url}")

    # First pass: extract info to get available captions
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info.get("id", "video")
        title = info.get("title", "")

        # Extract video resolution
        video_width = info.get("width")
        video_height = info.get("height")
        if verbose and video_width and video_height:
            logger.info(f"Video resolution: {video_width}x{video_height}")

        # Detect original audio language from multiple sources
        original_lang = (
            info.get("language")  # Video metadata language
            or info.get("audio_language")  # Audio track language
            or _detect_audio_language(info)  # From audio tracks
            or "en"  # Default fallback
        )

        if verbose:
            logger.info(f"Detected original language: {original_lang}")

        # Determine best caption language
        subtitles = info.get("subtitles", {})  # Manual/uploaded
        auto_subs = info.get("automatic_captions", {})  # Auto-generated

        # Priority: original language first, then common languages
        preferred_langs = [original_lang]
        # Add language variants (e.g., en -> en-US, en-GB)
        for lang_code in list(subtitles.keys()) + list(auto_subs.keys()):
            if lang_code.startswith(original_lang):
                if lang_code not in preferred_langs:
                    preferred_langs.append(lang_code)
        # Add fallback languages
        for fallback in ["en", "zh", "ja", "ko"]:
            if fallback not in preferred_langs:
                preferred_langs.append(fallback)

        # Priority: manual original > auto original > manual other > auto other
        sub_lang = None
        sub_source = None

        # 1. Check manual captions for original language first
        for lang in preferred_langs[: len(preferred_langs)]:
            if lang in subtitles:
                sub_lang = lang
                sub_source = "manual"
                # Prefer original language, stop searching
                if lang.startswith(original_lang):
                    break

        # 2. Check auto-generated for original language
        if not sub_lang or not sub_lang.startswith(original_lang):
            for lang in preferred_langs:
                if lang in auto_subs and lang.startswith(original_lang):
                    sub_lang = lang
                    sub_source = "auto"
                    break

        # 3. Fall back to any available caption
        if not sub_lang:
            for lang in preferred_langs:
                if lang in subtitles:
                    sub_lang = lang
                    sub_source = "manual"
                    break
                if lang in auto_subs:
                    sub_lang = lang
                    sub_source = "auto"
                    break

        if sub_lang:
            ydl_opts["subtitleslangs"] = [sub_lang]
            if verbose:
                lang_match = (
                    "matches audio"
                    if sub_lang.startswith(original_lang)
                    else "different from audio"
                )
                logger.info(f"Selected {sub_source} caption: {sub_lang} ({lang_match})")
        else:
            # No captions available, disable caption download
            ydl_opts["writesubtitles"] = False
            ydl_opts["writeautomaticsub"] = False
            if verbose:
                logger.info("No captions available")

    # Second pass: download with determined options
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find downloaded files
    audio_path = None
    video_path = None
    caption_path = None

    audio_exts = (".mp3", ".m4a", ".mp3", ".opus", ".ogg", ".wav")
    video_exts = (".mp4", ".mkv", ".avi", ".mov", ".flv")
    caption_exts = (".vtt", ".srt", ".ass", ".srv3")

    for f in output_dir.iterdir():
        # Check if filename starts with video_id
        if f.name.startswith(video_id):
            if f.suffix in video_exts:
                video_path = f
            elif f.suffix in audio_exts:
                audio_path = f
            elif f.suffix in caption_exts:
                caption_path = f

    # For video downloads, we expect video_path; for audio, we expect audio_path
    if is_video and not video_path:
        raise RuntimeError(f"Failed to download video for {url}")
    if not is_video and not audio_path:
        raise RuntimeError(f"Failed to download audio for {url}")

    if verbose:
        if video_path:
            logger.info(f"Downloaded video: {video_path}")
        if audio_path:
            logger.info(f"Downloaded audio: {audio_path}")
        if caption_path:
            logger.info(f"Downloaded caption: {caption_path}")

    # Save metadata to .meta.json for later use (e.g., ASS font scaling, video_url tracking)
    meta_path = output_dir / f"{video_id}.meta.json"
    meta_data = {
        "video_id": video_id,
        "video_url": url,
        "title": title,
        "width": video_width,
        "height": video_height,
    }
    meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2))
    if verbose:
        logger.info(f"Saved metadata: {meta_path}")

    return DownloadResult(
        audio_path=audio_path,
        video_path=video_path,
        caption_path=caption_path,
        video_id=video_id,
        title=title,
        width=video_width,
        height=video_height,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_media(
    url: str,
    output_dir: Optional[str] = None,
    quality: str = "audio",
    verbose: bool = False,
) -> ToolResult:
    """Download media from a video platform URL.

    Args:
        url: Video URL or YouTube video ID.
        output_dir: Directory to save downloaded files. Uses a temp dir when None.
        quality: Quality preset - "audio", "best", "1080p", "720p", "480p", "360p".
        verbose: Enable verbose logging.

    Returns:
        ToolResult with data containing audio_path, video_path, caption_path,
        video_id, title, width, height on success.
    """
    # Validate URL
    if not url or not url.strip():
        return ToolResult(
            status="error",
            message="URL is required.",
            error_code=ErrorCode.INVALID_URL,
        )

    url = url.strip()

    # Resolve bare YouTube video IDs
    try:
        resolved_url = resolve_video_input(url)
    except ValueError as e:
        return ToolResult(
            status="error",
            message=str(e),
            error_code=ErrorCode.INVALID_URL,
        )

    # Download
    out_path = Path(output_dir) if output_dir else None
    try:
        result = _download_with_ytdlp(
            resolved_url,
            output_dir=out_path,
            quality=quality,
            verbose=verbose,
        )
    except Exception as e:
        return ToolResult(
            status="error",
            message=f"Download failed: {e}",
            error_code=ErrorCode.DOWNLOAD_FAILED,
        )

    return ToolResult(
        status="success",
        message=f"Downloaded: {result.title or resolved_url}",
        data={
            "audio_path": str(result.audio_path) if result.audio_path else None,
            "video_path": str(result.video_path) if result.video_path else None,
            "caption_path": str(result.caption_path) if result.caption_path else None,
            "video_id": result.video_id,
            "title": result.title,
            "width": result.width,
            "height": result.height,
        },
    )
