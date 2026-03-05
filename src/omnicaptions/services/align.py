"""Align service -- forced alignment via LattifAI API.

Provides standalone functions for aligning audio with caption files
using the LattifAI Lattice-1 model.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from omnicaptions.config import LATTIFAI_API_KEY_URL, get_lattifai_api_key
from omnicaptions.services.base import ErrorCode, ToolResult
from omnicaptions.services.download import download_media, is_youtube_video_id

logger = logging.getLogger(__name__)


def _detect_metadata(audio_path: Path) -> dict | None:
    """Detect metadata from .meta.json in the same directory.

    Looks for ``<stem>.meta.json`` next to the audio file (e.g.
    ``e882eXLtwkI.m4a`` -> ``e882eXLtwkI.meta.json``).
    """
    meta_path = audio_path.parent / f"{audio_path.stem}.meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            pass
    return None


def forced_align(
    audio: str,
    caption: str,
    output: Optional[str] = None,
    output_format: str = "json",
    api_key: Optional[str] = None,
    split_sentence: bool = False,
    verbose: bool = False,
) -> ToolResult:
    """Align audio with a caption file using LattifAI forced alignment.

    Args:
        audio: Path to the audio file.
        caption: Path to the caption/subtitle file.
        output: Output file path. Auto-generated if None.
        output_format: Output format (default: "json").
        api_key: LattifAI API key. Falls back to config if None.
        split_sentence: Whether to split sentences.
        verbose: Enable verbose logging.

    Returns:
        ToolResult with data containing output_path, output_format,
        word_level.
    """
    # Resolve API key
    resolved_key = api_key or get_lattifai_api_key()
    if not resolved_key:
        return ToolResult(
            status="error",
            message=f"LattifAI API key not found. Get one at {LATTIFAI_API_KEY_URL}",
            error_code=ErrorCode.MISSING_API_KEY,
        )

    # Check if audio is a URL or YouTube video ID
    is_url = str(audio).startswith(("http://", "https://"))
    is_vid_id = is_youtube_video_id(str(audio))

    if is_url or is_vid_id:
        # Download media first
        import tempfile

        dl_dir = tempfile.mkdtemp(prefix="lattifai_")
        dl_result = download_media(
            url=str(audio),
            output_dir=dl_dir,
            quality="audio",
            verbose=verbose,
        )
        if dl_result.status == "error":
            return ToolResult(
                status="error",
                message=f"Failed to download audio: {dl_result.message}",
                error_code=ErrorCode.DOWNLOAD_FAILED,
            )
        # Use downloaded audio path
        audio = dl_result.data.get("audio_path") or dl_result.data.get("video_path")
        if not audio:
            return ToolResult(
                status="error",
                message="Download succeeded but no audio/video file found",
                error_code=ErrorCode.DOWNLOAD_FAILED,
            )

    # Validate input files
    audio_path = Path(audio)
    if not audio_path.exists():
        return ToolResult(
            status="error",
            message=f"Audio file not found: {audio}",
            error_code=ErrorCode.INVALID_INPUT,
        )

    caption_path = Path(caption)
    if not caption_path.exists():
        return ToolResult(
            status="error",
            message=f"Caption file not found: {caption}",
            error_code=ErrorCode.INVALID_INPUT,
        )

    # Auto-detect metadata from .meta.json
    metadata = _detect_metadata(audio_path)

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        suffix = f".{output_format}"
        output_path = caption_path.with_stem(f"{caption_path.stem}_LaiCut").with_suffix(suffix)

    # Enable word_level for JSON output to preserve word timing
    is_json_output = str(output_path).lower().endswith(".json")
    word_level = is_json_output

    if verbose and is_json_output:
        logger.info("JSON output: word-level timing enabled")

    # Import lattifai (optional dependency)
    try:
        from lattifai.client import LattifAI
        from lattifai.config import (
            AlignmentConfig,
            CaptionConfig,
            ClientConfig,
            EventConfig,
        )
    except ImportError:
        return ToolResult(
            status="error",
            message=(
                "lattifai package not installed. Install with: pip install omnicaptions[alignment]"
            ),
            error_code=ErrorCode.MISSING_DEPENDENCY,
        )

    # Run alignment
    try:
        client = LattifAI(
            client_config=ClientConfig(api_key=resolved_key),
            alignment_config=AlignmentConfig(
                model_name="LattifAI/Lattice-1", model_hub="modelscope"
            ),
            caption_config=CaptionConfig(
                output_path=str(output_path),
                split_sentence=split_sentence,
                word_level=word_level,
            ),
            event_config=EventConfig(enabled=True),
        )

        client.alignment(
            input_media=str(audio_path),
            input_caption=str(caption_path),
            output_caption_path=str(output_path),
            metadata=metadata,
        )
    except Exception as e:
        return ToolResult(
            status="error",
            message=f"Alignment failed: {e}",
            error_code=ErrorCode.ALIGNMENT_FAILED,
        )

    return ToolResult(
        status="success",
        message=f"Alignment saved to {output_path}",
        data={
            "output_path": str(output_path),
            "output_format": output_format,
            "word_level": word_level,
        },
    )
