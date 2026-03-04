"""Transcribe service — media transcription via Gemini API.

Provides standalone functions (no class dependency) for transcribing
audio/video files and URLs using the Gemini generative API.
"""

import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional, Union

from omnicaptions.config import API_KEY_URL, get_gemini_api_key
from omnicaptions.services.base import ErrorCode, ToolResult
from omnicaptions.services.download import is_youtube_video_id

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"

# YouTube URL patterns for extracting video ID
_YT_URL_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"
)


def get_transcription_prompt(
    language: Optional[str] = None,
    custom_prompt: Optional[str] = None,
) -> str:
    """Build the transcription system prompt.

    Args:
        language: Optional language hint appended to the prompt.
        custom_prompt: If provided, replaces the default prompt file entirely.

    Returns:
        The fully assembled prompt string.
    """
    if custom_prompt is not None:
        base_prompt = custom_prompt
    else:
        prompt_path = Path(__file__).parent.parent / "prompts" / "transcription_dotey.md"
        if prompt_path.exists():
            base_prompt = prompt_path.read_text(encoding="utf-8").strip()
        else:
            logger.warning("Prompt file not found: %s, using default prompt", prompt_path)
            base_prompt = (
                "You are an expert transcript specialist.\n"
                "Transcribe the audio/video content verbatim with timestamps "
                "in [HH:MM:SS] format.\n"
                "Include speaker labels when multiple speakers are present."
            )

    if language:
        base_prompt += f"\n\n* Use {language} language for transcription."

    return base_prompt


async def _transcribe_async(
    url_or_path: Union[str, Path],
    api_key: str,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    use_thinking: bool = False,
    custom_prompt: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Transcribe audio/video from URL or local file (async).

    Args:
        url_or_path: URL or local file path to transcribe.
        api_key: Gemini API key.
        model: Model name to use.
        language: Optional language hint.
        use_thinking: Whether to enable thinking mode.
        custom_prompt: Custom system prompt override.
        verbose: Enable verbose logging.

    Returns:
        Transcription text.
    """
    from google.genai import Client
    from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

    client = Client(api_key=api_key)
    prompt = get_transcription_prompt(language=language, custom_prompt=custom_prompt)

    config_kwargs = {
        "system_instruction": prompt,
        "response_modalities": ["TEXT"],
    }
    if use_thinking:
        config_kwargs["thinking_config"] = ThinkingConfig(
            include_thoughts=False, thinking_budget=-1
        )

    gen_config = GenerateContentConfig(**config_kwargs)
    url_or_path_str = str(url_or_path)

    if url_or_path_str.startswith(("http://", "https://")):
        if verbose:
            logger.info("Transcribing URL: %s", url_or_path_str)
        contents = Part.from_uri(file_uri=url_or_path_str, mime_type="video/*")
    else:
        path = Path(url_or_path)
        if verbose:
            logger.info("Transcribing local file: %s", path)

        # Handle non-ASCII filenames (Google GenAI SDK requires ASCII in headers)
        try:
            str(path).encode("ascii")
            upload_path = path
            temp_link = None
        except UnicodeEncodeError:
            temp_link = Path(tempfile.gettempdir()) / f"upload_{path.suffix}"
            temp_link.unlink(missing_ok=True)
            temp_link.symlink_to(path.resolve())
            upload_path = temp_link
            if verbose:
                logger.info("Using temp link for non-ASCII filename: %s", temp_link)

        try:
            uploaded = client.files.upload(file=str(upload_path))
            contents = Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
        finally:
            if temp_link:
                temp_link.unlink(missing_ok=True)

    if verbose:
        logger.info("Transcribing with %s...", model)

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model,
            contents=contents,
            config=gen_config,
        ),
    )

    if not response.text:
        raise RuntimeError("Empty response from Gemini API")

    if verbose:
        logger.info("Transcription completed: %d chars", len(response.text))

    return response.text.strip()


def _run_transcription(
    url_or_path: Union[str, Path],
    api_key: str,
    model: str = DEFAULT_MODEL,
    **kwargs,
) -> str:
    """Synchronous wrapper around _transcribe_async.

    Args:
        url_or_path: URL or local file path.
        api_key: Gemini API key.
        model: Model name.
        **kwargs: Forwarded to _transcribe_async (language, use_thinking, etc.).

    Returns:
        Transcription text.
    """
    return asyncio.run(
        _transcribe_async(url_or_path, api_key=api_key, model=model, **kwargs)
    )


def _generate_output_path(input_path: str) -> Path:
    """Generate an output path for the transcription result.

    For URLs, extracts a YouTube video ID (or uses 'transcription').
    For local files, uses the input file stem.
    """
    if input_path.startswith(("http://", "https://")):
        # Try to extract YouTube video ID from URL
        m = _YT_URL_RE.search(input_path)
        if m:
            stem = m.group(1)
        else:
            stem = "transcription"
    elif is_youtube_video_id(input_path):
        stem = input_path
    else:
        stem = Path(input_path).stem

    return Path(f"{stem}_GeminiUnd.md")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe_media(
    input_path: str,
    output: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    use_thinking: bool = False,
    custom_prompt: Optional[str] = None,
    verbose: bool = False,
) -> ToolResult:
    """Transcribe audio/video and save the result.

    Args:
        input_path: URL or local file path to transcribe.
        output: Output file path. Auto-generated if None.
        api_key: Gemini API key. Falls back to config if None.
        model: Gemini model name.
        language: Optional language hint for transcription.
        use_thinking: Enable thinking mode.
        custom_prompt: Custom system prompt override.
        verbose: Enable verbose logging.

    Returns:
        ToolResult with data containing output_path, text (truncated), char_count.
    """
    # Resolve API key
    resolved_key = api_key or get_gemini_api_key()
    if not resolved_key:
        return ToolResult(
            status="error",
            message=f"Gemini API key not found. Get one at {API_KEY_URL}",
            error_code=ErrorCode.MISSING_API_KEY,
        )

    # Validate local file exists (skip for URLs)
    is_url = input_path.startswith(("http://", "https://"))
    is_vid_id = is_youtube_video_id(input_path)
    if not is_url and not is_vid_id:
        if not Path(input_path).exists():
            return ToolResult(
                status="error",
                message=f"File not found: {input_path}",
                error_code=ErrorCode.INVALID_INPUT,
            )

    # Run transcription
    try:
        text = _run_transcription(
            input_path,
            api_key=resolved_key,
            model=model,
            language=language,
            use_thinking=use_thinking,
            custom_prompt=custom_prompt,
            verbose=verbose,
        )
    except Exception as e:
        return ToolResult(
            status="error",
            message=f"Transcription failed: {e}",
            error_code=ErrorCode.TRANSCRIPTION_FAILED,
        )

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = _generate_output_path(input_path)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    return ToolResult(
        status="success",
        message=f"Transcription saved to {output_path}",
        data={
            "output_path": str(output_path),
            "text": text[:500],
            "char_count": len(text),
        },
    )
