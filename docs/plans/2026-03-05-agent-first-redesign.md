# Agent-First Skills & CLI Redesign

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor omni-captions-skills into a layered architecture (services → CLI → skills) with agent-friendly JSON output and a unified pipeline skill.

**Architecture:** Extract business logic from GeminiCaption class and cli.py into 5 independent service modules. CLI becomes a thin adapter with `--json` output. Skills collapse from 5 to 3 (1 pipeline + 2 specialized). Backward compatibility preserved via re-exports.

**Tech Stack:** Python 3.10+, google-genai, lattifai-captions, yt-dlp, pytest

---

### Task 1: Create ToolResult base module

**Files:**
- Create: `src/omnicaptions/services/__init__.py`
- Create: `src/omnicaptions/services/base.py`
- Test: `tests/test_services_base.py`

**Step 1: Write the failing test**

```python
# tests/test_services_base.py
"""Tests for ToolResult and ErrorCode."""

from omnicaptions.services.base import ErrorCode, ToolResult


def test_tool_result_success():
    r = ToolResult(status="success", message="done", data={"path": "/tmp/out.srt"})
    assert r.status == "success"
    assert r.error_code is None
    d = r.to_json()
    assert d["status"] == "success"
    assert d["error_code"] is None
    assert d["data"]["path"] == "/tmp/out.srt"


def test_tool_result_error():
    r = ToolResult(
        status="error",
        message="Key missing",
        error_code=ErrorCode.MISSING_API_KEY,
    )
    d = r.to_json()
    assert d["status"] == "error"
    assert d["error_code"] == "MISSING_API_KEY"
    assert d["data"] == {}


def test_error_code_values():
    assert ErrorCode.MISSING_API_KEY.value == "MISSING_API_KEY"
    assert ErrorCode.NETWORK_TIMEOUT.value == "NETWORK_TIMEOUT"
    assert ErrorCode.UNSUPPORTED_FORMAT.value == "UNSUPPORTED_FORMAT"
```

**Step 2: Run test to verify it fails**

Run: `cd omni-captions-skills && python -m pytest tests/test_services_base.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'omnicaptions.services'"

**Step 3: Write minimal implementation**

```python
# src/omnicaptions/services/__init__.py
"""Services layer - pure business logic with structured results."""

from .base import ErrorCode, ToolResult

__all__ = ["ErrorCode", "ToolResult"]
```

```python
# src/omnicaptions/services/base.py
"""Base types for services layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    MISSING_API_KEY = "MISSING_API_KEY"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_URL = "INVALID_URL"
    INVALID_INPUT = "INVALID_INPUT"
    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    TRANSLATION_FAILED = "TRANSLATION_FAILED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    CONVERSION_FAILED = "CONVERSION_FAILED"


@dataclass
class ToolResult:
    status: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error_code: ErrorCode | None = None

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "error_code": self.error_code.value if self.error_code else None,
            "message": self.message,
            "data": self.data,
        }
```

**Step 4: Run test to verify it passes**

Run: `cd omni-captions-skills && python -m pytest tests/test_services_base.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/omnicaptions/services/__init__.py src/omnicaptions/services/base.py tests/test_services_base.py
git commit -m "feat: add ToolResult and ErrorCode base types for services layer"
```

---

### Task 2: Extract download service

**Files:**
- Create: `src/omnicaptions/services/download.py`
- Test: `tests/test_services_download.py`
- Reference: `src/omnicaptions/caption.py:210-404` (GeminiCaption._download_with_ytdlp)

**Step 1: Write the failing test**

```python
# tests/test_services_download.py
"""Tests for download service."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from omnicaptions.services.download import download_media, DownloadResult


def test_download_result_fields():
    r = DownloadResult()
    assert r.audio_path is None
    assert r.video_path is None
    assert r.caption_path is None
    assert r.video_id == ""
    assert r.title == ""


def test_download_media_invalid_url():
    result = download_media("")
    assert result.status == "error"
    assert result.error_code is not None


def test_download_media_returns_tool_result():
    """Verify download_media returns ToolResult with correct data keys."""
    with patch("omnicaptions.services.download._download_with_ytdlp") as mock_dl:
        mock_dl.return_value = DownloadResult(
            audio_path=Path("/tmp/test.m4a"),
            video_id="abc123",
            title="Test Video",
        )
        result = download_media("https://youtube.com/watch?v=abc123", output_dir=Path(tempfile.gettempdir()))
        assert result.status == "success"
        assert "audio_path" in result.data
        assert result.data["video_id"] == "abc123"
        assert result.data["title"] == "Test Video"
```

**Step 2: Run test to verify it fails**

Run: `cd omni-captions-skills && python -m pytest tests/test_services_download.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

Extract `_download_with_ytdlp` and helpers from `caption.py:210-404` into `services/download.py`. Key changes:
- Remove `self` references, make pure functions
- Return `ToolResult` instead of raising exceptions
- Move `DownloadResult`, `VIDEO_QUALITY_FORMATS`, `VIDEO_PLATFORM_PATTERNS`, `YOUTUBE_VIDEO_ID_RE`, `is_youtube_video_id`, `resolve_video_input`, `_detect_audio_language`, `_is_video_platform_url` from `caption.py`

```python
# src/omnicaptions/services/download.py
"""Download service - media download via yt-dlp."""

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional

from .base import ErrorCode, ToolResult

logger = logging.getLogger("omnicaptions.services.download")

# --- Constants moved from caption.py ---

VIDEO_QUALITY_FORMATS = {
    "best": "bestvideo+bestaudio/best",
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
    "audio": "bestaudio/best",
}

VIDEO_PLATFORM_PATTERNS = [
    r"youtube\.com/watch",
    r"youtube\.com/shorts",
    r"youtu\.be/",
    r"bilibili\.com/video",
    r"vimeo\.com/",
    r"twitter\.com/.*/status",
    r"x\.com/.*/status",
]

YOUTUBE_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


class DownloadResult(NamedTuple):
    """Result from yt-dlp download."""
    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    caption_path: Optional[Path] = None
    video_id: str = ""
    title: str = ""
    width: Optional[int] = None
    height: Optional[int] = None


def is_youtube_video_id(value: str) -> bool:
    return bool(YOUTUBE_VIDEO_ID_RE.match(value))


def resolve_video_input(url_or_id: str) -> str:
    """Resolve bare YouTube video ID to full URL. Returns as-is if not an ID."""
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
    return full_url


def is_video_platform_url(url: str) -> bool:
    for pattern in VIDEO_PLATFORM_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def _detect_audio_language(info: dict) -> Optional[str]:
    """Detect original audio language from video info."""
    for fmt in info.get("formats", []):
        if fmt.get("acodec") and fmt.get("acodec") != "none":
            lang = fmt.get("language")
            if lang:
                return lang
    for fmt in info.get("requested_formats", []):
        lang = fmt.get("language")
        if lang:
            return lang
    return None


def _download_with_ytdlp(
    url: str,
    output_dir: Path,
    quality: str = "audio",
    verbose: bool = False,
) -> DownloadResult:
    """Core download logic extracted from GeminiCaption._download_with_ytdlp.

    This is the internal function. Use download_media() for the public API.
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    format_str = VIDEO_QUALITY_FORMATS.get(quality, VIDEO_QUALITY_FORMATS["audio"])
    is_video = quality != "audio"

    ydl_opts = {
        "format": format_str,
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": not verbose,
        "no_warnings": not verbose,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt/srt/best",
        "ignoreerrors": "only_download",
    }

    if is_video:
        ydl_opts["merge_output_format"] = "mp4"

    # First pass: extract info
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info.get("id", "video")
        title = info.get("title", "")
        video_width = info.get("width")
        video_height = info.get("height")

        original_lang = (
            info.get("language")
            or info.get("audio_language")
            or _detect_audio_language(info)
            or "en"
        )

        subtitles = info.get("subtitles", {})
        auto_subs = info.get("automatic_captions", {})

        preferred_langs = [original_lang]
        for lang_code in list(subtitles.keys()) + list(auto_subs.keys()):
            if lang_code.startswith(original_lang) and lang_code not in preferred_langs:
                preferred_langs.append(lang_code)
        for fallback in ["en", "zh", "ja", "ko"]:
            if fallback not in preferred_langs:
                preferred_langs.append(fallback)

        sub_lang = None
        for lang in preferred_langs:
            if lang in subtitles:
                sub_lang = lang
                if lang.startswith(original_lang):
                    break
        if not sub_lang or not sub_lang.startswith(original_lang):
            for lang in preferred_langs:
                if lang in auto_subs and lang.startswith(original_lang):
                    sub_lang = lang
                    break
        if not sub_lang:
            for lang in preferred_langs:
                if lang in subtitles:
                    sub_lang = lang
                    break
                if lang in auto_subs:
                    sub_lang = lang
                    break

        if sub_lang:
            ydl_opts["subtitleslangs"] = [sub_lang]
        else:
            ydl_opts["writesubtitles"] = False
            ydl_opts["writeautomaticsub"] = False

    # Second pass: download
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find downloaded files
    audio_path = None
    video_path = None
    caption_path = None

    audio_exts = (".mp3", ".m4a", ".opus", ".ogg", ".wav")
    video_exts = (".mp4", ".mkv", ".avi", ".mov", ".flv")
    caption_exts = (".vtt", ".srt", ".ass", ".srv3")

    for f in output_dir.iterdir():
        if f.name.startswith(video_id):
            if f.suffix in video_exts:
                video_path = f
            elif f.suffix in audio_exts:
                audio_path = f
            elif f.suffix in caption_exts:
                caption_path = f

    if is_video and not video_path:
        raise RuntimeError(f"Failed to download video for {url}")
    if not is_video and not audio_path:
        raise RuntimeError(f"Failed to download audio for {url}")

    # Save metadata
    meta_path = output_dir / f"{video_id}.meta.json"
    meta_data = {
        "video_id": video_id,
        "video_url": url,
        "title": title,
        "width": video_width,
        "height": video_height,
    }
    meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2))

    return DownloadResult(
        audio_path=audio_path,
        video_path=video_path,
        caption_path=caption_path,
        video_id=video_id,
        title=title,
        width=video_width,
        height=video_height,
    )


def download_media(
    url: str,
    output_dir: Path | None = None,
    quality: str = "audio",
    verbose: bool = False,
) -> ToolResult:
    """Download media from URL. Returns ToolResult.

    Args:
        url: Video URL or YouTube video ID
        output_dir: Output directory (default: cwd)
        quality: "audio", "best", "1080p", "720p", "480p", "360p"
        verbose: Enable verbose logging
    """
    if not url or not url.strip():
        return ToolResult(
            status="error",
            message="URL is required",
            error_code=ErrorCode.INVALID_URL,
        )

    if output_dir is None:
        output_dir = Path.cwd()

    try:
        resolved_url = resolve_video_input(url)
    except ValueError as e:
        return ToolResult(
            status="error",
            message=str(e),
            error_code=ErrorCode.INVALID_URL,
        )

    try:
        result = _download_with_ytdlp(resolved_url, output_dir, quality, verbose)
        return ToolResult(
            status="success",
            message=f"Downloaded {result.title or result.video_id}",
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
    except Exception as e:
        return ToolResult(
            status="error",
            message=str(e),
            error_code=ErrorCode.DOWNLOAD_FAILED,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd omni-captions-skills && python -m pytest tests/test_services_download.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/omnicaptions/services/download.py tests/test_services_download.py
git commit -m "feat: extract download service from GeminiCaption"
```

---

### Task 3: Extract transcribe service

**Files:**
- Create: `src/omnicaptions/services/transcribe.py`
- Test: `tests/test_services_transcribe.py`
- Reference: `src/omnicaptions/caption.py:440-536` (transcribe methods)

**Step 1: Write the failing test**

```python
# tests/test_services_transcribe.py
"""Tests for transcribe service."""

from unittest.mock import MagicMock, patch

from omnicaptions.services.transcribe import transcribe_media
from omnicaptions.services.base import ErrorCode


def test_transcribe_missing_api_key():
    with patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value=None):
        result = transcribe_media("video.mp4")
        assert result.status == "error"
        assert result.error_code == ErrorCode.MISSING_API_KEY


def test_transcribe_file_not_found():
    with patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key"):
        result = transcribe_media("/nonexistent/file.mp4")
        assert result.status == "error"
        assert result.error_code == ErrorCode.INVALID_INPUT


def test_transcribe_returns_tool_result_keys():
    """Verify transcribe_media ToolResult has expected data keys on success."""
    with patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key"), \
         patch("omnicaptions.services.transcribe._run_transcription", return_value="transcript text"):
        result = transcribe_media("https://youtube.com/watch?v=abc123")
        assert result.status == "success"
        assert "text" in result.data
        assert "output_path" in result.data or result.data.get("text") == "transcript text"
```

**Step 2: Run test to verify it fails**

Run: `cd omni-captions-skills && python -m pytest tests/test_services_transcribe.py -v`
Expected: FAIL

**Step 3: Write implementation**

Extract from `caption.py:440-536`. Key changes:
- `_get_transcription_prompt()` becomes module-level `get_transcription_prompt(language, custom_prompt)`
- `transcribe_async` becomes `_run_transcription(url_or_path, api_key, model, language, use_thinking)`
- Public API: `transcribe_media(input, output, api_key, model, language, ...) -> ToolResult`

```python
# src/omnicaptions/services/transcribe.py
"""Transcribe service - media transcription via Gemini API."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from .base import ErrorCode, ToolResult

logger = logging.getLogger("omnicaptions.services.transcribe")

DEFAULT_MODEL = "gemini-3-flash-preview"


def get_gemini_api_key() -> Optional[str]:
    from omnicaptions.config import get_gemini_api_key as _get_key
    return _get_key()


def get_transcription_prompt(language: Optional[str] = None, custom_prompt: Optional[str] = None) -> str:
    if custom_prompt:
        return custom_prompt
    prompt_path = Path(__file__).parent.parent / "prompts" / "transcription_dotey.md"
    if prompt_path.exists():
        base_prompt = prompt_path.read_text(encoding="utf-8").strip()
    else:
        base_prompt = "You are an expert transcript specialist.\nTranscribe the audio/video content verbatim with timestamps in [HH:MM:SS] format."
    if language:
        base_prompt += f"\n\n* Use {language} language for transcription."
    return base_prompt


async def _transcribe_async(
    url_or_path: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    use_thinking: bool = True,
    custom_prompt: Optional[str] = None,
    verbose: bool = False,
) -> str:
    from google import genai
    from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

    client = genai.Client(api_key=api_key)
    prompt = get_transcription_prompt(language, custom_prompt)

    config_kwargs = {
        "system_instruction": prompt,
        "response_modalities": ["TEXT"],
    }
    if use_thinking:
        config_kwargs["thinking_config"] = ThinkingConfig(include_thoughts=False, thinking_budget=-1)
    gen_config = GenerateContentConfig(**config_kwargs)

    if url_or_path.startswith(("http://", "https://")):
        contents = Part.from_uri(file_uri=url_or_path, mime_type="video/*")
    else:
        path = Path(url_or_path)
        # Handle non-ASCII filenames
        try:
            str(path).encode("ascii")
            upload_path = path
            temp_link = None
        except UnicodeEncodeError:
            temp_link = Path(tempfile.gettempdir()) / f"upload_{path.suffix}"
            temp_link.unlink(missing_ok=True)
            temp_link.symlink_to(path.resolve())
            upload_path = temp_link

        try:
            uploaded = client.files.upload(file=str(upload_path))
            contents = Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
        finally:
            if temp_link:
                temp_link.unlink(missing_ok=True)

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model, contents=contents, config=gen_config,
        ),
    )

    if not response.text:
        raise RuntimeError("Empty response from Gemini API")
    return response.text.strip()


def _run_transcription(
    url_or_path: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    **kwargs,
) -> str:
    key = api_key or get_gemini_api_key()
    if not key:
        raise ValueError("Gemini API key required")
    return asyncio.run(_transcribe_async(url_or_path, key, model, **kwargs))


def transcribe_media(
    input_path: str,
    output: str | Path | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    use_thinking: bool = True,
    custom_prompt: str | None = None,
    verbose: bool = False,
) -> ToolResult:
    """Transcribe audio/video. Returns ToolResult.

    Args:
        input_path: URL or local file path
        output: Output file path (auto-generated if None)
        api_key: Gemini API key (reads from config if None)
        model: Gemini model name
        language: Force language
        use_thinking: Enable thinking mode
        custom_prompt: Custom system prompt
        verbose: Enable verbose logging
    """
    key = api_key or get_gemini_api_key()
    if not key:
        return ToolResult(
            status="error",
            message="Gemini API key not found. Get one from https://aistudio.google.com/apikey",
            error_code=ErrorCode.MISSING_API_KEY,
        )

    # Validate local file
    if not input_path.startswith(("http://", "https://")) and not Path(input_path).exists():
        return ToolResult(
            status="error",
            message=f"File not found: {input_path}",
            error_code=ErrorCode.INVALID_INPUT,
        )

    try:
        text = _run_transcription(
            input_path, key, model,
            language=language, use_thinking=use_thinking,
            custom_prompt=custom_prompt, verbose=verbose,
        )

        # Save output
        output_path = None
        if output:
            output_path = Path(output)
        else:
            from .download import is_youtube_video_id
            import re
            if is_youtube_video_id(input_path):
                stem = input_path
            elif input_path.startswith(("http://", "https://")):
                match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", input_path)
                stem = match.group(1) if match else "output"
            else:
                stem = Path(input_path).stem
            output_path = Path.cwd() / f"{stem}_GeminiUnd.md"

        output_path.write_text(text, encoding="utf-8")

        return ToolResult(
            status="success",
            message=f"Transcribed to {output_path}",
            data={
                "output_path": str(output_path),
                "text": text,
                "char_count": len(text),
            },
        )
    except Exception as e:
        return ToolResult(
            status="error",
            message=str(e),
            error_code=ErrorCode.TRANSCRIPTION_FAILED,
        )
```

**Step 4: Run tests**

Run: `cd omni-captions-skills && python -m pytest tests/test_services_transcribe.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/omnicaptions/services/transcribe.py tests/test_services_transcribe.py
git commit -m "feat: extract transcribe service from GeminiCaption"
```

---

### Task 4: Extract translate service

**Files:**
- Create: `src/omnicaptions/services/translate.py`
- Test: `tests/test_services_translate.py`
- Reference: `src/omnicaptions/caption.py:540-686` (translate methods)

**Step 1: Write the failing test**

```python
# tests/test_services_translate.py
"""Tests for translate service."""

from pathlib import Path
from unittest.mock import patch

from omnicaptions.services.translate import translate_caption
from omnicaptions.services.base import ErrorCode


def test_translate_missing_api_key():
    with patch("omnicaptions.services.translate.get_gemini_api_key", return_value=None):
        result = translate_caption("input.srt", target_lang="zh")
        assert result.status == "error"
        assert result.error_code == ErrorCode.MISSING_API_KEY


def test_translate_file_not_found():
    with patch("omnicaptions.services.translate.get_gemini_api_key", return_value="fake-key"):
        result = translate_caption("/nonexistent/file.srt", target_lang="zh")
        assert result.status == "error"
        assert result.error_code == ErrorCode.INVALID_INPUT


def test_translate_returns_tool_result(tmp_path):
    """Verify translate_caption returns correct ToolResult on success."""
    srt = tmp_path / "test.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")

    with patch("omnicaptions.services.translate.get_gemini_api_key", return_value="fake-key"), \
         patch("omnicaptions.services.translate._translate_batch_async") as mock_batch:
        mock_batch.return_value = ["你好"]
        result = translate_caption(str(srt), target_lang="zh", output=str(tmp_path / "out.srt"))
        assert result.status == "success"
        assert "output_path" in result.data
```

**Step 2: Run test → FAIL**

**Step 3: Write implementation**

Extract `_build_translate_prompt`, `_translate_batch`, `translate_async` from `caption.py:540-686`.

Key structure:
```python
# src/omnicaptions/services/translate.py
LANGUAGE_NAMES = {...}  # moved from caption.py

def _build_translate_prompt(...) -> str: ...
async def _translate_batch_async(...) -> list: ...
async def _translate_async(...) -> Path: ...
def translate_caption(input_file, target_lang, output, ...) -> ToolResult: ...
```

**Step 4: Run tests → PASS**

**Step 5: Commit**

```bash
git add src/omnicaptions/services/translate.py tests/test_services_translate.py
git commit -m "feat: extract translate service from GeminiCaption"
```

---

### Task 5: Extract convert service

**Files:**
- Create: `src/omnicaptions/services/convert.py`
- Test: `tests/test_services_convert.py`
- Reference: `src/omnicaptions/cli.py:30-318,512-686` (resolution, ASS styles, cmd_convert)

**Step 1: Write the failing test**

```python
# tests/test_services_convert.py
"""Tests for convert service."""

from pathlib import Path

from omnicaptions.services.convert import convert_caption
from omnicaptions.services.base import ErrorCode

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_VTT_FILE = TEST_DATA_DIR / "SA1.vtt"


def test_convert_file_not_found():
    result = convert_caption("/nonexistent/file.srt", output="/tmp/out.vtt")
    assert result.status == "error"
    assert result.error_code == ErrorCode.INVALID_INPUT


def test_convert_vtt_to_srt(tmp_path):
    output = tmp_path / "out.srt"
    result = convert_caption(str(TEST_VTT_FILE), output=str(output))
    assert result.status == "success"
    assert output.exists()
    assert "She had your dark suit" in output.read_text()


def test_convert_returns_tool_result(tmp_path):
    output = tmp_path / "out.srt"
    result = convert_caption(str(TEST_VTT_FILE), output=str(output))
    assert result.data["output_path"] == str(output)
    assert result.data["output_format"] == "srt"
```

**Step 2: Run test → FAIL**

**Step 3: Write implementation**

Move from `cli.py` into `services/convert.py`:
- `RESOLUTION_PRESETS`, `get_video_resolution`, `calculate_fontsize`, `parse_resolution`, `find_video_metadata`
- `ASS_STYLE_PRESETS`, `hex_to_ass_color`, `apply_bilingual_colors`, `build_ass_metadata`, `read_ass_styles`, `color_to_ass`
- Core conversion logic from `cmd_convert`

Public API:
```python
def convert_caption(
    input_file: str,
    output: str | None = None,
    input_format: str | None = None,
    output_format: str | None = None,
    style: str | None = None,
    line1_color: str | None = None,
    line2_color: str | None = None,
    karaoke: str | None = None,
    fontsize: int | None = None,
    resolution: str | None = None,
    video_file: str | None = None,
    verbose: bool = False,
) -> ToolResult:
```

**Step 4: Run tests → PASS**

**Step 5: Commit**

```bash
git add src/omnicaptions/services/convert.py tests/test_services_convert.py
git commit -m "feat: extract convert service from cli.py"
```

---

### Task 6: Extract align service

**Files:**
- Create: `src/omnicaptions/services/align.py`
- Test: `tests/test_services_align.py`
- Reference: `src/omnicaptions/cli.py:734-806` (cmd_laicut_align)

**Step 1: Write the failing test**

```python
# tests/test_services_align.py
"""Tests for align service."""

from omnicaptions.services.align import forced_align
from omnicaptions.services.base import ErrorCode
from unittest.mock import patch


def test_align_missing_api_key():
    with patch("omnicaptions.services.align.get_lattifai_api_key", return_value=None):
        result = forced_align("audio.mp4", "caption.srt")
        assert result.status == "error"
        assert result.error_code == ErrorCode.MISSING_API_KEY


def test_align_audio_not_found():
    with patch("omnicaptions.services.align.get_lattifai_api_key", return_value="fake-key"):
        result = forced_align("/nonexistent/audio.mp4", "/nonexistent/caption.srt")
        assert result.status == "error"
        assert result.error_code == ErrorCode.INVALID_INPUT
```

**Step 2: Run test → FAIL**

**Step 3: Write implementation**

```python
# src/omnicaptions/services/align.py
def forced_align(
    audio: str,
    caption: str,
    output: str | None = None,
    output_format: str | None = None,
    api_key: str | None = None,
    split_sentence: bool = False,
    verbose: bool = False,
) -> ToolResult:
```

**Step 4: Run tests → PASS**

**Step 5: Commit**

```bash
git add src/omnicaptions/services/align.py tests/test_services_align.py
git commit -m "feat: extract align service from cli.py"
```

---

### Task 7: Add `--json` output to CLI

**Files:**
- Modify: `src/omnicaptions/cli.py`
- Test: `tests/test_cli_json.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_json.py
"""Tests for CLI --json output."""

import json
import subprocess
import sys
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_VTT_FILE = TEST_DATA_DIR / "SA1.vtt"


def test_convert_json_output(tmp_path):
    output = tmp_path / "out.srt"
    result = subprocess.run(
        [sys.executable, "-m", "omnicaptions", "convert", str(TEST_VTT_FILE), "-o", str(output), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["status"] == "success"
    assert "output_path" in data["data"]
    assert data["error_code"] is None


def test_convert_json_error(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "omnicaptions", "convert", "/nonexistent.srt", "-o", str(tmp_path / "out.srt"), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["status"] == "error"
    assert data["error_code"] is not None
```

**Step 2: Run test → FAIL**

**Step 3: Refactor cli.py**

Key changes:
1. Add `--json` flag to all subparsers
2. Each `cmd_*` function calls its service and either prints human-readable or JSON
3. Standardize exit codes: 0=success, 1=error, 2=arg error

```python
# Pattern for each command:
def cmd_convert(args):
    from omnicaptions.services.convert import convert_caption
    result = convert_caption(
        input_file=args.input,
        output=args.output,
        # ... map args to service params
    )
    _output_result(result, getattr(args, "json", False))

def _output_result(result: ToolResult, as_json: bool):
    if as_json:
        import json
        print(json.dumps(result.to_json(), ensure_ascii=False))
    else:
        if result.status == "success":
            print(result.message)
        else:
            print(f"Error: {result.message}", file=sys.stderr)
    if result.status == "error":
        sys.exit(1)
```

Add `--json` to every subparser:
```python
for p in [p_transcribe, p_download, p_convert, p_translate, p_laicut]:
    p.add_argument("--json", action="store_true", help="Output JSON (for agent consumption)")
```

**Step 4: Run all tests**

Run: `cd omni-captions-skills && python -m pytest -v`
Expected: All existing tests PASS + new JSON tests PASS

**Step 5: Commit**

```bash
git add src/omnicaptions/cli.py tests/test_cli_json.py
git commit -m "feat: refactor CLI to use services layer, add --json output"
```

---

### Task 8: Add `doctor` command

**Files:**
- Create: `src/omnicaptions/services/doctor.py`
- Modify: `src/omnicaptions/cli.py` (add doctor subparser)
- Test: `tests/test_services_doctor.py`

**Step 1: Write the failing test**

```python
# tests/test_services_doctor.py
"""Tests for doctor (preflight check) service."""

from unittest.mock import patch
from omnicaptions.services.doctor import run_doctor


def test_doctor_returns_tool_result():
    result = run_doctor()
    assert result.status in ("success", "error")
    assert "checks" in result.data


def test_doctor_checks_structure():
    result = run_doctor()
    checks = result.data["checks"]
    assert any(c["name"] == "ffmpeg" for c in checks)
    assert any(c["name"] == "yt-dlp" for c in checks)
    assert any(c["name"] == "gemini_api_key" for c in checks)
    assert any(c["name"] == "lattifai_api_key" for c in checks)


def test_doctor_missing_key():
    with patch("omnicaptions.services.doctor.get_gemini_api_key", return_value=None), \
         patch("omnicaptions.services.doctor.get_lattifai_api_key", return_value=None):
        result = run_doctor()
        checks = {c["name"]: c for c in result.data["checks"]}
        assert checks["gemini_api_key"]["status"] == "missing"
        assert "aistudio.google.com" in checks["gemini_api_key"]["help"]
```

**Step 2: Run test → FAIL**

**Step 3: Write implementation**

```python
# src/omnicaptions/services/doctor.py
"""Doctor service - preflight environment check."""

import shutil
from .base import ToolResult
from omnicaptions.config import get_gemini_api_key, get_lattifai_api_key


def run_doctor(verbose: bool = False) -> ToolResult:
    checks = []
    all_ok = True

    # ffmpeg
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    checks.append({
        "name": "ffmpeg",
        "status": "ok" if ffmpeg_ok else "missing",
        "help": "brew install ffmpeg" if not ffmpeg_ok else None,
    })
    if not ffmpeg_ok:
        all_ok = False

    # yt-dlp
    ytdlp_ok = shutil.which("yt-dlp") is not None
    checks.append({
        "name": "yt-dlp",
        "status": "ok" if ytdlp_ok else "missing",
        "help": "pip install yt-dlp" if not ytdlp_ok else None,
    })
    if not ytdlp_ok:
        all_ok = False

    # Gemini API key
    gemini_key = get_gemini_api_key()
    checks.append({
        "name": "gemini_api_key",
        "status": "ok" if gemini_key else "missing",
        "help": "Get key from https://aistudio.google.com/apikey then: export GEMINI_API_KEY=<key>" if not gemini_key else None,
    })

    # LattifAI API key
    lattifai_key = get_lattifai_api_key()
    checks.append({
        "name": "lattifai_api_key",
        "status": "ok" if lattifai_key else "missing",
        "help": "Get key from https://lattifai.com/dashboard then: export LATTIFAI_API_KEY=<key>" if not lattifai_key else None,
    })

    return ToolResult(
        status="success" if all_ok else "error",
        message="All checks passed" if all_ok else "Some checks failed",
        data={"checks": checks},
    )
```

Add to `cli.py`:
```python
p_doctor = subparsers.add_parser("doctor", help="Check environment dependencies")
p_doctor.add_argument("--json", action="store_true")
p_doctor.add_argument("-v", "--verbose", action="store_true")
p_doctor.set_defaults(func=cmd_doctor)
```

**Step 4: Run tests → PASS**

**Step 5: Commit**

```bash
git add src/omnicaptions/services/doctor.py tests/test_services_doctor.py src/omnicaptions/cli.py
git commit -m "feat: add doctor command for preflight environment checks"
```

---

### Task 9: Backward compatibility - update caption.py and __init__.py

**Files:**
- Modify: `src/omnicaptions/caption.py` (delegate to services)
- Modify: `src/omnicaptions/__init__.py` (add re-exports)
- Test: `tests/test_import.py` (must still pass)

**Step 1: Verify existing import tests still pass**

Run: `cd omni-captions-skills && python -m pytest tests/test_import.py -v`
Expected: PASS (existing API unchanged)

**Step 2: Update caption.py**

Keep `GeminiCaption` class but delegate to services internally:

```python
# caption.py - thin wrapper for backward compat
from .services.download import (
    DownloadResult, is_youtube_video_id, resolve_video_input,
    VIDEO_QUALITY_FORMATS, VIDEO_PLATFORM_PATTERNS, YOUTUBE_VIDEO_ID_RE,
    is_video_platform_url as _is_video_platform_url,
    _detect_audio_language,
)
from .services.translate import LANGUAGE_NAMES

class GeminiCaption:
    """Backward-compatible wrapper. Delegates to services layer."""
    # ... keep existing __init__ and method signatures
    # ... internally call services functions
```

**Step 3: Run ALL tests**

Run: `cd omni-captions-skills && python -m pytest -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/omnicaptions/caption.py src/omnicaptions/__init__.py
git commit -m "refactor: delegate GeminiCaption methods to services layer"
```

---

### Task 10: Redesign Skills - Pipeline Skill

**Files:**
- Create: `skills/omnicaptions-pipeline/SKILL.md`
- Delete: `skills/omnicaptions-download/SKILL.md`
- Delete: `skills/omnicaptions-translate/SKILL.md`
- Keep: `skills/omnicaptions-transcribe/SKILL.md` (update)
- Keep: `skills/omnicaptions-LaiCut/SKILL.md` (update)
- Keep: `skills/omnicaptions-convert/SKILL.md` (update)

**Step 1: Create pipeline skill**

```markdown
# skills/omnicaptions-pipeline/SKILL.md
---
name: omnicaptions-pipeline
description: End-to-end caption workflow. Use when user wants captions for a video, asks to "subtitle this", "transcribe and translate", or any multi-step caption task.
allowed-tools: Bash(omnicaptions:*)
---

# OmniCaptions Pipeline

One-shot caption workflow. Choose a template, run CLI commands with `--json`.

## Step 0: Preflight

```bash
omnicaptions doctor --json
```

Check output. If any dependency is missing, show the `help` field and stop.

## Templates

### Template A: Full Pipeline (URL → translated captions)
Use when: User provides a URL and wants captions in another language.

```bash
# 1. Download
omnicaptions download "URL" -q audio --json

# 2. Check if captions were downloaded
#    If caption_path exists → skip transcribe, use downloaded caption
#    If no caption → transcribe:
omnicaptions transcribe "URL" --json

# 3. Align (optional, if user wants precise timing)
omnicaptions LaiCut <audio> <caption> --json

# 4. Translate (use Claude native for zero-key experience, or Gemini CLI)
#    Claude native: read the caption file, translate in conversation
#    Gemini CLI: omnicaptions translate <caption> -l <lang> --json

# 5. Convert to final format
omnicaptions convert <caption> -o <output> --json
```

### Template B: Transcribe + Translate
Use when: User has a local file, no existing captions.

```bash
omnicaptions transcribe <file> --json
# Then Claude native translate or:
omnicaptions translate <output> -l <lang> --json
```

### Template C: Align + Translate
Use when: User already has captions but wants better timing + translation.

```bash
omnicaptions LaiCut <audio> <caption> --json
omnicaptions translate <aligned_caption> -l <lang> --json
omnicaptions convert <translated> -o <output.ass> --style bilingual --json
```

## Translation Decision

- **No GEMINI_API_KEY** → use Claude native (read file, translate in conversation, write output)
- **Has GEMINI_API_KEY** → use `omnicaptions translate` CLI (batch + context-aware)
- **User explicitly asks for Claude/Gemini** → follow their preference

## Output Naming Convention

Follow the chain: `base.lang_op1_op2.ext`
- `video.en.vtt` → downloaded
- `video.en_LaiCut.json` → aligned
- `video.en_LaiCut_Claude_zh.srt` → translated
- `video_GeminiUnd.md` → transcribed
```

**Step 2: Update remaining skills**

Update `omnicaptions-transcribe/SKILL.md`, `omnicaptions-LaiCut/SKILL.md`, `omnicaptions-convert/SKILL.md`:
- Add note: "For multi-step workflows, use /omnicaptions:pipeline instead"
- Add `--json` flag in all command examples
- Remove redundant decision logic (pipeline handles it)

**Step 3: Remove download and translate skills**

Delete `skills/omnicaptions-download/SKILL.md` and `skills/omnicaptions-translate/SKILL.md` (pipeline covers these).

**Step 4: Commit**

```bash
git add skills/
git rm skills/omnicaptions-download/SKILL.md skills/omnicaptions-translate/SKILL.md
git commit -m "feat: add pipeline skill, remove download/translate skills"
```

---

### Task 11: Update CLAUDE.md and documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

Reflect new architecture:
- Services layer documentation
- Updated CLI with `--json` and `doctor`
- New skill structure (pipeline + transcribe + LaiCut + convert)

**Step 2: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README for new architecture"
```

---

### Task 12: Final integration test

**Step 1: Run full test suite**

```bash
cd omni-captions-skills && python -m pytest -v --tb=short
```

Expected: ALL PASS

**Step 2: Manual smoke test**

```bash
# Doctor check
omnicaptions doctor --json

# Convert with JSON output
omnicaptions convert tests/data/SA1.vtt -o /tmp/test.srt --json
```

**Step 3: Verify backward compat**

```python
python -c "from omnicaptions import GeminiCaption; gc = GeminiCaption(); print('OK')"
```

**Step 4: Final commit tag**

```bash
git tag -a v0.2.0 -m "Agent-first redesign: services layer, --json output, pipeline skill"
```
