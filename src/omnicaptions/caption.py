"""Unified Gemini Caption class for transcription and translation."""

import asyncio
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional, Union

from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig
from lattifai.caption import Caption, GeminiReader, GeminiWriter

from .config import get_gemini_api_key, get_setup_instructions


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

LANGUAGE_NAMES = {
    "zh": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
}


@dataclass
class GeminiCaptionConfig:
    """Configuration for GeminiCaption."""

    model_name: str = "gemini-3-flash-preview"
    gemini_api_key: Optional[str] = field(default_factory=get_gemini_api_key)
    verbose: bool = False
    # Transcription
    language: Optional[str] = None
    use_thinking: bool = True
    # Translation
    batch_size: int = 30
    context_lines: int = 5  # Context lines before/after batch for coherent translation
    # yt-dlp
    use_ytdlp: bool = True  # Auto-download videos from platforms
    keep_downloaded: bool = True  # Keep downloaded files after transcription


class GeminiCaption:
    """
    Unified class for media transcription and caption translation.

    Usage:
        gc = GeminiCaption(verbose=True)

        # Transcribe
        result = gc.transcribe("video.mp4")
        gc.write(result, "transcript.md")

        # Translate
        gc.translate("input.srt", "output.zh.srt", "zh")

        # TODO: Transcribe + Translate in one step
        # result = gc.transcribe("video.mp4", translate="zh")
    """

    def __init__(
        self,
        config: Optional[GeminiCaptionConfig] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config = config or GeminiCaptionConfig()

        if api_key:
            self.config.gemini_api_key = api_key
        if model_name:
            self.config.model_name = model_name
        if verbose:
            self.config.verbose = verbose

        self._client: Optional["genai.Client"] = None
        self._transcription_prompt: Optional[str] = None
        self.logger = logging.getLogger("gemini_caption")

        if not self.config.gemini_api_key:
            self.logger.warning(f"Gemini API key not found.\n{get_setup_instructions()}")

    # ==================== Common ====================

    def _get_client(self) -> "genai.Client":
        if not self.config.gemini_api_key:
            raise ValueError("Gemini API key is required")
        if self._client is None:
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    @staticmethod
    def _is_video_platform_url(url: str) -> bool:
        """Check if URL is from a supported video platform."""
        for pattern in VIDEO_PLATFORM_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False

    @staticmethod
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
        self, url: str, output_dir: Optional[Path] = None, quality: str = "audio"
    ) -> DownloadResult:
        """Download audio/video and captions using yt-dlp.

        Downloads:
        - Audio/Video: Based on quality setting
        - Captions: Manual (user-uploaded) first, then auto-generated, original language

        Args:
            url: Video URL
            output_dir: Output directory (uses temp dir if None)
            quality: Quality preset - "audio", "best", "1080p", "720p", "480p", "360p"

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
            "quiet": not self.config.verbose,
            "no_warnings": not self.config.verbose,
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

        if self.config.verbose:
            self.logger.info(f"Downloading with yt-dlp ({quality}): {url}")

        # First pass: extract info to get available captions
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get("id", "video")
            title = info.get("title", "")

            # Extract video resolution
            video_width = info.get("width")
            video_height = info.get("height")
            if self.config.verbose and video_width and video_height:
                self.logger.info(f"Video resolution: {video_width}x{video_height}")

            # Detect original audio language from multiple sources
            original_lang = (
                info.get("language")  # Video metadata language
                or info.get("audio_language")  # Audio track language
                or self._detect_audio_language(info)  # From audio tracks
                or "en"  # Default fallback
            )

            if self.config.verbose:
                self.logger.info(f"Detected original language: {original_lang}")

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
                if self.config.verbose:
                    lang_match = (
                        "✓ matches audio"
                        if sub_lang.startswith(original_lang)
                        else "≠ different from audio"
                    )
                    self.logger.info(f"Selected {sub_source} caption: {sub_lang} ({lang_match})")
            else:
                # No captions available, disable caption download
                ydl_opts["writesubtitles"] = False
                ydl_opts["writeautomaticsub"] = False
                if self.config.verbose:
                    self.logger.info("No captions available")

        # Second pass: download with determined options
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find downloaded files
        audio_path = None
        video_path = None
        caption_path = None

        audio_exts = (".webm", ".m4a", ".mp3", ".opus", ".ogg", ".wav")
        video_exts = (".mp4", ".mkv", ".avi", ".mov", ".flv")
        caption_exts = (".vtt", ".srt", ".ass")

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

        if self.config.verbose:
            if video_path:
                self.logger.info(f"Downloaded video: {video_path}")
            if audio_path:
                self.logger.info(f"Downloaded audio: {audio_path}")
            if caption_path:
                self.logger.info(f"Downloaded caption: {caption_path}")

        # Save metadata to .meta.json for later use (e.g., ASS font scaling)
        meta_path = output_dir / f"{video_id}.meta.json"
        meta_data = {
            "video_id": video_id,
            "title": title,
            "width": video_width,
            "height": video_height,
        }
        meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2))
        if self.config.verbose:
            self.logger.info(f"Saved metadata: {meta_path}")

        return DownloadResult(
            audio_path=audio_path,
            video_path=video_path,
            caption_path=caption_path,
            video_id=video_id,
            title=title,
            width=video_width,
            height=video_height,
        )

    # ==================== Download ====================

    def download(
        self,
        url: str,
        output_dir: Optional[Union[str, Path]] = None,
        quality: str = "audio",
    ) -> DownloadResult:
        """Download audio/video and captions from a video URL.

        Args:
            url: Video URL (YouTube, Bilibili, etc.)
            output_dir: Output directory (uses current dir if None)
            quality: Quality preset - "audio", "best", "1080p", "720p", "480p", "360p"

        Returns:
            DownloadResult with audio_path/video_path, caption_path, video_id, title
        """
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path.cwd()

        # Override keep_downloaded since user explicitly wants to download
        original_keep = self.config.keep_downloaded
        self.config.keep_downloaded = True

        try:
            return self._download_with_ytdlp(url, output_path, quality)
        finally:
            self.config.keep_downloaded = original_keep

    # ==================== Transcription ====================

    def _get_transcription_prompt(self) -> str:
        if self._transcription_prompt is not None:
            return self._transcription_prompt

        prompt_path = Path(__file__).parent / "prompts" / "transcription_dotey.md"

        if prompt_path.exists():
            base_prompt = prompt_path.read_text(encoding="utf-8").strip()
        else:
            self.logger.warning(f"Prompt file not found: {prompt_path}, using default prompt")
            base_prompt = """You are an expert transcript specialist.
Transcribe the audio/video content verbatim with timestamps in [HH:MM:SS] format.
Include speaker labels when multiple speakers are present."""

        if self.config.language:
            base_prompt += f"\n\n* Use {self.config.language} language for transcription."

        self._transcription_prompt = base_prompt
        return self._transcription_prompt

    async def _transcribe_media(self, contents: "Part", source: str) -> str:
        """Run transcription generation."""
        client = self._get_client()

        config_kwargs = {
            "system_instruction": self._get_transcription_prompt(),
            "response_modalities": ["TEXT"],
        }
        if self.config.use_thinking:
            config_kwargs["thinking_config"] = ThinkingConfig(
                include_thoughts=False, thinking_budget=-1
            )

        gen_config = GenerateContentConfig(**config_kwargs)

        if self.config.verbose:
            self.logger.info(f"Transcribing with {self.config.model_name}...")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=gen_config,
            ),
        )

        if not response.text:
            raise RuntimeError("Empty response from Gemini API")

        if self.config.verbose:
            self.logger.info(f"Transcription completed: {len(response.text)} chars")

        return response.text.strip()

    async def transcribe_async(self, url_or_path: Union[str, Path]) -> str:
        """Transcribe audio/video from URL or local file."""
        url_or_path_str = str(url_or_path)

        if url_or_path_str.startswith(("http://", "https://")):
            # URL: Gemini can directly process YouTube and other video platform URLs
            if self.config.verbose:
                self.logger.info(f"Transcribing URL: {url_or_path_str}")
            contents = Part.from_uri(file_uri=url_or_path_str, mime_type="video/*")
            return await self._transcribe_media(contents, url_or_path_str)
        else:
            # Local file
            path = Path(url_or_path)
            if self.config.verbose:
                self.logger.info(f"Transcribing local file: {path}")
            client = self._get_client()

            # Handle non-ASCII filenames (Google GenAI SDK requires ASCII in headers)
            try:
                str(path).encode("ascii")
                upload_path = path
                temp_link = None
            except UnicodeEncodeError:
                # Create temp symlink with ASCII name
                temp_link = Path(tempfile.gettempdir()) / f"upload_{path.suffix}"
                temp_link.unlink(missing_ok=True)
                temp_link.symlink_to(path.resolve())
                upload_path = temp_link
                if self.config.verbose:
                    self.logger.info(f"Using temp link for non-ASCII filename: {temp_link}")

            try:
                uploaded = client.files.upload(file=str(upload_path))
                contents = Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
                return await self._transcribe_media(contents, str(path))
            finally:
                if temp_link:
                    temp_link.unlink(missing_ok=True)

    def transcribe(self, url_or_path: Union[str, Path]) -> str:
        """Transcribe audio/video (sync)."""
        return asyncio.run(self.transcribe_async(url_or_path))

    # ==================== Translation ====================

    def _build_translate_prompt(
        self,
        texts: list[str],
        target_lang: str,
        bilingual: bool,
        context_before: list[str] | None = None,
        context_after: list[str] | None = None,
    ) -> str:
        lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        if bilingual:
            instruction = f"""Translate captions to {lang_name}. Return JSON array with "original" and "translated" keys.

Rules:
1. ONLY translate the lines in "to_translate" - context is for reference only
2. Maintain speaker voice and tone consistency
3. Handle split sentences naturally (context helps understand full meaning)
4. Adapt idioms and cultural references appropriately
5. Keep the exact same order and count as input"""
        else:
            instruction = f"""Translate captions to {lang_name}. Return JSON array of translated strings.

Rules:
1. ONLY translate the lines in "to_translate" - context is for reference only
2. Maintain speaker voice and tone consistency
3. Handle split sentences naturally (context helps understand full meaning)
4. Adapt idioms and cultural references appropriately
5. Keep the exact same order and count as input"""

        # Build input with context
        input_data = {"to_translate": texts}
        if context_before:
            input_data["context_before"] = context_before
        if context_after:
            input_data["context_after"] = context_after

        return f"{instruction}\n\nInput:\n{json.dumps(input_data, ensure_ascii=False)}"

    async def _translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        bilingual: bool,
        context_before: list[str] | None = None,
        context_after: list[str] | None = None,
    ) -> list:
        """Translate a batch of texts with surrounding context."""
        client = self._get_client()
        prompt = self._build_translate_prompt(
            texts, target_lang, bilingual, context_before, context_after
        )

        config = GenerateContentConfig(response_mime_type="application/json")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=config,
            ),
        )

        if not response.text:
            raise RuntimeError("Empty response from Gemini API")

        return json.loads(response.text)

    async def translate_async(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        target_lang: str,
        bilingual: bool = False,
    ) -> Path:
        """Translate caption file."""
        input_path = Path(input_file)
        output_path = Path(output_file)

        if self.config.verbose:
            self.logger.info(f"Loading: {input_path}")

        # Handle Gemini markdown format
        if input_path.suffix.lower() == ".md":
            supervisions = GeminiReader.extract_for_alignment(str(input_path))
            cap = Caption.from_supervisions(supervisions)
        else:
            cap = Caption.read(str(input_path))

        texts = [sup.text for sup in cap.supervisions]

        if self.config.verbose:
            self.logger.info(f"Translating {len(texts)} segments to {target_lang}...")

        # Translate in batches with context
        results = []
        batch_size = self.config.batch_size
        context_lines = self.config.context_lines

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Extract context before and after current batch
            context_before = texts[max(0, i - context_lines) : i] if i > 0 else None
            batch_end = i + batch_size
            context_after = (
                texts[batch_end : batch_end + context_lines] if batch_end < len(texts) else None
            )

            if self.config.verbose:
                self.logger.info(
                    f"Batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}"
                )
            batch_result = await self._translate_batch(
                batch, target_lang, bilingual, context_before, context_after
            )
            results.extend(batch_result)

        # Update supervisions
        for idx, sup in enumerate(cap.supervisions):
            if bilingual:
                original = results[idx].get("original", sup.text)
                translated = results[idx].get("translated", "")
                sup.text = f"{original}\n{translated}"
            else:
                sup.text = results[idx] if isinstance(results[idx], str) else results[idx]

        # Write output - use GeminiWriter for .md files
        if output_path.suffix.lower() == ".md":
            GeminiWriter.write(cap.supervisions, str(output_path))
        else:
            cap.write(str(output_path))

        if self.config.verbose:
            self.logger.info(f"Saved: {output_path}")

        return output_path

    def translate(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        target_lang: str,
        bilingual: bool = False,
    ) -> Path:
        """Translate caption file (sync)."""
        return asyncio.run(self.translate_async(input_file, output_file, target_lang, bilingual))

    # ==================== Utilities ====================

    def write(self, text: str, output_file: Union[str, Path], encoding: str = "utf-8") -> Path:
        """Save text to file."""
        output_path = Path(output_file)
        output_path.write_text(text, encoding=encoding)
        return output_path
