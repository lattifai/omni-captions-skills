"""Unified Gemini Caption class for transcription and translation."""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig
from lattifai.caption import Caption, GeminiReader, GeminiWriter

from .config import get_gemini_api_key, get_setup_instructions

# Re-export from services for backward compatibility:
# ``from omnicaptions.caption import DownloadResult`` etc. still works.
from .services.download import (  # noqa: F401
    DownloadResult,
    VIDEO_PLATFORM_PATTERNS,
    VIDEO_QUALITY_FORMATS,
    YOUTUBE_VIDEO_ID_RE,
    _detect_audio_language,
    _download_with_ytdlp,
    is_video_platform_url,
    is_youtube_video_id,
    resolve_video_input,
)
from .services.translate import LANGUAGE_NAMES  # noqa: F401


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
        return is_video_platform_url(url)

    @staticmethod
    def _detect_audio_language(info: dict) -> Optional[str]:
        """Detect original audio language from video info."""
        return _detect_audio_language(info)

    def _download_with_ytdlp(
        self, url: str, output_dir: Optional[Path] = None, quality: str = "audio"
    ) -> DownloadResult:
        """Download audio/video and captions using yt-dlp.

        Delegates to ``services.download._download_with_ytdlp``.
        """
        return _download_with_ytdlp(
            url,
            output_dir=output_dir,
            quality=quality,
            verbose=self.config.verbose,
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
