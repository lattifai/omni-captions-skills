"""Pipeline service -- orchestrates download -> transcribe -> align -> translate -> convert.

Chains existing services into a single operation, handling conditional
step skipping and error propagation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from omnicaptions.config import get_gemini_api_key, get_lattifai_api_key
from omnicaptions.services.align import forced_align
from omnicaptions.services.base import ToolResult
from omnicaptions.services.convert import convert_caption
from omnicaptions.services.download import download_media
from omnicaptions.services.transcribe import transcribe_media
from omnicaptions.services.translate import translate_caption

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in the pipeline."""

    name: str
    status: str = "pending"  # pending, running, success, skipped, error
    message: str = ""
    output_path: str | None = None


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    url: str
    target_lang: str = "zh"
    output_format: str = "srt"
    output_dir: str | None = None
    bilingual: bool = True
    style: str | None = None
    model: str = "gemini-3-flash-preview"
    quality: str = "audio"
    skip_align: bool = False
    skip_translate: bool = False
    gemini_api_key: str | None = None
    lattifai_api_key: str | None = None
    verbose: bool = False
    on_step: Callable[[str, str], None] | None = None


def _notify(config: PipelineConfig, name: str, status: str):
    """Fire the on_step callback if present."""
    if config.on_step:
        config.on_step(name, status)


def run_pipeline(config: PipelineConfig) -> ToolResult:
    """Execute the full caption pipeline.

    Flow:
    1. download(url) -> audio_path, caption_path?
    2. IF no caption_path: transcribe(audio_path) -> caption_path
    3. IF not skip_align AND lattifai_key: align(audio, caption) -> caption_path
    4. IF not skip_translate: translate(caption, lang) -> caption_path
    5. convert(caption, format) -> final output

    Returns:
        ToolResult with data containing output_path and steps list.
    """
    steps: list[dict] = []
    output_dir = config.output_dir or str(Path.cwd())

    # ── Step 1: Download ──
    _notify(config, "Downloading...", "running")
    dl_result = download_media(
        url=config.url,
        output_dir=output_dir,
        quality=config.quality,
        verbose=config.verbose,
    )
    if dl_result.status == "error":
        _notify(config, "Download failed", "error")
        steps.append({"name": "download", "status": "error", "message": dl_result.message})
        return ToolResult(
            status="error",
            message=f"Pipeline failed at download: {dl_result.message}",
            error_code=dl_result.error_code,
            data={"failed_step": "download", "steps": steps},
        )

    audio_path = dl_result.data.get("audio_path") or dl_result.data.get("video_path")
    caption_path = dl_result.data.get("caption_path")
    title = dl_result.data.get("title", "")
    video_id = dl_result.data.get("video_id", "")

    _notify(config, f"Downloaded: {title or config.url}", "success")
    steps.append(
        {
            "name": "download",
            "status": "success",
            "output_path": audio_path,
            "caption_path": caption_path,
        }
    )

    # ── Step 2: Transcribe (conditional) ──
    if not caption_path:
        _notify(config, "Transcribing...", "running")
        stem = Path(audio_path).stem if audio_path else "output"
        transcribe_output = str(Path(output_dir) / f"{stem}_GeminiUnd.md")

        tr_result = transcribe_media(
            input_path=audio_path,
            output=transcribe_output,
            api_key=config.gemini_api_key or get_gemini_api_key(),
            model=config.model,
            verbose=config.verbose,
        )
        if tr_result.status == "error":
            _notify(config, "Transcription failed", "error")
            steps.append({"name": "transcribe", "status": "error", "message": tr_result.message})
            return ToolResult(
                status="error",
                message=f"Pipeline failed at transcribe: {tr_result.message}",
                error_code=tr_result.error_code,
                data={"failed_step": "transcribe", "steps": steps},
            )

        caption_path = tr_result.data.get("output_path", transcribe_output)
        _notify(config, "Transcribed", "success")
        steps.append({"name": "transcribe", "status": "success", "output_path": caption_path})
    else:
        _notify(config, "Captions found, skipping transcription", "skipped")
        steps.append(
            {
                "name": "transcribe",
                "status": "skipped",
                "reason": "captions found in download",
            }
        )

    # ── Step 3: Align (conditional) ──
    lattifai_key = config.lattifai_api_key or get_lattifai_api_key()
    if not config.skip_align and lattifai_key and audio_path:
        _notify(config, "Aligning with LaiCut...", "running")
        align_result = forced_align(
            audio=audio_path,
            caption=caption_path,
            output_format="json",
            api_key=lattifai_key,
            verbose=config.verbose,
        )
        if align_result.status == "error":
            # Alignment failure is non-fatal — continue with unaligned captions
            _notify(config, f"Alignment skipped: {align_result.message}", "skipped")
            steps.append(
                {
                    "name": "align",
                    "status": "skipped",
                    "reason": f"alignment failed: {align_result.message}",
                }
            )
        else:
            caption_path = align_result.data.get("output_path", caption_path)
            _notify(config, "Aligned (word-level timing)", "success")
            steps.append({"name": "align", "status": "success", "output_path": caption_path})
    else:
        reason = (
            "skip_align flag set"
            if config.skip_align
            else "no LattifAI API key"
            if not lattifai_key
            else "no audio file"
        )
        _notify(config, f"Alignment skipped: {reason}", "skipped")
        steps.append({"name": "align", "status": "skipped", "reason": reason})

    # ── Step 4: Translate (conditional) ──
    if not config.skip_translate:
        _notify(config, f"Translating to {config.target_lang}...", "running")
        stem = Path(caption_path).stem
        ext = Path(caption_path).suffix or ".srt"
        translate_output = str(Path(output_dir) / f"{stem}_Gemini_{config.target_lang}{ext}")

        tl_result = translate_caption(
            input_file=caption_path,
            target_lang=config.target_lang,
            output=translate_output,
            bilingual=config.bilingual,
            api_key=config.gemini_api_key or get_gemini_api_key(),
            model=config.model,
            verbose=config.verbose,
        )
        if tl_result.status == "error":
            _notify(config, "Translation failed", "error")
            steps.append({"name": "translate", "status": "error", "message": tl_result.message})
            return ToolResult(
                status="error",
                message=f"Pipeline failed at translate: {tl_result.message}",
                error_code=tl_result.error_code,
                data={"failed_step": "translate", "steps": steps},
            )

        caption_path = tl_result.data.get("output_path", translate_output)
        _notify(
            config,
            f"Translated ({config.target_lang}{'+ bilingual' if config.bilingual else ''})",
            "success",
        )
        steps.append({"name": "translate", "status": "success", "output_path": caption_path})
    else:
        _notify(config, "Translation skipped", "skipped")
        steps.append(
            {"name": "translate", "status": "skipped", "reason": "skip_translate flag set"}
        )

    # ── Step 5: Convert ──
    # Determine if conversion is needed (input format != output format)
    current_ext = Path(caption_path).suffix.lstrip(".")
    target_fmt = config.output_format

    if current_ext == target_fmt and not config.style:
        # Already in the right format with no style override
        _notify(config, f"Already in {target_fmt} format", "skipped")
        steps.append({"name": "convert", "status": "skipped", "reason": "already in target format"})
        final_path = caption_path
    else:
        _notify(config, f"Converting to {target_fmt}...", "running")
        stem = Path(caption_path).stem
        convert_output = str(Path(output_dir) / f"{stem}.{target_fmt}")

        cv_result = convert_caption(
            input_file=caption_path,
            output=convert_output,
            output_format=target_fmt,
            style=config.style,
            video_file=audio_path if audio_path else None,
            verbose=config.verbose,
        )
        if cv_result.status == "error":
            _notify(config, "Conversion failed", "error")
            steps.append({"name": "convert", "status": "error", "message": cv_result.message})
            return ToolResult(
                status="error",
                message=f"Pipeline failed at convert: {cv_result.message}",
                error_code=cv_result.error_code,
                data={"failed_step": "convert", "steps": steps},
            )

        final_path = cv_result.data.get("output_path", convert_output)
        _notify(config, f"Converted to {target_fmt.upper()}", "success")
        steps.append({"name": "convert", "status": "success", "output_path": final_path})

    # ── Done ──
    return ToolResult(
        status="success",
        message=f"Pipeline complete: {final_path}",
        data={
            "output_path": final_path,
            "title": title,
            "video_id": video_id,
            "steps": steps,
        },
    )
