"""Command-line interface for OmniCaptions."""

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .config import (
    KEY_MAP,
    get_gemini_api_key,
    list_config,
    mask_key,
    set_gemini_api_key,
    set_lattifai_api_key,
)
from .services.base import ErrorCode, ToolResult

# =============================================================================
# Output helpers
# =============================================================================


def _output_result(result: ToolResult, as_json: bool = False):
    """Output ToolResult as JSON or human-readable."""
    if as_json:
        print(json.dumps(result.to_json(), ensure_ascii=False))
    else:
        from .ui import print_error, print_success

        if result.status == "success":
            print_success(result.message)
        else:
            print_error(result.message)
    if result.status == "error":
        sys.exit(1)


# =============================================================================
# Path helpers (still used for CLI-level output path logic)
# =============================================================================


def is_url(path: str) -> bool:
    """Check if path is a URL or a bare YouTube video ID."""
    from .services.download import is_youtube_video_id

    return path.startswith(
        ("http://", "https://", "youtube.com", "youtu.be")
    ) or is_youtube_video_id(path)


def get_default_output_dir(input_path: str) -> Path:
    """Get default output directory based on input type.

    - URL input -> current working directory
    - File input -> same directory as input file
    """
    if is_url(input_path):
        return Path.cwd()
    else:
        return Path(input_path).parent


def get_stem_from_input(input_path: str) -> str:
    """Extract stem (filename without extension) from input.

    For URLs, extract video ID or use 'output'.
    For bare YouTube video IDs, return the ID directly.
    For files, use the file stem.
    """
    from .services.download import is_youtube_video_id

    if is_youtube_video_id(input_path):
        return input_path
    if is_url(input_path):
        # Try to extract YouTube video ID
        import re

        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", input_path)
        if match:
            return match.group(1)
        return "output"
    else:
        return Path(input_path).stem


def generate_output_path(
    input_path: str,
    output: str | Path | None,
    operation: str,
    default_ext: str = ".srt",
) -> Path:
    """Generate output file path, supporting directory output.

    Args:
        input_path: Input file path or URL
        output: Output path (file, directory, or None for auto)
        operation: Operation name for suffix (transcribed, translated, converted)
        default_ext: Default extension if output is a directory

    Returns:
        Full output file path
    """
    stem = get_stem_from_input(input_path)

    if output is None:
        # No output specified: use default directory
        output_dir = get_default_output_dir(input_path)
        return output_dir / f"{stem}_{operation}{default_ext}"

    output_path = Path(output)

    if output_path.is_dir():
        # Directory: generate filename from input
        return output_path / f"{stem}_{operation}{default_ext}"
    else:
        return output_path


# =============================================================================
# Command handlers
# =============================================================================


def cmd_transcribe(args):
    """Transcribe audio/video."""
    from .services.transcribe import transcribe_media

    as_json = getattr(args, "json", False)

    # Handle API key from CLI arg
    api_key = getattr(args, "api_key", None)
    if api_key:
        set_gemini_api_key(api_key)

    # Determine output path
    output = args.output
    if not output and not args.translate:
        output = str(generate_output_path(args.input, args.output, "GeminiUnd", ".md"))

    from .ui import step_spinner

    with step_spinner("Transcribing...", as_json=as_json):
        result = transcribe_media(
            input_path=args.input,
            output=output,
            api_key=api_key or get_gemini_api_key(),
            model=args.model,
            language=args.language,
            use_thinking=not args.no_thinking,
            custom_prompt=args.prompt,
            verbose=args.verbose,
        )

    # If --translate specified and transcription succeeded, translate too
    if args.translate and result.status == "success":
        from .services.translate import translate_caption

        transcription_output = result.data.get("output_path")
        translate_output = str(
            generate_output_path(args.input, args.output, f"Gemini_{args.translate}", ".srt")
        )

        translate_result = translate_caption(
            input_file=transcription_output,
            target_lang=args.translate,
            output=translate_output,
            bilingual=args.bilingual,
            api_key=api_key or get_gemini_api_key(),
            model=args.model,
            verbose=args.verbose,
        )

        # Clean up temporary transcription file if translate succeeded
        if transcription_output and not args.output:
            Path(transcription_output).unlink(missing_ok=True)

        _output_result(translate_result, as_json=as_json)
    else:
        _output_result(result, as_json=as_json)


def cmd_download(args):
    """Download audio/video and captions from video platforms."""
    from .services.download import download_media

    as_json = getattr(args, "json", False)

    output_dir = args.output if args.output else str(Path.cwd())
    quality = getattr(args, "quality", "audio")

    from .ui import step_spinner

    with step_spinner("Downloading...", as_json=as_json):
        result = download_media(
            url=args.url,
            output_dir=output_dir,
            quality=quality,
            verbose=args.verbose,
        )

    if as_json:
        _output_result(result, as_json=True)
    else:
        if result.status == "success":
            data = result.data
            if data.get("video_path"):
                print(f"Video: {data['video_path']}")
            if data.get("audio_path"):
                print(f"Audio: {data['audio_path']}")
            if data.get("caption_path"):
                print(f"Caption: {data['caption_path']}")
            if data.get("title"):
                print(f"Title: {data['title']}")
        else:
            _output_result(result, as_json=False)


def cmd_convert(args):
    """Convert between caption formats."""
    from .services.convert import convert_caption

    as_json = getattr(args, "json", False)

    result = convert_caption(
        input_file=args.input,
        output=args.output,
        input_format=getattr(args, "from", None),
        output_format=getattr(args, "to", None),
        style=getattr(args, "style", None),
        line1_color=getattr(args, "line1_color", None),
        line2_color=getattr(args, "line2_color", None),
        karaoke=getattr(args, "karaoke", None),
        fontsize=getattr(args, "fontsize", None),
        resolution=getattr(args, "resolution", None),
        video_file=getattr(args, "video", None),
        verbose=getattr(args, "verbose", False),
    )

    if as_json:
        _output_result(result, as_json=True)
    else:
        # Old behavior: silent on success (only verbose prints in the service),
        # error prints to stderr and exits
        if result.status == "error":
            _output_result(result, as_json=False)


def cmd_translate(args):
    """Translate captions."""
    from .services.translate import translate_caption

    as_json = getattr(args, "json", False)

    # Handle API key from CLI arg
    api_key = getattr(args, "api_key", None)
    if api_key:
        set_gemini_api_key(api_key)

    # Determine output path
    input_path = Path(args.input)
    ext = input_path.suffix if input_path.suffix else ".srt"
    output = args.output or str(
        generate_output_path(args.input, args.output, f"Gemini_{args.language}", ext)
    )

    from .ui import step_spinner

    with step_spinner("Translating...", as_json=as_json):
        result = translate_caption(
            input_file=args.input,
            target_lang=args.language,
            output=output,
            bilingual=args.bilingual,
            api_key=api_key or get_gemini_api_key(),
            model=args.model,
            verbose=args.verbose,
        )

    _output_result(result, as_json=as_json)


def cmd_doctor(args):
    """Check environment dependencies."""
    from omnicaptions.services.doctor import run_doctor

    result = run_doctor(verbose=args.verbose)

    if getattr(args, "json", False):
        _output_result(result, as_json=True)
    else:
        from .ui import print_table

        rows = []
        for check in result.data.get("checks", []):
            status_str = "[green]ok[/green]" if check["status"] == "ok" else "[red]MISSING[/red]"
            help_str = check.get("help") or ""
            rows.append([check["name"], status_str, help_str])
        print_table(["Check", "Status", "Help"], rows)
        if result.status == "error":
            sys.exit(1)


def cmd_laicut_align(args):
    """Align audio with caption using LattifAI forced alignment."""
    from .services.align import forced_align

    as_json = getattr(args, "json", False)

    # Handle API key from CLI arg
    api_key = getattr(args, "api_key", None)

    result = forced_align(
        audio=args.audio,
        caption=args.caption,
        output=args.output,
        output_format=getattr(args, "format", None) or "json",
        api_key=api_key,
        split_sentence=getattr(args, "split_sentence", False),
        verbose=getattr(args, "verbose", False),
    )

    _output_result(result, as_json=as_json)


def cmd_pipeline(args):
    """Run the full pipeline: URL -> bilingual subtitles."""
    from .services.pipeline import PipelineConfig, run_pipeline
    from .ui import step_progress

    as_json = getattr(args, "json", False)

    steps = ["download", "transcribe", "align", "translate", "convert"]
    tracker = step_progress(steps, as_json=as_json)

    def on_step(name: str, status: str):
        tracker.advance(name, status)

    config = PipelineConfig(
        url=args.url,
        target_lang=getattr(args, "language", "zh"),
        output_format=getattr(args, "format", "srt"),
        output_dir=getattr(args, "output", None),
        bilingual=getattr(args, "bilingual", True),
        style=getattr(args, "style", None),
        model=getattr(args, "model", "gemini-3-flash-preview"),
        quality=getattr(args, "quality", "audio"),
        skip_align=getattr(args, "no_align", False),
        skip_translate=getattr(args, "no_translate", False),
        gemini_api_key=getattr(args, "api_key", None) or get_gemini_api_key(),
        lattifai_api_key=getattr(args, "lattifai_key", None),
        verbose=getattr(args, "verbose", False),
        on_step=on_step,
    )

    result = run_pipeline(config)
    _output_result(result, as_json=as_json)


def cmd_config(args):
    """Manage configuration (show / set / get)."""
    as_json = getattr(args, "json", False)
    action = args.config_action

    if action == "show":
        config_data = list_config()
        if as_json:
            result = ToolResult(
                status="success",
                message="Configuration listing",
                data=config_data,
            )
            _output_result(result, as_json=True)
        else:
            print("OmniCaptions Configuration:")
            print("-" * 50)
            for name, info in config_data.items():
                status = "SET" if info["set"] else "NOT SET"
                value_display = info["value"] if info["set"] else "-"
                print(f"  {name:<20} {status:<10} {value_display:<20} (source: {info['source']})")

    elif action == "set":
        friendly_key = args.key.lower()
        if friendly_key not in KEY_MAP:
            result = ToolResult(
                status="error",
                message=f"Unknown config key: {args.key}. Valid keys: {', '.join(KEY_MAP.keys())}",
                error_code=ErrorCode.INVALID_INPUT,
            )
            _output_result(result, as_json=as_json)
            return
        env_name = KEY_MAP[friendly_key]
        if env_name == "GEMINI_API_KEY":
            set_gemini_api_key(args.value)
        elif env_name == "LATTIFAI_API_KEY":
            set_lattifai_api_key(args.value)
        result = ToolResult(
            status="success",
            message=f"Saved {friendly_key} to config file ({mask_key(args.value)})",
            data={"key": friendly_key, "masked_value": mask_key(args.value)},
        )
        _output_result(result, as_json=as_json)

    elif action == "get":
        friendly_key = args.key.lower()
        if friendly_key not in KEY_MAP:
            result = ToolResult(
                status="error",
                message=f"Unknown config key: {args.key}. Valid keys: {', '.join(KEY_MAP.keys())}",
                error_code=ErrorCode.INVALID_INPUT,
            )
            _output_result(result, as_json=as_json)
            return
        config_data = list_config()
        info = config_data[friendly_key]
        if as_json:
            result = ToolResult(
                status="success",
                message=f"{friendly_key}: {info['value']} (source: {info['source']})",
                data={friendly_key: info},
            )
            _output_result(result, as_json=True)
        else:
            if info["set"]:
                print(f"{friendly_key}: {info['value']} (source: {info['source']})")
            else:
                print(f"{friendly_key}: not set")


# =============================================================================
# Main entry point
# =============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="omnicaptions",
        description="Transcribe, translate, and convert captions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Transcribe
    p_transcribe = subparsers.add_parser("transcribe", help="Transcribe audio/video")
    p_transcribe.add_argument("input", help="URL or local file path")
    p_transcribe.add_argument("-k", "--api-key", dest="api_key", help="Gemini API key")
    p_transcribe.add_argument("-p", "--prompt", help="Custom system prompt")
    p_transcribe.add_argument("-o", "--output", help="Output file or directory")
    p_transcribe.add_argument("-m", "--model", default="gemini-3-flash-preview", help="Model name")
    p_transcribe.add_argument("-l", "--language", help="Force language (zh, en, ja)")
    p_transcribe.add_argument(
        "-t", "--translate", metavar="LANG", help="Translate to language (zh, en, ja)"
    )
    p_transcribe.add_argument(
        "--bilingual", action="store_true", help="Bilingual output (with --translate)"
    )
    p_transcribe.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    p_transcribe.add_argument("--no-ytdlp", action="store_true", help="Disable yt-dlp download")
    p_transcribe.add_argument(
        "--keep-downloaded", action="store_true", help="Keep downloaded files"
    )
    p_transcribe.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_transcribe.set_defaults(func=cmd_transcribe)

    # Download
    p_download = subparsers.add_parser(
        "download", help="Download audio/video and captions from URL"
    )
    p_download.add_argument("url", help="Video URL or YouTube video ID (e.g. e882eXLtwkI)")
    p_download.add_argument("-o", "--output", help="Output directory (default: current)")
    p_download.add_argument(
        "-q",
        "--quality",
        default="audio",
        choices=["audio", "best", "1080p", "720p", "480p", "360p"],
        help="Quality: audio (default), best, 1080p, 720p, 480p, 360p",
    )
    p_download.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_download.set_defaults(func=cmd_download)

    # Convert
    p_convert = subparsers.add_parser("convert", help="Convert caption formats")
    p_convert.add_argument("input", help="Input caption file")
    p_convert.add_argument(
        "-o", "--output", help="Output file or directory (default: same dir as input)"
    )
    p_convert.add_argument(
        "-f",
        "--from",
        dest="from",
        metavar="FMT",
        help="Input format (auto-detect if not specified). Formats: srt, vtt, ass, txt, json, gemini, etc.",
    )
    p_convert.add_argument(
        "-t",
        "--to",
        dest="to",
        metavar="FMT",
        help="Output format (from extension if not specified). Formats: srt, vtt, ass, ttml, lrc, json, etc.",
    )
    p_convert.add_argument(
        "-s",
        "--style",
        choices=["default", "top", "bilingual", "yellow"],
        help="ASS style preset: default (white, bottom), top (white, top), bilingual (white+yellow), yellow",
    )
    p_convert.add_argument(
        "--line1-color",
        metavar="COLOR",
        help="First line color (#RRGGBB), e.g. #FFFFFF for white",
    )
    p_convert.add_argument(
        "--line2-color",
        metavar="COLOR",
        help="Second line color (#RRGGBB), e.g. #FFFF00 for yellow",
    )
    p_convert.add_argument(
        "--karaoke",
        nargs="?",
        const="sweep",
        choices=["sweep", "instant", "outline"],
        help="Enable karaoke mode (requires word-level timing). Effects: sweep (default), instant, outline",
    )
    p_convert.add_argument(
        "--fontsize",
        type=int,
        default=None,
        metavar="SIZE",
        help="Font size for ASS output (auto-calculated from resolution if not set)",
    )
    p_convert.add_argument(
        "--resolution",
        metavar="RES",
        help="Video resolution for font scaling: 720p, 1080p, 4k, or WxH (e.g. 1920x1080)",
    )
    p_convert.add_argument(
        "--video",
        metavar="FILE",
        help="Video file to auto-detect resolution (overridden by --resolution)",
    )
    p_convert.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_convert.set_defaults(func=cmd_convert)

    # Translate
    p_translate = subparsers.add_parser("translate", help="Translate captions")
    p_translate.add_argument("input", help="Input caption file")
    p_translate.add_argument("-k", "--api-key", dest="api_key", help="Gemini API key")
    p_translate.add_argument(
        "-o", "--output", help="Output file or directory (default: same dir as input)"
    )
    p_translate.add_argument("-l", "--language", required=True, help="Target language (zh, en, ja)")
    p_translate.add_argument(
        "--bilingual", action="store_true", help="Output original + translation"
    )
    p_translate.add_argument("-m", "--model", default="gemini-3-flash-preview", help="Model name")
    p_translate.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_translate.set_defaults(func=cmd_translate)

    # LaiCut Align
    p_laicut = subparsers.add_parser(
        "LaiCut", help="Align audio/video with captions using LattifAI"
    )
    p_laicut.add_argument("audio", help="Audio/video file path")
    p_laicut.add_argument("caption", help="Caption file (SRT/VTT/ASS/LRC/TXT/MD)")
    p_laicut.add_argument(
        "-o", "--output", help="Output file path (default: <caption>_LaiCut.<ext>)"
    )
    p_laicut.add_argument("-f", "--format", help="Output format (default: same as input)")
    p_laicut.add_argument("-k", "--api-key", dest="api_key", help="LattifAI API key")
    p_laicut.add_argument(
        "--split-sentence",
        action="store_true",
        dest="split_sentence",
        help="Re-segment captions intelligently based on punctuation and semantics (AI-powered)",
    )
    p_laicut.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_laicut.set_defaults(func=cmd_laicut_align)

    # Doctor
    p_doctor = subparsers.add_parser("doctor", help="Check environment dependencies and API keys")
    p_doctor.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_doctor.set_defaults(func=cmd_doctor)

    # Pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="URL to bilingual subtitles in one command")
    p_pipeline.add_argument("url", help="Video URL or YouTube video ID")
    p_pipeline.add_argument("-l", "--language", default="zh", help="Target language (default: zh)")
    p_pipeline.add_argument("-f", "--format", default="srt", help="Output format (srt, ass, vtt)")
    p_pipeline.add_argument("-o", "--output", help="Output directory")
    p_pipeline.add_argument("-k", "--api-key", dest="api_key", help="Gemini API key")
    p_pipeline.add_argument("--lattifai-key", dest="lattifai_key", help="LattifAI API key")
    p_pipeline.add_argument("-m", "--model", default="gemini-3-flash-preview", help="Model name")
    p_pipeline.add_argument(
        "-q",
        "--quality",
        default="audio",
        choices=["audio", "best", "1080p", "720p", "480p", "360p"],
        help="Download quality (default: audio)",
    )
    p_pipeline.add_argument(
        "--bilingual", action="store_true", default=True, help="Bilingual output (default)"
    )
    p_pipeline.add_argument("--no-bilingual", dest="bilingual", action="store_false")
    p_pipeline.add_argument(
        "--style",
        choices=["default", "top", "bilingual", "yellow"],
        help="ASS style preset",
    )
    p_pipeline.add_argument("--no-align", action="store_true", help="Skip forced alignment")
    p_pipeline.add_argument("--no-translate", action="store_true", help="Skip translation")
    p_pipeline.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_pipeline.set_defaults(func=cmd_pipeline)

    # Config
    p_config = subparsers.add_parser("config", help="Manage configuration")
    config_sub = p_config.add_subparsers(dest="config_action", required=True)

    p_config_show = config_sub.add_parser("show", help="Show all configuration")
    p_config_set = config_sub.add_parser("set", help="Set a config value")
    p_config_set.add_argument("key", help="Config key (GEMINI_API_KEY, LATTIFAI_API_KEY)")
    p_config_set.add_argument("value", help="Value to set")
    p_config_get = config_sub.add_parser("get", help="Get a config value")
    p_config_get.add_argument("key", help="Config key (GEMINI_API_KEY, LATTIFAI_API_KEY)")
    p_config.set_defaults(func=cmd_config)

    # Add --json flag to every subparser
    for p in [
        p_transcribe,
        p_download,
        p_convert,
        p_translate,
        p_laicut,
        p_doctor,
        p_pipeline,
        p_config_show,
        p_config_set,
        p_config_get,
    ]:
        p.add_argument(
            "--json", action="store_true", help="Output as JSON (for agent/script consumption)"
        )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
