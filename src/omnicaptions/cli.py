"""Command-line interface for OmniCaptions."""

import argparse
import sys
from pathlib import Path

from lattifai.caption import Caption, GeminiReader

from .caption import GeminiCaption, GeminiCaptionConfig
from .config import (
    get_gemini_api_key,
    get_lattifai_api_key,
    save_lattifai_api_key,
    set_gemini_api_key,
)

# =============================================================================
# ASS Style Presets
# =============================================================================

ASS_STYLE_PRESETS = {
    "default": {
        "alignment": 2,  # bottom-center
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
    },
    "top": {
        "alignment": 8,  # top-center
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
    },
    "bilingual": {
        "alignment": 2,
        "primary_color": "#FFFFFF",
        "line2_color": "#FFFF00",  # yellow for second line
        "outline_color": "#000000",
    },
    "yellow": {
        "alignment": 2,
        "primary_color": "#FFFF00",
        "outline_color": "#000000",
    },
}


def hex_to_ass_color(hex_color: str) -> str:
    """Convert #RRGGBB to ASS &HBBGGRR format."""
    hex_color = hex_color.lstrip("#")
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H00{b}{g}{r}"


def apply_bilingual_colors(cap: Caption, line2_color: str) -> Caption:
    """Apply color tag to second line of bilingual captions.

    Converts "Hello\\N你好" to "Hello\\N{\\c&H00FFFF&}你好"

    Note: pysubs2 uses \\N (literal backslash-N) for ASS line breaks,
    not actual newline characters.
    """
    ass_color = hex_to_ass_color(line2_color)

    for sup in cap.supervisions:
        text = sup.text or ""
        # pysubs2 uses \\N for ASS line breaks (literal string, not escape)
        if "\\N" in text:
            parts = text.split("\\N", 1)
            sup.text = f"{parts[0]}\\N{{\\c{ass_color}&}}{parts[1]}"
        elif "\n" in text:
            # Handle actual newline if present
            lines = text.split("\n", 1)
            sup.text = f"{lines[0]}\\N{{\\c{ass_color}&}}{lines[1]}"

    return cap


def build_ass_metadata(preset_name: str) -> dict:
    """Build ASS metadata from preset name."""
    preset = ASS_STYLE_PRESETS.get(preset_name, ASS_STYLE_PRESETS["default"])

    # Build style dict for ASS
    style = {
        "fontname": "Arial",
        "fontsize": 48,
        "primarycolor": hex_to_ass_color(preset["primary_color"]),
        "secondarycolor": "&H000000FF",
        "outlinecolor": hex_to_ass_color(preset["outline_color"]),
        "backcolor": "&H00000000",
        "bold": False,
        "italic": False,
        "outline": 2.0,
        "shadow": 1.0,
        "alignment": preset["alignment"],
        "marginl": 20,
        "marginr": 20,
        "marginv": 20,
    }

    return {
        "ass_styles": {"Default": style},
    }


def ensure_api_key(api_key: str | None = None) -> bool:
    """Ensure API key is available.

    Args:
        api_key: API key from command line argument

    Returns:
        True if API key is available
    """
    # Priority: CLI arg > saved config > env var
    if api_key:
        set_gemini_api_key(api_key)
        return True

    if get_gemini_api_key():
        return True

    print("Error: API key not found.", file=sys.stderr)
    print("", file=sys.stderr)
    print("Set it with one of:", file=sys.stderr)
    print("  1. omnicaptions transcribe -k YOUR_API_KEY ...", file=sys.stderr)
    print("  2. export GEMINI_API_KEY=YOUR_API_KEY", file=sys.stderr)
    print(
        "  3. python -c \"from omnicaptions import set_gemini_api_key; set_gemini_api_key('YOUR_KEY')\"",
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    print("Get a key from: https://aistudio.google.com/apikey", file=sys.stderr)
    return False


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    return path.startswith(("http://", "https://", "youtube.com", "youtu.be"))


def get_default_output_dir(input_path: str) -> Path:
    """Get default output directory based on input type.

    - URL input → current working directory
    - File input → same directory as input file
    """
    if is_url(input_path):
        return Path.cwd()
    else:
        return Path(input_path).parent


def get_stem_from_input(input_path: str) -> str:
    """Extract stem (filename without extension) from input.

    For URLs, extract video ID or use 'output'.
    For files, use the file stem.
    """
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


def cmd_transcribe(args):
    """Transcribe audio/video."""
    if not ensure_api_key(getattr(args, "api_key", None)):
        sys.exit(1)

    config = GeminiCaptionConfig(
        model_name=args.model,
        language=args.language,
        verbose=args.verbose,
        use_thinking=not args.no_thinking,
        use_ytdlp=not getattr(args, "no_ytdlp", False),
        keep_downloaded=getattr(args, "keep_downloaded", False),
    )

    gc = GeminiCaption(config=config)

    if args.prompt:
        gc._transcription_prompt = args.prompt

    result = gc.transcribe(args.input)

    # If --translate specified, translate and output directly
    if args.translate:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(result)
            temp_md = f.name

        output_file = generate_output_path(
            args.input, args.output, f"Gemini_{args.translate}", ".srt"
        )
        gc.translate(temp_md, str(output_file), args.translate, args.bilingual)
        Path(temp_md).unlink()
        if args.verbose:
            print(f"Transcribed and translated to: {output_file}", file=sys.stderr)
    elif args.output:
        output_file = generate_output_path(args.input, args.output, "GeminiUnd", ".md")
        gc.write(result, str(output_file))
        if args.verbose:
            print(f"Saved to: {output_file}", file=sys.stderr)
    else:
        # Default: auto-generate output path
        output_file = generate_output_path(args.input, None, "GeminiUnd", ".md")
        gc.write(result, str(output_file))
        if args.verbose:
            print(f"Saved to: {output_file}", file=sys.stderr)


def cmd_download(args):
    """Download audio/video and captions from video platforms."""
    config = GeminiCaptionConfig(verbose=args.verbose)
    gc = GeminiCaption(config=config)

    output_dir = Path(args.output) if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = getattr(args, "quality", "audio")

    try:
        result = gc.download(args.url, output_dir, quality)
        if result.video_path:
            print(f"Video: {result.video_path}")
        if result.audio_path:
            print(f"Audio: {result.audio_path}")
        if result.caption_path:
            print(f"Caption: {result.caption_path}")
        if result.title:
            print(f"Title: {result.title}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_convert(args):
    """Convert between caption formats."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Check if bilingual style - need to preserve line breaks
    style_name = getattr(args, "style", None)
    preserve_newlines = style_name == "bilingual"

    # Determine input format
    input_format = getattr(args, "from", None)  # --from is a reserved word
    if input_format == "gemini" or (input_format is None and input_path.suffix.lower() == ".md"):
        supervisions = GeminiReader.extract_for_alignment(str(input_path))
        cap = Caption.from_supervisions(supervisions)
    else:
        # Disable normalize_text for bilingual to preserve \N line breaks
        cap = Caption.read(str(input_path), format=input_format, normalize_text=not preserve_newlines)

    # Determine output format and path
    output_format = args.to
    ext_map = {
        "srt": ".srt",
        "vtt": ".vtt",
        "ass": ".ass",
        "ttml": ".ttml",
        "lrc": ".lrc",
        "json": ".json",
    }
    default_ext = ext_map.get(output_format, ".srt") if output_format else ".srt"

    # Convert: only change extension, no suffix
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / f"{get_stem_from_input(args.input)}{default_ext}"
    else:
        output_path = (
            get_default_output_dir(args.input) / f"{get_stem_from_input(args.input)}{default_ext}"
        )

    # Handle ASS style presets
    metadata = None
    style_name = getattr(args, "style", None)
    is_ass_output = output_format == "ass" or str(output_path).lower().endswith(".ass")

    if is_ass_output and style_name:
        preset = ASS_STYLE_PRESETS.get(style_name, ASS_STYLE_PRESETS["default"])
        metadata = build_ass_metadata(style_name)

        # Apply bilingual colors if preset has line2_color
        if "line2_color" in preset:
            apply_bilingual_colors(cap, preset["line2_color"])

        if args.verbose:
            print(f"Using ASS style preset: {style_name}", file=sys.stderr)

    cap.write(str(output_path), output_format=output_format, metadata=metadata)
    if args.verbose:
        print(f"Converted: {input_path} -> {output_path}", file=sys.stderr)


def cmd_translate(args):
    """Translate captions."""
    if not ensure_api_key(getattr(args, "api_key", None)):
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    config = GeminiCaptionConfig(
        model_name=args.model,
        verbose=args.verbose,
    )

    gc = GeminiCaption(config=config)

    # Determine output path
    ext = input_path.suffix if input_path.suffix else ".srt"
    output_path = generate_output_path(args.input, args.output, f"Gemini_{args.language}", ext)

    try:
        gc.translate(args.input, str(output_path), args.language, args.bilingual)
        if args.verbose:
            print(f"Translated: {args.input} -> {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_laicut_align(args):
    """Align audio with caption using LattifAI forced alignment."""
    # Get API key
    api_key = getattr(args, "api_key", None)
    if api_key:
        save_lattifai_api_key(api_key)
    else:
        api_key = get_lattifai_api_key()

    if not api_key:
        from .config import get_lattifai_setup_instructions

        print(get_lattifai_setup_instructions(), file=sys.stderr)
        sys.exit(1)

    audio_path = Path(args.audio)
    caption_path = Path(args.caption)

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not caption_path.exists():
        print(f"Error: Caption file not found: {caption_path}", file=sys.stderr)
        sys.exit(1)

    from lattifai.client import LattifAI
    from lattifai.config import AlignmentConfig, CaptionConfig, ClientConfig

    # Determine output path and format
    if args.output:
        output_path = Path(args.output)
    else:
        # Default to JSON for word-level timing preservation
        suffix = f".{args.format}" if args.format else ".json"
        output_path = caption_path.with_stem(f"{caption_path.stem}_LaiCut").with_suffix(suffix)

    # Enable word_level for JSON output to preserve word timing
    is_json_output = str(output_path).lower().endswith(".json")
    word_level = is_json_output or getattr(args, "word_level", False)

    if args.verbose and is_json_output:
        print("JSON output: word-level timing enabled", file=sys.stderr)

    try:
        client = LattifAI(
            client_config=ClientConfig(api_key=api_key),
            alignment_config=AlignmentConfig(
                model_name="LattifAI/Lattice-1",
                model_hub="modelscope"
            ),
            caption_config=CaptionConfig(
                output_path=str(output_path),
                split_sentence=getattr(args, "split_sentence", False),
                word_level=word_level,
            ),
        )

        client.alignment(
            input_media=str(audio_path),
            input_caption=str(caption_path),
            output_caption_path=str(output_path),
        )
        print(f"LaiCut aligned: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


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
    p_download.add_argument("url", help="Video URL (YouTube, Bilibili, etc.)")
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
