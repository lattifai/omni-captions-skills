"""Command-line interface for OmniCaptions."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lattifai.caption import Caption, GeminiReader

from .caption import GeminiCaption, GeminiCaptionConfig
from .config import (
    get_gemini_api_key,
    get_lattifai_api_key,
    set_gemini_api_key,
    set_lattifai_api_key,
)

# =============================================================================
# Resolution Detection and Font Size Calculation
# =============================================================================

# Standard resolution presets: (width, height)
RESOLUTION_PRESETS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}


def get_video_resolution(video_path: str | Path) -> tuple[int, int] | None:
    """Get video resolution using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        (width, height) tuple or None if detection fails
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            width = streams[0].get("width")
            height = streams[0].get("height")
            if width and height:
                return (int(width), int(height))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return None


def calculate_fontsize(height: int) -> int:
    """Calculate recommended font size based on video height.

    The font size scales linearly with video height.
    Base: 48 for 1080p (height=1080)

    Args:
        height: Video height in pixels

    Returns:
        Recommended font size
    """
    # Base ratio: 48 / 1080 ≈ 0.044
    base_ratio = 48 / 1080
    fontsize = int(height * base_ratio)
    # Clamp to reasonable range
    return max(24, min(fontsize, 144))


def parse_resolution(resolution_str: str) -> tuple[int, int] | None:
    """Parse resolution string to (width, height).

    Args:
        resolution_str: Resolution like "1080p", "4k", or "1920x1080"

    Returns:
        (width, height) tuple or None if invalid
    """
    resolution_str = resolution_str.lower().strip()

    # Check presets
    if resolution_str in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[resolution_str]

    # Try WxH format
    if "x" in resolution_str:
        parts = resolution_str.split("x")
        if len(parts) == 2:
            try:
                return (int(parts[0]), int(parts[1]))
            except ValueError:
                pass

    return None


def find_video_metadata(caption_path: Path) -> tuple[int, int] | None:
    """Find video metadata from .meta.json file.

    Looks for .meta.json file in the same directory with matching video ID prefix.
    This metadata is saved by the download command from yt-dlp.

    Args:
        caption_path: Path to caption file (e.g., "abc123.en.srt")

    Returns:
        (width, height) tuple or None if not found
    """
    # Try to find metadata file based on video ID prefix
    # Common patterns: "abc123.en.srt" -> "abc123.meta.json"
    stem = caption_path.stem
    parent = caption_path.parent

    # Try different possible prefixes
    # e.g., "abc123.en.srt" has stem "abc123.en", try "abc123"
    parts = stem.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        meta_path = parent / f"{prefix}.meta.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                width = data.get("width")
                height = data.get("height")
                if width and height:
                    return (int(width), int(height))
            except (json.JSONDecodeError, ValueError):
                pass

    # Also try first part only (most common case)
    if parts:
        meta_path = parent / f"{parts[0]}.meta.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                width = data.get("width")
                height = data.get("height")
                if width and height:
                    return (int(width), int(height))
            except (json.JSONDecodeError, ValueError):
                pass

    return None


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


def build_ass_metadata(
    preset_name: str, fontsize: int = 48, resolution: tuple[int, int] = (1920, 1080)
) -> dict:
    """Build ASS metadata from preset name.

    Args:
        preset_name: Style preset name
        fontsize: Font size (default 48 for 1080p, 96 for 4K, 32 for 720p)
        resolution: Video resolution (width, height), default 1920x1080
    """
    preset = ASS_STYLE_PRESETS.get(preset_name, ASS_STYLE_PRESETS["default"])

    # Build style dict for ASS
    style = {
        "fontname": "Arial",
        "fontsize": fontsize,
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
        "play_res_x": resolution[0],
        "play_res_y": resolution[1],
        "ass_styles": {"Default": style},
    }


def color_to_ass(color) -> str:
    """Convert pysubs2 Color to ASS color string (&HAABBGGRR)."""
    return f"&H{color.a:02X}{color.b:02X}{color.g:02X}{color.r:02X}"


def read_ass_styles(input_path: Path) -> dict | None:
    """Read existing ASS styles from input file.

    Returns dict with ass_styles, play_res_x, play_res_y or None if not an ASS file.
    """
    if input_path.suffix.lower() not in (".ass", ".ssa"):
        return None

    try:
        import pysubs2

        subs = pysubs2.load(str(input_path))
        styles = {}
        for name, style in subs.styles.items():
            styles[name] = {
                "fontname": style.fontname,
                "fontsize": style.fontsize,
                "primarycolor": color_to_ass(style.primarycolor),
                "secondarycolor": color_to_ass(style.secondarycolor),
                "outlinecolor": color_to_ass(style.outlinecolor),
                "backcolor": color_to_ass(style.backcolor),
                "bold": style.bold,
                "italic": style.italic,
                "outline": style.outline,
                "shadow": style.shadow,
                "alignment": style.alignment,
                "marginl": style.marginl,
                "marginr": style.marginr,
                "marginv": style.marginv,
            }
        result = {"ass_styles": styles}
        # Preserve PlayResX/PlayResY from original file
        if hasattr(subs, "info"):
            if "PlayResX" in subs.info:
                result["play_res_x"] = int(subs.info["PlayResX"])
            if "PlayResY" in subs.info:
                result["play_res_y"] = int(subs.info["PlayResY"])
        return result
    except Exception:
        return None


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

    # Check if bilingual mode - need to preserve line breaks
    style_name = getattr(args, "style", None)
    line1_color = getattr(args, "line1_color", None)
    line2_color = getattr(args, "line2_color", None)
    fontsize = getattr(args, "fontsize", None)
    resolution_str = getattr(args, "resolution", None)
    video_file = getattr(args, "video", None)
    preserve_newlines = style_name == "bilingual" or line2_color is not None

    # Determine video resolution (priority: --resolution > --video > .meta.json > default)
    resolution = None
    resolution_source = None

    if resolution_str:
        # 1. Explicit --resolution argument
        resolution = parse_resolution(resolution_str)
        if resolution:
            resolution_source = "argument"
        else:
            print(f"Warning: Invalid resolution '{resolution_str}'", file=sys.stderr)

    if not resolution and video_file:
        # 2. Detect from --video file using ffprobe
        video_path = Path(video_file)
        if video_path.exists():
            resolution = get_video_resolution(video_path)
            if resolution:
                resolution_source = "ffprobe"
            else:
                print(f"Warning: Could not detect resolution from '{video_file}'", file=sys.stderr)
        else:
            print(f"Warning: Video file not found: {video_file}", file=sys.stderr)

    if not resolution:
        # 3. Try to find .meta.json file from download command
        resolution = find_video_metadata(input_path)
        if resolution:
            resolution_source = "metadata"

    if not resolution:
        # 4. Default to 1080p
        resolution = (1920, 1080)
        resolution_source = "default"

    if args.verbose:
        print(f"Resolution: {resolution[0]}x{resolution[1]} (source: {resolution_source})", file=sys.stderr)

    # Determine input format
    input_format = getattr(args, "from", None)  # --from is a reserved word
    if input_format == "gemini" or (input_format is None and input_path.suffix.lower() == ".md"):
        supervisions = GeminiReader.extract_for_alignment(str(input_path))
        cap = Caption.from_supervisions(supervisions)
    else:
        # Disable normalize_text for bilingual to preserve \N line breaks
        cap = Caption.read(
            str(input_path), format=input_format, normalize_text=not preserve_newlines
        )

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

    # Handle ASS style presets and custom colors
    metadata = None
    is_ass_output = output_format == "ass" or str(output_path).lower().endswith(".ass")

    # Calculate fontsize from resolution if not specified
    actual_fontsize = fontsize if fontsize is not None else calculate_fontsize(resolution[1])

    # Only create metadata if user explicitly specified ASS options or resolution
    has_resolution = resolution_str or video_file
    has_ass_options = style_name or line1_color or line2_color or fontsize is not None or has_resolution
    only_resolution = has_resolution and not style_name and not line1_color and not line2_color

    if is_ass_output and has_ass_options:
        # If only resolution/fontsize specified, try to preserve existing styles from input ASS
        if only_resolution or (fontsize is not None and not style_name and not line1_color and not line2_color):
            metadata = read_ass_styles(input_path)
            if metadata:
                # Update fontsize and PlayRes in all styles
                for style in metadata["ass_styles"].values():
                    style["fontsize"] = actual_fontsize
                metadata["play_res_x"] = resolution[0]
                metadata["play_res_y"] = resolution[1]
                if args.verbose:
                    print(
                        f"Preserving styles, resolution: {resolution[0]}x{resolution[1]}, fontsize: {actual_fontsize}",
                        file=sys.stderr,
                    )
            else:
                # Input is not ASS, create new metadata
                metadata = build_ass_metadata("default", actual_fontsize, resolution)
                if args.verbose:
                    print(
                        f"ASS style: default, resolution: {resolution[0]}x{resolution[1]}, fontsize: {actual_fontsize}",
                        file=sys.stderr,
                    )
        else:
            # Use preset or custom style
            preset = (
                ASS_STYLE_PRESETS.get(style_name, ASS_STYLE_PRESETS["default"])
                if style_name
                else ASS_STYLE_PRESETS["default"]
            )
            metadata = build_ass_metadata(style_name or "default", actual_fontsize, resolution)
            if args.verbose:
                print(
                    f"ASS style: {style_name or 'default'}, resolution: {resolution[0]}x{resolution[1]}, fontsize: {actual_fontsize}",
                    file=sys.stderr,
                )

            # Override with custom colors
            if line1_color:
                metadata["ass_styles"]["Default"]["primarycolor"] = hex_to_ass_color(line1_color)
                if args.verbose:
                    print(f"Line 1 color: {line1_color}", file=sys.stderr)

            # Handle line2 color (bilingual mode)
            actual_line2_color = line2_color or preset.get("line2_color")
            if actual_line2_color:
                apply_bilingual_colors(cap, actual_line2_color)
                if args.verbose:
                    print(f"Line 2 color: {actual_line2_color}", file=sys.stderr)

    # Handle karaoke mode
    karaoke_config = None
    karaoke_effect = getattr(args, "karaoke", None)
    if karaoke_effect:
        from lattifai.caption.config import KaraokeConfig

        karaoke_config = KaraokeConfig(enabled=True, effect=karaoke_effect)
        if args.verbose:
            print(f"Karaoke mode: {karaoke_effect}", file=sys.stderr)

    cap.write(
        str(output_path),
        output_format=output_format,
        metadata=metadata,
        word_level=karaoke_effect is not None,
        karaoke_config=karaoke_config,
    )
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
        set_lattifai_api_key(api_key)
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
                model_name="LattifAI/Lattice-1", model_hub="modelscope"
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
