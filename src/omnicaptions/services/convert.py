"""Convert service -- caption format conversion.

Provides standalone functions for converting between caption formats
(SRT, VTT, ASS, TTML, LRC, JSON, Gemini MD) with ASS style presets,
resolution detection, custom colors, and karaoke support.
"""

import json
import logging
import subprocess
from pathlib import Path

from omnicaptions.services.base import ErrorCode, ToolResult

logger = logging.getLogger(__name__)

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
    # Base ratio: 48 / 1080 ~ 0.044
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


def apply_bilingual_colors(cap, line2_color: str):
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


# =============================================================================
# Extension / format helpers
# =============================================================================

_EXT_MAP = {
    "srt": ".srt",
    "vtt": ".vtt",
    "ass": ".ass",
    "ttml": ".ttml",
    "lrc": ".lrc",
    "json": ".json",
}


def _default_ext_for_format(output_format: str | None) -> str:
    """Return the default file extension for *output_format*."""
    if output_format:
        return _EXT_MAP.get(output_format, ".srt")
    return ".srt"


# =============================================================================
# Public API
# =============================================================================


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
    """Convert a caption file between formats.

    Args:
        input_file: Path to the input caption file.
        output: Output file path. Auto-generated if None.
        input_format: Input format override (e.g. "gemini", "srt"). Auto-detected if None.
        output_format: Output format (e.g. "srt", "ass"). Inferred from extension if None.
        style: ASS style preset name ("default", "top", "bilingual", "yellow").
        line1_color: Custom primary color (#RRGGBB) for ASS output.
        line2_color: Custom second-line color (#RRGGBB) for bilingual ASS.
        karaoke: Karaoke effect ("sweep", "instant", "outline") or None.
        fontsize: Font size for ASS output. Auto-calculated from resolution if None.
        resolution: Video resolution string ("1080p", "4k", "1920x1080").
        video_file: Video file path for auto-detecting resolution via ffprobe.
        verbose: Enable verbose logging.

    Returns:
        ToolResult with data containing output_path, output_format, input_format.
    """
    from lattifai.caption import Caption, GeminiReader

    input_path = Path(input_file)
    if not input_path.exists():
        return ToolResult(
            status="error",
            message=f"File not found: {input_file}",
            error_code=ErrorCode.INVALID_INPUT,
        )

    # ------------------------------------------------------------------
    # Resolve video resolution
    # Priority: --resolution > --video > .meta.json > default 1080p
    # ------------------------------------------------------------------
    resolved_resolution = None
    resolution_source = None

    if resolution:
        resolved_resolution = parse_resolution(resolution)
        if resolved_resolution:
            resolution_source = "argument"
        else:
            logger.warning("Invalid resolution '%s'", resolution)

    if not resolved_resolution and video_file:
        video_path = Path(video_file)
        if video_path.exists():
            resolved_resolution = get_video_resolution(video_path)
            if resolved_resolution:
                resolution_source = "ffprobe"
            else:
                logger.warning("Could not detect resolution from '%s'", video_file)
        else:
            logger.warning("Video file not found: %s", video_file)

    if not resolved_resolution:
        resolved_resolution = find_video_metadata(input_path)
        if resolved_resolution:
            resolution_source = "metadata"

    if not resolved_resolution:
        resolved_resolution = (1920, 1080)
        resolution_source = "default"

    if verbose:
        logger.info(
            "Resolution: %dx%d (source: %s)",
            resolved_resolution[0],
            resolved_resolution[1],
            resolution_source,
        )

    # ------------------------------------------------------------------
    # Read input caption
    # ------------------------------------------------------------------
    preserve_newlines = style == "bilingual" or line2_color is not None

    try:
        if input_format == "gemini" or (
            input_format is None and input_path.suffix.lower() == ".md"
        ):
            supervisions = GeminiReader.extract_for_alignment(str(input_path))
            cap = Caption.from_supervisions(supervisions)
            effective_input_format = "gemini"
        else:
            cap = Caption.read(
                str(input_path),
                format=input_format,
                normalize_text=not preserve_newlines,
            )
            effective_input_format = input_format or input_path.suffix.lstrip(".").lower()
    except Exception as e:
        return ToolResult(
            status="error",
            message=f"Failed to read input: {e}",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    # ------------------------------------------------------------------
    # Determine output path and format
    # ------------------------------------------------------------------
    default_ext = _default_ext_for_format(output_format)

    if output is not None:
        output_path = Path(output)
        if output_path.is_dir():
            output_path = output_path / f"{input_path.stem}{default_ext}"
    else:
        output_path = input_path.parent / f"{input_path.stem}{default_ext}"

    # ------------------------------------------------------------------
    # ASS style / metadata handling
    # ------------------------------------------------------------------
    metadata = None
    is_ass_output = output_format == "ass" or str(output_path).lower().endswith(".ass")

    actual_fontsize = (
        fontsize if fontsize is not None else calculate_fontsize(resolved_resolution[1])
    )

    has_resolution = resolution or video_file
    has_ass_options = (
        style or line1_color or line2_color or fontsize is not None or has_resolution
    )
    only_resolution = has_resolution and not style and not line1_color and not line2_color

    if is_ass_output and has_ass_options:
        if only_resolution or (
            fontsize is not None and not style and not line1_color and not line2_color
        ):
            metadata = read_ass_styles(input_path)
            if metadata:
                for s in metadata["ass_styles"].values():
                    s["fontsize"] = actual_fontsize
                metadata["play_res_x"] = resolved_resolution[0]
                metadata["play_res_y"] = resolved_resolution[1]
                if verbose:
                    logger.info(
                        "Preserving styles, resolution: %dx%d, fontsize: %d",
                        resolved_resolution[0],
                        resolved_resolution[1],
                        actual_fontsize,
                    )
            else:
                metadata = build_ass_metadata(
                    "default", actual_fontsize, resolved_resolution
                )
                if verbose:
                    logger.info(
                        "ASS style: default, resolution: %dx%d, fontsize: %d",
                        resolved_resolution[0],
                        resolved_resolution[1],
                        actual_fontsize,
                    )
        else:
            preset = (
                ASS_STYLE_PRESETS.get(style, ASS_STYLE_PRESETS["default"])
                if style
                else ASS_STYLE_PRESETS["default"]
            )
            metadata = build_ass_metadata(
                style or "default", actual_fontsize, resolved_resolution
            )
            if verbose:
                logger.info(
                    "ASS style: %s, resolution: %dx%d, fontsize: %d",
                    style or "default",
                    resolved_resolution[0],
                    resolved_resolution[1],
                    actual_fontsize,
                )

            # Override with custom colors
            if line1_color:
                metadata["ass_styles"]["Default"]["primarycolor"] = hex_to_ass_color(
                    line1_color
                )
                if verbose:
                    logger.info("Line 1 color: %s", line1_color)

            # Handle line2 color (bilingual mode)
            actual_line2_color = line2_color or preset.get("line2_color")
            if actual_line2_color:
                apply_bilingual_colors(cap, actual_line2_color)
                if verbose:
                    logger.info("Line 2 color: %s", actual_line2_color)

    # ------------------------------------------------------------------
    # Karaoke
    # ------------------------------------------------------------------
    karaoke_config = None
    if karaoke:
        from lattifai.caption.config import KaraokeConfig

        karaoke_config = KaraokeConfig(enabled=True, effect=karaoke)
        if verbose:
            logger.info("Karaoke mode: %s", karaoke)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    try:
        cap.write(
            str(output_path),
            output_format=output_format,
            metadata=metadata,
            word_level=karaoke is not None,
            karaoke_config=karaoke_config,
        )
    except Exception as e:
        return ToolResult(
            status="error",
            message=f"Failed to write output: {e}",
            error_code=ErrorCode.CONVERSION_FAILED,
        )

    if verbose:
        logger.info("Converted: %s -> %s", input_path, output_path)

    return ToolResult(
        status="success",
        message=f"Converted to {output_path}",
        data={
            "output_path": str(output_path),
            "output_format": output_format or output_path.suffix.lstrip(".").lower(),
            "input_format": effective_input_format,
        },
    )
