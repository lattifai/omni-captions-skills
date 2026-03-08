"""MCP Server for omnicaptions - thin wrapper over the services layer."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Callable

from omnicaptions.services.base import ToolResult

MCP_INSTALL_HINT = "pip install 'omni-captions-skills[mcp]'"


def _to_mcp(result: ToolResult) -> str:
    """Convert a ToolResult to MCP-friendly text. Raise on error."""
    payload = result.to_json()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if result.status == "error":
        raise ValueError(text)
    return text


@dataclass
class _RegisteredTool:
    """Registered MCP tool metadata."""

    name: str
    description: str
    fn: Callable


class _ToolManager:
    """Minimal tool manager API used by tests and wrappers."""

    def __init__(self):
        self._tools: dict[str, _RegisteredTool] = {}

    def add_tool(self, fn: Callable):
        description = inspect.getdoc(fn) or ""
        self._tools[fn.__name__] = _RegisteredTool(
            name=fn.__name__,
            description=description,
            fn=fn,
        )

    def list_tools(self) -> list[_RegisteredTool]:
        return list(self._tools.values())


class _FastMCPFallback:
    """Fallback MCP server used when `mcp` package is not installed."""

    def __init__(self, name: str):
        self.name = name
        self._tool_manager = _ToolManager()

    def tool(self):
        def _decorator(fn: Callable):
            self._tool_manager.add_tool(fn)
            return fn

        return _decorator

    def run(self, *args, **kwargs):
        raise RuntimeError(
            f"The `mcp` package is required to run the MCP server. Install with: {MCP_INSTALL_HINT}"
        )


def _load_fastmcp():
    """Load FastMCP from mcp package, or a local fallback for tests/dev."""
    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP
    except ImportError:
        return _FastMCPFallback


def create_server():
    """Create and configure the MCP server with all tools."""
    FastMCP = _load_fastmcp()

    mcp = FastMCP(name="omnicaptions")

    @mcp.tool()
    def doctor() -> str:
        """Check environment: ffmpeg, yt-dlp, API keys, and optional dependencies."""
        from omnicaptions.services.doctor import run_doctor

        return _to_mcp(run_doctor())

    @mcp.tool()
    def download(
        url: str,
        output_dir: str | None = None,
        quality: str = "audio",
    ) -> str:
        """Download media from YouTube or other video platforms.

        Args:
            url: Video URL or YouTube video ID.
            output_dir: Directory to save files. Uses temp dir if not specified.
            quality: Quality preset - "audio", "best", "1080p", "720p", "480p", "360p".
        """
        from omnicaptions.services.download import download_media

        return _to_mcp(download_media(url=url, output_dir=output_dir, quality=quality))

    @mcp.tool()
    def transcribe(
        input_path: str,
        output: str | None = None,
        model: str = "gemini-2.5-flash",
        language: str | None = None,
    ) -> str:
        """Transcribe audio/video to text with timestamps using Gemini.

        Args:
            input_path: Local file path or URL to transcribe.
            output: Output file path. Auto-generated if not specified.
            model: Gemini model name.
            language: Language hint (e.g. "en", "zh", "ja").
        """
        from omnicaptions.services.transcribe import transcribe_media

        return _to_mcp(
            transcribe_media(input_path=input_path, output=output, model=model, language=language)
        )

    @mcp.tool()
    def check_captions(url: str) -> str:
        """Check if a video URL has existing captions available for download.

        Useful for deciding whether to transcribe (expensive) or download (free).

        Args:
            url: YouTube or other video URL to check.
        """
        from omnicaptions.services.transcribe import check_existing_captions

        return _to_mcp(check_existing_captions(url=url))

    @mcp.tool()
    def translate(
        input_file: str,
        target_lang: str,
        output: str | None = None,
        bilingual: bool = False,
    ) -> str:
        """Translate a caption file to another language.

        Args:
            input_file: Path to the input caption file.
            target_lang: Target language code (e.g. "zh", "ja", "en").
            output: Output file path. Auto-generated if not specified.
            bilingual: Include original text alongside translation.
        """
        from omnicaptions.services.translate import translate_caption

        return _to_mcp(
            translate_caption(
                input_file=input_file,
                target_lang=target_lang,
                output=output,
                bilingual=bilingual,
            )
        )

    @mcp.tool()
    def convert(
        input_file: str,
        output: str | None = None,
        output_format: str | None = None,
        style: str | None = None,
        resolution: str | None = None,
    ) -> str:
        """Convert a caption file between formats (SRT, VTT, ASS, TTML, Gemini MD, etc.).

        Args:
            input_file: Path to the input caption file.
            output: Output file path. Auto-generated if not specified.
            output_format: Target format (e.g. "srt", "ass", "vtt"). Inferred from output extension if not specified.
            style: ASS style preset name.
            resolution: Video resolution for ASS styling (e.g. "1920x1080").
        """
        from omnicaptions.services.convert import convert_caption

        return _to_mcp(
            convert_caption(
                input_file=input_file,
                output=output,
                output_format=output_format,
                style=style,
                resolution=resolution,
            )
        )

    @mcp.tool()
    def forced_align(
        audio: str,
        caption: str,
        output: str | None = None,
        output_format: str = "json",
    ) -> str:
        """Align audio with a caption file using LattifAI forced alignment for precise timing.

        Args:
            audio: Path to the audio file.
            caption: Path to the caption/subtitle file.
            output: Output file path. Auto-generated if not specified.
            output_format: Output format (default: "json").
        """
        from omnicaptions.services.align import forced_align as _forced_align

        return _to_mcp(
            _forced_align(audio=audio, caption=caption, output=output, output_format=output_format)
        )

    return mcp


def main():
    """Entry point for omnicaptions-mcp command."""
    import argparse

    parser = argparse.ArgumentParser(description="OmniCaptions MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for SSE transport")
    args = parser.parse_args()

    server = create_server()

    try:
        if args.transport == "sse":
            server.run(transport="sse", sse_params={"port": args.port})
        else:
            server.run(transport="stdio")
    except RuntimeError as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
