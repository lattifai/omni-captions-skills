"""Doctor service - preflight environment check."""

import shutil

from .base import ToolResult
from omnicaptions.config import get_gemini_api_key, get_lattifai_api_key


def run_doctor(verbose: bool = False) -> ToolResult:
    """Check environment dependencies and API keys."""
    checks = []
    all_ok = True

    # ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    checks.append(
        {
            "name": "ffmpeg",
            "status": "ok" if ffmpeg_path else "missing",
            "help": (
                "brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
                if not ffmpeg_path
                else None
            ),
        }
    )
    if not ffmpeg_path:
        all_ok = False

    # yt-dlp
    ytdlp_path = shutil.which("yt-dlp")
    checks.append(
        {
            "name": "yt-dlp",
            "status": "ok" if ytdlp_path else "missing",
            "help": "pip install yt-dlp" if not ytdlp_path else None,
        }
    )
    if not ytdlp_path:
        all_ok = False

    # Gemini API key
    gemini_key = get_gemini_api_key()
    checks.append(
        {
            "name": "gemini_api_key",
            "status": "ok" if gemini_key else "missing",
            "help": (
                "Get key from https://aistudio.google.com/apikey then: export GEMINI_API_KEY=<key>"
                if not gemini_key
                else None
            ),
        }
    )

    # LattifAI API key
    lattifai_key = get_lattifai_api_key()
    checks.append(
        {
            "name": "lattifai_api_key",
            "status": "ok" if lattifai_key else "missing",
            "help": (
                "Get key from https://lattifai.com/dashboard then: export LATTIFAI_API_KEY=<key>"
                if not lattifai_key
                else None
            ),
        }
    )

    # lattifai package
    try:
        import lattifai.client  # noqa: F401

        checks.append({"name": "lattifai", "status": "ok", "help": None})
    except ImportError:
        checks.append(
            {
                "name": "lattifai",
                "status": "missing",
                "help": "pip install 'omni-captions-skills[laicut]'",
            }
        )

    return ToolResult(
        status="success" if all_ok else "error",
        message="All checks passed" if all_ok else "Some checks failed",
        data={"checks": checks},
    )
