"""Gemini Caption - Media transcription and caption translation using Gemini API.

Usage:
    from gemini_caption import GeminiCaption

    gc = GeminiCaption(verbose=True)
    result = gc.transcribe("video.mp4")
    gc.translate("input.srt", "output.zh.srt", "zh")
"""

from lattifai.caption import (
    Caption,
    GeminiReader,
    GeminiSegment,
    GeminiWriter,
)

from .caption import DownloadResult, GeminiCaption, GeminiCaptionConfig
from .config import (
    clear_gemini_api_key,
    get_config_path,
    get_gemini_api_key,
    get_lattifai_api_key,
    get_lattifai_setup_instructions,
    get_setup_instructions,
    save_lattifai_api_key,
    set_gemini_api_key,
)

__version__ = "0.1.0"
__all__ = [
    # Main class
    "GeminiCaption",
    "GeminiCaptionConfig",
    "DownloadResult",
    # Gemini API key management
    "get_gemini_api_key",
    "set_gemini_api_key",
    "clear_gemini_api_key",
    "get_config_path",
    "get_setup_instructions",
    # LattifAI API key management
    "get_lattifai_api_key",
    "save_lattifai_api_key",
    "get_lattifai_setup_instructions",
    # Gemini format from lattifai-captions
    "GeminiReader",
    "GeminiWriter",
    "GeminiSegment",
    "Caption",
]
