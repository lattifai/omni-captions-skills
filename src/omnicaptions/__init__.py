"""OmniCaptions - AI-powered media transcription, translation, and caption tools.

Usage:
    from omnicaptions.services import ToolResult, ErrorCode
    from omnicaptions.services.transcribe import transcribe_media
    from omnicaptions.services.download import download_media
"""

from lattifai.caption import (
    Caption,
    GeminiReader,
    GeminiSegment,
    GeminiWriter,
)

from .services import ErrorCode, ToolResult
from .services.download import DownloadResult
from .config import (
    clear_gemini_api_key,
    get_config_path,
    get_gemini_api_key,
    get_lattifai_api_key,
    get_lattifai_setup_instructions,
    get_setup_instructions,
    set_gemini_api_key,
    set_lattifai_api_key,
)

__version__ = "0.2.0"
__all__ = [
    # Services layer
    "ErrorCode",
    "ToolResult",
    "DownloadResult",
    # Config
    "get_gemini_api_key",
    "set_gemini_api_key",
    "clear_gemini_api_key",
    "get_config_path",
    "get_setup_instructions",
    "get_lattifai_api_key",
    "set_lattifai_api_key",
    "get_lattifai_setup_instructions",
    # Formats from lattifai-captions
    "GeminiReader",
    "GeminiWriter",
    "GeminiSegment",
    "Caption",
]
