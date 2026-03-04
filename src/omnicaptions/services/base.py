"""Base types for the service layer.

Provides ToolResult (structured return type) and ErrorCode (error taxonomy)
used by all service implementations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Canonical error codes for omnicaptions tools."""

    MISSING_API_KEY = "MISSING_API_KEY"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_URL = "INVALID_URL"
    INVALID_INPUT = "INVALID_INPUT"
    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    TRANSLATION_FAILED = "TRANSLATION_FAILED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    CONVERSION_FAILED = "CONVERSION_FAILED"


@dataclass
class ToolResult:
    """Structured result returned by every service function.

    Attributes:
        status: Either ``"success"`` or ``"error"``.
        message: Human-readable description of the outcome.
        data: Arbitrary payload (file paths, metrics, etc.).
        error_code: Machine-readable error category (None on success).
    """

    status: str  # "success" | "error"
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error_code: ErrorCode | None = None

    def to_json(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "status": self.status,
            "error_code": self.error_code.value if self.error_code else None,
            "message": self.message,
            "data": self.data,
        }
