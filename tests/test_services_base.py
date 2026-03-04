"""Tests for omnicaptions.services.base module."""

import pytest

from omnicaptions.services.base import ErrorCode, ToolResult


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_all_error_codes_exist(self):
        expected = [
            "MISSING_API_KEY",
            "MISSING_DEPENDENCY",
            "NETWORK_TIMEOUT",
            "UNSUPPORTED_FORMAT",
            "FILE_TOO_LARGE",
            "INVALID_URL",
            "INVALID_INPUT",
            "ALIGNMENT_FAILED",
            "TRANSCRIPTION_FAILED",
            "TRANSLATION_FAILED",
            "DOWNLOAD_FAILED",
            "CONVERSION_FAILED",
        ]
        actual = [e.name for e in ErrorCode]
        assert actual == expected

    def test_error_code_values_match_names(self):
        for code in ErrorCode:
            assert code.value == code.name

    def test_error_code_is_str(self):
        assert isinstance(ErrorCode.MISSING_API_KEY, str)
        assert ErrorCode.MISSING_API_KEY == "MISSING_API_KEY"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_with_data(self):
        result = ToolResult(
            status="success",
            message="Transcription complete",
            data={"output_path": "/tmp/out.srt", "duration": 12.5},
        )
        assert result.status == "success"
        assert result.message == "Transcription complete"
        assert result.data == {"output_path": "/tmp/out.srt", "duration": 12.5}
        assert result.error_code is None

    def test_error_with_error_code(self):
        result = ToolResult(
            status="error",
            message="API key not found",
            error_code=ErrorCode.MISSING_API_KEY,
        )
        assert result.status == "error"
        assert result.message == "API key not found"
        assert result.error_code == ErrorCode.MISSING_API_KEY
        assert result.data == {}

    def test_default_values(self):
        result = ToolResult(status="success")
        assert result.message == ""
        assert result.data == {}
        assert result.error_code is None

    def test_data_default_is_independent(self):
        """Ensure each instance gets its own dict (no shared mutable default)."""
        r1 = ToolResult(status="success")
        r2 = ToolResult(status="success")
        r1.data["key"] = "value"
        assert "key" not in r2.data

    def test_to_json_success(self):
        result = ToolResult(
            status="success",
            message="Done",
            data={"count": 42},
        )
        expected = {
            "status": "success",
            "error_code": None,
            "message": "Done",
            "data": {"count": 42},
        }
        assert result.to_json() == expected

    def test_to_json_error(self):
        result = ToolResult(
            status="error",
            message="File too large",
            error_code=ErrorCode.FILE_TOO_LARGE,
        )
        json_out = result.to_json()
        assert json_out["status"] == "error"
        assert json_out["error_code"] == "FILE_TOO_LARGE"
        assert json_out["message"] == "File too large"
        assert json_out["data"] == {}
