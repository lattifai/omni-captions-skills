"""Tests for the MCP server."""

import json
from unittest.mock import patch

import pytest

from omnicaptions.mcp_server import MCP_INSTALL_HINT, _to_mcp, create_server
from omnicaptions.services.base import ErrorCode, ToolResult

# --- _to_mcp tests ---


class TestToMcp:
    def test_success_result(self):
        result = ToolResult(status="success", message="Done", data={"path": "/tmp/out.srt"})
        text = _to_mcp(result)
        parsed = json.loads(text)
        assert parsed["status"] == "success"
        assert parsed["data"]["path"] == "/tmp/out.srt"

    def test_error_result_raises(self):
        result = ToolResult(
            status="error",
            message="API key missing",
            error_code=ErrorCode.MISSING_API_KEY,
        )
        with pytest.raises(ValueError) as exc_info:
            _to_mcp(result)
        parsed = json.loads(str(exc_info.value))
        assert parsed["status"] == "error"
        assert parsed["error_code"] == "MISSING_API_KEY"

    def test_empty_data(self):
        result = ToolResult(status="success", message="OK")
        text = _to_mcp(result)
        parsed = json.loads(text)
        assert parsed["data"] == {}


# --- Tool tests (mock services) ---


@pytest.fixture
def mcp():
    return create_server()


class TestDoctorTool:
    @patch("omnicaptions.services.doctor.run_doctor")
    def test_doctor_success(self, mock_doctor, mcp):
        mock_doctor.return_value = ToolResult(
            status="success", message="All checks passed", data={"checks": []}
        )
        tool_fn = _get_tool_fn(mcp, "doctor")
        result = tool_fn()
        parsed = json.loads(result)
        assert parsed["status"] == "success"


class TestDownloadTool:
    @patch("omnicaptions.services.download.download_media")
    def test_download_success(self, mock_dl, mcp):
        mock_dl.return_value = ToolResult(
            status="success",
            message="Downloaded",
            data={"audio_path": "/tmp/audio.m4a"},
        )
        tool_fn = _get_tool_fn(mcp, "download")
        result = tool_fn(url="https://youtube.com/watch?v=test")
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_dl.assert_called_once_with(
            url="https://youtube.com/watch?v=test", output_dir=None, quality="audio"
        )

    @patch("omnicaptions.services.download.download_media")
    def test_download_error(self, mock_dl, mcp):
        mock_dl.return_value = ToolResult(
            status="error", message="Invalid URL", error_code=ErrorCode.INVALID_URL
        )
        tool_fn = _get_tool_fn(mcp, "download")
        with pytest.raises(ValueError):
            tool_fn(url="not-a-url")


class TestTranscribeTool:
    @patch("omnicaptions.services.transcribe.transcribe_media")
    def test_transcribe_success(self, mock_tr, mcp):
        mock_tr.return_value = ToolResult(
            status="success",
            message="Transcribed",
            data={"output_path": "/tmp/out.md"},
        )
        tool_fn = _get_tool_fn(mcp, "transcribe")
        result = tool_fn(input_path="/tmp/audio.m4a")
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_tr.assert_called_once_with(
            input_path="/tmp/audio.m4a", output=None, model="gemini-2.5-flash", language=None
        )


class TestCheckCaptionsTool:
    @patch("omnicaptions.services.transcribe.check_existing_captions")
    def test_check_captions_success(self, mock_check, mcp):
        mock_check.return_value = ToolResult(
            status="success",
            message="Captions found",
            data={"has_manual_captions": True},
        )
        tool_fn = _get_tool_fn(mcp, "check_captions")
        result = tool_fn(url="https://youtube.com/watch?v=test")
        parsed = json.loads(result)
        assert parsed["data"]["has_manual_captions"] is True


class TestTranslateTool:
    @patch("omnicaptions.services.translate.translate_caption")
    def test_translate_success(self, mock_tl, mcp):
        mock_tl.return_value = ToolResult(
            status="success",
            message="Translated",
            data={"output_path": "/tmp/out_zh.srt"},
        )
        tool_fn = _get_tool_fn(mcp, "translate")
        result = tool_fn(input_file="/tmp/in.srt", target_lang="zh", bilingual=True)
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_tl.assert_called_once_with(
            input_file="/tmp/in.srt", target_lang="zh", output=None, bilingual=True
        )


class TestConvertTool:
    @patch("omnicaptions.services.convert.convert_caption")
    def test_convert_success(self, mock_cv, mcp):
        mock_cv.return_value = ToolResult(
            status="success",
            message="Converted",
            data={"output_path": "/tmp/out.ass"},
        )
        tool_fn = _get_tool_fn(mcp, "convert")
        result = tool_fn(input_file="/tmp/in.srt", output_format="ass")
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_cv.assert_called_once_with(
            input_file="/tmp/in.srt", output=None, output_format="ass", style=None, resolution=None
        )


class TestForcedAlignTool:
    @patch("omnicaptions.services.align.forced_align")
    def test_align_success(self, mock_align, mcp):
        mock_align.return_value = ToolResult(
            status="success",
            message="Aligned",
            data={"output_path": "/tmp/out.json"},
        )
        tool_fn = _get_tool_fn(mcp, "forced_align")
        result = tool_fn(audio="/tmp/audio.m4a", caption="/tmp/cap.srt")
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_align.assert_called_once_with(
            audio="/tmp/audio.m4a", caption="/tmp/cap.srt", output=None, output_format="json"
        )


# --- Integration: list_tools ---


class TestListTools:
    def test_server_has_eight_tools(self, mcp):
        tools = mcp._tool_manager.list_tools()
        names = {t.name for t in tools}
        expected = {
            "doctor",
            "download",
            "transcribe",
            "check_captions",
            "translate",
            "convert",
            "forced_align",
            "pipeline",
        }
        assert names == expected

    def test_tools_have_descriptions(self, mcp):
        tools = mcp._tool_manager.list_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name} has no description"


class TestFallbackBehavior:
    def test_run_raises_install_hint_without_mcp_dependency(self, mcp):
        with pytest.raises(RuntimeError) as exc_info:
            mcp.run(transport="stdio")
        message = str(exc_info.value)
        assert "mcp" in message.lower()
        assert MCP_INSTALL_HINT in message


# --- Helper ---


def _get_tool_fn(mcp, name: str):
    """Extract the raw Python function for a registered MCP tool."""
    tool = mcp._tool_manager._tools[name]
    return tool.fn
