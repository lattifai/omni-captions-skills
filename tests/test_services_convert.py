"""Tests for the convert service layer."""

from pathlib import Path

from omnicaptions.services.base import ErrorCode, ToolResult
from omnicaptions.services.convert import (
    calculate_fontsize,
    convert_caption,
    hex_to_ass_color,
    parse_resolution,
)

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_VTT_FILE = TEST_DATA_DIR / "SA1.vtt"


class TestHelpers:
    """Unit tests for pure helper functions."""

    def test_calculate_fontsize_1080(self):
        assert calculate_fontsize(1080) == 48

    def test_calculate_fontsize_720(self):
        assert calculate_fontsize(720) == 32

    def test_calculate_fontsize_2160(self):
        # 4K: 2160 * 48/1080 = 96
        assert calculate_fontsize(2160) == 96

    def test_calculate_fontsize_clamp_low(self):
        # Very small height clamps to 24
        assert calculate_fontsize(100) == 24

    def test_calculate_fontsize_clamp_high(self):
        # Very large height clamps to 144
        assert calculate_fontsize(10000) == 144

    def test_parse_resolution_preset(self):
        assert parse_resolution("1080p") == (1920, 1080)

    def test_parse_resolution_preset_4k(self):
        assert parse_resolution("4k") == (3840, 2160)

    def test_parse_resolution_wxh(self):
        assert parse_resolution("2560x1440") == (2560, 1440)

    def test_parse_resolution_case_insensitive(self):
        assert parse_resolution("4K") == (3840, 2160)

    def test_parse_resolution_invalid(self):
        assert parse_resolution("not_a_resolution") is None

    def test_hex_to_ass_color_white(self):
        assert hex_to_ass_color("#FFFFFF") == "&H00FFFFFF"

    def test_hex_to_ass_color_red(self):
        # #FF0000 -> BGR: 0000FF
        assert hex_to_ass_color("#FF0000") == "&H000000FF"

    def test_hex_to_ass_color_yellow(self):
        # #FFFF00 -> BGR: 00FFFF
        assert hex_to_ass_color("#FFFF00") == "&H0000FFFF"

    def test_hex_to_ass_color_no_hash(self):
        assert hex_to_ass_color("FF6600") == "&H000066FF"


class TestConvertCaption:
    """Tests for the convert_caption public API."""

    def test_convert_file_not_found(self):
        result = convert_caption("/nonexistent/file.srt")
        assert result.status == "error"
        assert result.error_code == ErrorCode.INVALID_INPUT
        assert "not found" in result.message.lower()

    def test_convert_vtt_to_srt(self, tmp_path):
        output_file = tmp_path / "SA1.srt"
        result = convert_caption(
            str(TEST_VTT_FILE),
            output=str(output_file),
        )
        assert result.status == "success"
        assert output_file.exists()
        content = output_file.read_text()
        assert "She had your dark suit" in content
        assert "00:00:00,480 --> 00:00:03,980" in content

    def test_convert_returns_tool_result(self, tmp_path):
        output_file = tmp_path / "out.srt"
        result = convert_caption(
            str(TEST_VTT_FILE),
            output=str(output_file),
        )
        assert isinstance(result, ToolResult)
        assert result.status == "success"
        assert result.error_code is None
        assert result.data["output_path"] == str(output_file)
        assert "output_format" in result.data
        assert "input_format" in result.data

    def test_convert_auto_output_path(self, tmp_path, monkeypatch):
        """When output is None, output is generated next to input."""
        # Copy VTT to tmp_path so the auto-output lands in tmp_path
        vtt_copy = tmp_path / "test.vtt"
        vtt_copy.write_text(TEST_VTT_FILE.read_text())

        result = convert_caption(
            str(vtt_copy),
            output_format="srt",
        )
        assert result.status == "success"
        expected = tmp_path / "test.srt"
        assert result.data["output_path"] == str(expected)
        assert expected.exists()

    def test_convert_with_style_preset(self, tmp_path):
        output_file = tmp_path / "styled.ass"
        result = convert_caption(
            str(TEST_VTT_FILE),
            output=str(output_file),
            output_format="ass",
            style="yellow",
        )
        assert result.status == "success"
        assert output_file.exists()
        content = output_file.read_text()
        assert "[Script Info]" in content
        # Yellow primary color in BGR: 00FFFF
        assert "&H0000FFFF" in content

    def test_convert_with_explicit_output_format(self, tmp_path):
        output_file = tmp_path / "out.txt"
        result = convert_caption(
            str(TEST_VTT_FILE),
            output=str(output_file),
            output_format="ass",
        )
        assert result.status == "success"
        content = output_file.read_text()
        assert "[Script Info]" in content

    def test_convert_output_dir(self, tmp_path):
        """When output is a directory, file is placed inside it."""
        result = convert_caption(
            str(TEST_VTT_FILE),
            output=str(tmp_path),
            output_format="srt",
        )
        assert result.status == "success"
        expected = tmp_path / "SA1.srt"
        assert expected.exists()
