"""Tests for CLI convert command with --from/--to options."""

import argparse
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnicaptions.cli import cmd_convert

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_VTT_FILE = TEST_DATA_DIR / "SA1.vtt"


class TestConvertArgparse:
    """Test convert command argument parsing (no external deps)."""

    def get_parser(self):
        """Create parser matching cli.py structure."""
        parser = argparse.ArgumentParser(prog="omnicaptions")
        subparsers = parser.add_subparsers(dest="command", required=True)

        p_convert = subparsers.add_parser("convert", help="Convert caption formats")
        p_convert.add_argument("input", help="Input caption file")
        p_convert.add_argument("output", help="Output caption file")
        p_convert.add_argument("-f", "--from", dest="from", metavar="FMT", help="Input format")
        p_convert.add_argument("-t", "--to", dest="to", metavar="FMT", help="Output format")
        p_convert.add_argument("-v", "--verbose", action="store_true")
        return parser

    def test_basic_convert_args(self):
        """Test basic convert without format options."""
        parser = self.get_parser()
        args = parser.parse_args(["convert", "input.srt", "output.vtt"])

        assert args.command == "convert"
        assert args.input == "input.srt"
        assert args.output == "output.vtt"
        assert getattr(args, "from") is None
        assert args.to is None

    def test_from_option_short(self):
        """Test -f option for input format."""
        parser = self.get_parser()
        args = parser.parse_args(["convert", "input.txt", "output.srt", "-f", "txt"])

        assert getattr(args, "from") == "txt"

    def test_from_option_long(self):
        """Test --from option for input format."""
        parser = self.get_parser()
        args = parser.parse_args(["convert", "input.txt", "output.srt", "--from", "json"])

        assert getattr(args, "from") == "json"

    def test_to_option_short(self):
        """Test -t option for output format."""
        parser = self.get_parser()
        args = parser.parse_args(["convert", "input.srt", "output.txt", "-t", "vtt"])

        assert args.to == "vtt"

    def test_to_option_long(self):
        """Test --to option for output format."""
        parser = self.get_parser()
        args = parser.parse_args(["convert", "input.srt", "output.txt", "--to", "ass"])

        assert args.to == "ass"

    def test_both_options(self):
        """Test both --from and --to together."""
        parser = self.get_parser()
        args = parser.parse_args(
            ["convert", "data.json", "result.ass", "--from", "json", "--to", "ass"]
        )

        assert getattr(args, "from") == "json"
        assert args.to == "ass"

    def test_both_options_short(self):
        """Test both -f and -t together."""
        parser = self.get_parser()
        args = parser.parse_args(["convert", "data.json", "result.ass", "-f", "json", "-t", "ass"])

        assert getattr(args, "from") == "json"
        assert args.to == "ass"

    def test_gemini_format(self):
        """Test gemini format option."""
        parser = self.get_parser()
        args = parser.parse_args(["convert", "transcript.md", "output.srt", "--from", "gemini"])

        assert getattr(args, "from") == "gemini"


class TestConvertLogic:
    """Test cmd_convert function logic using mocks."""

    def _create_cmd_convert(self, mock_caption_cls, mock_gemini_reader):
        """Create cmd_convert function with mocked dependencies."""

        def cmd_convert(args):
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"Error: Input file not found: {input_path}", file=sys.stderr)
                sys.exit(1)

            input_format = getattr(args, "from", None)
            if input_format == "gemini" or (
                input_format is None and input_path.suffix.lower() == ".md"
            ):
                supervisions = mock_gemini_reader.extract_for_alignment(str(input_path))
                cap = mock_caption_cls.from_supervisions(supervisions)
            else:
                cap = mock_caption_cls.read(str(input_path), format=input_format)

            output_format = args.to
            cap.write(args.output, output_format=output_format)

        return cmd_convert

    @pytest.fixture
    def mock_caption(self):
        """Create mock Caption instance."""
        mock_cap = MagicMock()
        mock_cap.write = MagicMock()
        return mock_cap

    @pytest.fixture
    def mock_caption_cls(self, mock_caption):
        """Create mock Caption class."""
        mock_cls = MagicMock()
        mock_cls.read.return_value = mock_caption
        mock_cls.from_supervisions.return_value = mock_caption
        return mock_cls

    @pytest.fixture
    def mock_gemini_reader(self):
        """Create mock GeminiReader."""
        mock_reader = MagicMock()
        mock_reader.extract_for_alignment.return_value = []
        return mock_reader

    @pytest.fixture
    def temp_srt_file(self):
        """Create temporary SRT file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def temp_md_file(self):
        """Create temporary markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Transcript\n[00:00:01] Hello\n")
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    def test_from_format_passed_to_read(
        self, mock_caption_cls, mock_gemini_reader, temp_srt_file, mock_caption
    ):
        """Test --from option is passed to Caption.read()."""
        cmd_convert = self._create_cmd_convert(mock_caption_cls, mock_gemini_reader)

        args = argparse.Namespace(
            input=str(temp_srt_file),
            output="output.vtt",
            verbose=False,
            to=None,
        )
        setattr(args, "from", "txt")

        cmd_convert(args)

        mock_caption_cls.read.assert_called_once_with(str(temp_srt_file), format="txt")

    def test_to_format_passed_to_write(
        self, mock_caption_cls, mock_gemini_reader, temp_srt_file, mock_caption
    ):
        """Test --to option is passed to Caption.write()."""
        cmd_convert = self._create_cmd_convert(mock_caption_cls, mock_gemini_reader)

        args = argparse.Namespace(
            input=str(temp_srt_file),
            output="output.txt",
            verbose=False,
            to="ass",
        )
        setattr(args, "from", None)

        cmd_convert(args)

        mock_caption.write.assert_called_once_with("output.txt", output_format="ass")

    def test_gemini_format_uses_reader(self, mock_caption_cls, mock_gemini_reader, temp_md_file):
        """Test --from gemini uses GeminiReader."""
        cmd_convert = self._create_cmd_convert(mock_caption_cls, mock_gemini_reader)

        args = argparse.Namespace(
            input=str(temp_md_file),
            output="output.srt",
            verbose=False,
            to=None,
        )
        setattr(args, "from", "gemini")

        cmd_convert(args)

        mock_gemini_reader.extract_for_alignment.assert_called_once_with(str(temp_md_file))
        mock_caption_cls.from_supervisions.assert_called_once()
        mock_caption_cls.read.assert_not_called()

    def test_md_extension_auto_detects_gemini(
        self, mock_caption_cls, mock_gemini_reader, temp_md_file
    ):
        """Test .md files auto-detect as gemini format."""
        cmd_convert = self._create_cmd_convert(mock_caption_cls, mock_gemini_reader)

        args = argparse.Namespace(
            input=str(temp_md_file),
            output="output.srt",
            verbose=False,
            to=None,
        )
        setattr(args, "from", None)  # No --from, should auto-detect

        cmd_convert(args)

        mock_gemini_reader.extract_for_alignment.assert_called_once()
        mock_caption_cls.read.assert_not_called()

    def test_none_format_uses_auto_detect(
        self, mock_caption_cls, mock_gemini_reader, temp_srt_file, mock_caption
    ):
        """Test no --from passes None to Caption.read() for auto-detect."""
        cmd_convert = self._create_cmd_convert(mock_caption_cls, mock_gemini_reader)

        args = argparse.Namespace(
            input=str(temp_srt_file),
            output="output.vtt",
            verbose=False,
            to=None,
        )
        setattr(args, "from", None)

        cmd_convert(args)

        mock_caption_cls.read.assert_called_once_with(str(temp_srt_file), format=None)

    def test_file_not_found_exits(self, mock_caption_cls, mock_gemini_reader):
        """Test error handling for missing input file."""
        cmd_convert = self._create_cmd_convert(mock_caption_cls, mock_gemini_reader)

        args = argparse.Namespace(
            input="/nonexistent/file.srt",
            output="output.vtt",
            verbose=False,
            to=None,
        )
        setattr(args, "from", None)

        with pytest.raises(SystemExit) as exc_info:
            cmd_convert(args)

        assert exc_info.value.code == 1

    def test_both_from_and_to(
        self, mock_caption_cls, mock_gemini_reader, temp_srt_file, mock_caption
    ):
        """Test using both --from and --to options."""
        cmd_convert = self._create_cmd_convert(mock_caption_cls, mock_gemini_reader)

        args = argparse.Namespace(
            input=str(temp_srt_file),
            output="output.ass",
            verbose=False,
            to="ass",
        )
        setattr(args, "from", "json")

        cmd_convert(args)

        mock_caption_cls.read.assert_called_once_with(str(temp_srt_file), format="json")
        mock_caption.write.assert_called_once_with("output.ass", output_format="ass")


class TestConvertIntegration:
    """Integration tests using real data files."""

    @pytest.fixture
    def temp_output(self, tmp_path):
        """Create temporary output path."""
        return tmp_path / "output"

    def test_vtt_to_srt(self, temp_output):
        """Test VTT to SRT conversion."""
        output_file = temp_output.with_suffix(".srt")
        args = argparse.Namespace(
            input=str(TEST_VTT_FILE),
            output=str(output_file),
            verbose=False,
            to=None,
        )
        setattr(args, "from", None)
        setattr(args, "style", None)
        setattr(args, "karaoke", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        assert "She had your dark suit" in content
        assert "00:00:00,480 --> 00:00:03,980" in content

    def test_vtt_to_ass(self, temp_output):
        """Test VTT to ASS conversion."""
        output_file = temp_output.with_suffix(".ass")
        args = argparse.Namespace(
            input=str(TEST_VTT_FILE),
            output=str(output_file),
            verbose=False,
            to=None,
        )
        setattr(args, "from", None)
        setattr(args, "style", None)
        setattr(args, "karaoke", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        assert "[Script Info]" in content
        assert "She had your dark suit" in content

    def test_vtt_to_txt(self, temp_output):
        """Test VTT to TXT conversion."""
        output_file = temp_output.with_suffix(".txt")
        args = argparse.Namespace(
            input=str(TEST_VTT_FILE),
            output=str(output_file),
            verbose=False,
            to=None,
        )
        setattr(args, "from", None)
        setattr(args, "style", None)
        setattr(args, "karaoke", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        assert "She had your dark suit" in content

    def test_explicit_from_format(self, temp_output):
        """Test explicit --from vtt format."""
        output_file = temp_output.with_suffix(".srt")
        args = argparse.Namespace(
            input=str(TEST_VTT_FILE),
            output=str(output_file),
            verbose=False,
            to=None,
        )
        setattr(args, "from", "vtt")
        setattr(args, "style", None)
        setattr(args, "karaoke", None)

        cmd_convert(args)

        assert output_file.exists()

    def test_explicit_to_format(self, temp_output):
        """Test explicit --to srt format (ignore extension)."""
        output_file = temp_output.with_suffix(".txt")  # Wrong extension
        args = argparse.Namespace(
            input=str(TEST_VTT_FILE),
            output=str(output_file),
            verbose=False,
            to="srt",  # Force SRT format
        )
        setattr(args, "from", None)
        setattr(args, "style", None)
        setattr(args, "karaoke", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        # Should be SRT format despite .txt extension
        assert "00:00:00,480 --> 00:00:03,980" in content

    def test_both_from_and_to_formats(self, temp_output):
        """Test explicit --from and --to formats."""
        output_file = temp_output.with_suffix(".out")
        args = argparse.Namespace(
            input=str(TEST_VTT_FILE),
            output=str(output_file),
            verbose=False,
            to="ass",
        )
        setattr(args, "from", None)
        setattr(args, "style", None)
        setattr(args, "karaoke", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        assert "[Script Info]" in content


TEST_KARAOKE_JSON = TEST_DATA_DIR / "karaoke_test.json"


class TestKaraokeConvert:
    """Test karaoke conversion features."""

    @pytest.fixture
    def temp_output(self, tmp_path):
        """Create temporary output path."""
        return tmp_path / "output"

    def test_karaoke_ass_sweep(self, temp_output):
        """Test karaoke ASS output with sweep effect."""
        output_file = temp_output.with_suffix(".ass")
        args = argparse.Namespace(
            input=str(TEST_KARAOKE_JSON),
            output=str(output_file),
            verbose=False,
            to="ass",
            karaoke="sweep",
        )
        setattr(args, "from", None)
        setattr(args, "style", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        # Check for karaoke tags
        assert "\\kf" in content
        assert "Karaoke" in content  # Karaoke style
        assert "Hello" in content

    def test_karaoke_ass_instant(self, temp_output):
        """Test karaoke ASS output with instant effect."""
        output_file = temp_output.with_suffix(".ass")
        args = argparse.Namespace(
            input=str(TEST_KARAOKE_JSON),
            output=str(output_file),
            verbose=False,
            to="ass",
            karaoke="instant",
        )
        setattr(args, "from", None)
        setattr(args, "style", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        # Check for instant karaoke tags (not kf)
        assert "{\\k" in content
        assert "\\kf" not in content

    def test_karaoke_ass_outline(self, temp_output):
        """Test karaoke ASS output with outline effect."""
        output_file = temp_output.with_suffix(".ass")
        args = argparse.Namespace(
            input=str(TEST_KARAOKE_JSON),
            output=str(output_file),
            verbose=False,
            to="ass",
            karaoke="outline",
        )
        setattr(args, "from", None)
        setattr(args, "style", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        # Check for outline karaoke tags
        assert "\\ko" in content

    def test_karaoke_lrc(self, temp_output):
        """Test karaoke LRC output with word timestamps."""
        output_file = temp_output.with_suffix(".lrc")
        args = argparse.Namespace(
            input=str(TEST_KARAOKE_JSON),
            output=str(output_file),
            verbose=False,
            to="lrc",
            karaoke="sweep",
        )
        setattr(args, "from", None)
        setattr(args, "style", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        # Check for enhanced LRC word timestamps
        assert "<" in content  # Word timestamp markers
        assert "Hello" in content

    def test_karaoke_default_effect(self, temp_output):
        """Test --karaoke without effect defaults to sweep."""
        output_file = temp_output.with_suffix(".ass")
        args = argparse.Namespace(
            input=str(TEST_KARAOKE_JSON),
            output=str(output_file),
            verbose=False,
            to="ass",
            karaoke="sweep",  # Default when --karaoke is used without value
        )
        setattr(args, "from", None)
        setattr(args, "style", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        assert "\\kf" in content

    def test_no_karaoke_no_tags(self, temp_output):
        """Test conversion without karaoke has no karaoke tags."""
        output_file = temp_output.with_suffix(".ass")
        args = argparse.Namespace(
            input=str(TEST_KARAOKE_JSON),
            output=str(output_file),
            verbose=False,
            to="ass",
            karaoke=None,
        )
        setattr(args, "from", None)
        setattr(args, "style", None)

        cmd_convert(args)

        assert output_file.exists()
        content = output_file.read_text()
        # Should not have karaoke tags
        assert "\\kf" not in content
        assert "\\ko" not in content
        # But should have content
        assert "Hello world" in content
