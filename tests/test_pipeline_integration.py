"""Pipeline integration tests — verifies service chaining works correctly.

These tests mock external APIs but verify data flows through
download → transcribe → convert correctly.
"""

from pathlib import Path
from unittest.mock import patch

from omnicaptions.services.base import ErrorCode
from omnicaptions.services.convert import convert_caption
from omnicaptions.services.download import download_media
from omnicaptions.services.transcribe import transcribe_media


class TestDownloadToTranscribe:
    """Test download → transcribe pipeline."""

    @patch("omnicaptions.services.transcribe._run_transcription")
    @patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key")
    def test_transcribe_with_local_file(self, mock_key, mock_run, tmp_path):
        """Transcribe should work with a local file path (simulating post-download)."""
        # Simulate downloaded audio file
        audio_file = tmp_path / "e882eXLtwkI.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_run.return_value = "[00:00:00] Hello world\n[00:00:05] Goodbye"
        output = str(tmp_path / "result.md")

        result = transcribe_media(str(audio_file), output=output)

        assert result.status == "success"
        assert result.data["output_path"] == output
        assert Path(output).exists()
        assert "[00:00:00]" in Path(output).read_text()


class TestTranscribeToConvert:
    """Test transcribe → convert pipeline."""

    def test_convert_gemini_md_to_srt(self, tmp_path):
        """Convert Gemini markdown transcription output to SRT format."""
        # Simulate Gemini transcription output (markdown with timestamps)
        gemini_md = tmp_path / "transcription.md"
        gemini_md.write_text(
            "[00:00:01] Hello, welcome to the show.\n"
            "[00:00:05] Today we discuss AI.\n"
            "[00:00:10] Let's get started.\n"
        )

        output_srt = str(tmp_path / "output.srt")
        result = convert_caption(
            input_file=str(gemini_md),
            output=output_srt,
            input_format="gemini",
            output_format="srt",
        )

        assert result.status == "success"
        assert Path(output_srt).exists()
        content = Path(output_srt).read_text()
        assert "00:00:01" in content
        assert "Hello" in content

    def test_convert_srt_to_vtt(self, tmp_path):
        """Convert SRT to VTT format."""
        srt_file = tmp_path / "captions.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nGoodbye\n"
        )

        output_vtt = str(tmp_path / "output.vtt")
        result = convert_caption(
            input_file=str(srt_file),
            output=output_vtt,
            output_format="vtt",
        )

        assert result.status == "success"
        assert Path(output_vtt).exists()
        content = Path(output_vtt).read_text()
        assert "WEBVTT" in content
        assert "Hello world" in content

    def test_convert_srt_to_ass(self, tmp_path):
        """Convert SRT to ASS format."""
        srt_file = tmp_path / "captions.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nGoodbye\n"
        )

        output_ass = str(tmp_path / "output.ass")
        result = convert_caption(
            input_file=str(srt_file),
            output=output_ass,
            output_format="ass",
        )

        assert result.status == "success"
        assert Path(output_ass).exists()
        content = Path(output_ass).read_text()
        assert "[Script Info]" in content
        assert "Hello world" in content


class TestFullPipeline:
    """Test the full download → transcribe → convert pipeline with mocks."""

    @patch("omnicaptions.services.transcribe._run_transcription")
    @patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value="fake-key")
    def test_full_pipeline_mock(self, mock_key, mock_run, tmp_path):
        """Full pipeline: local file → transcribe → convert to SRT."""
        # Step 1: Simulate downloaded audio
        audio_file = tmp_path / "video123.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        # Step 2: Transcribe
        mock_run.return_value = (
            "[00:00:01] Welcome to our channel.\n"
            "[00:00:05] Today we talk about Python.\n"
            "[00:00:10] Let's dive in.\n"
        )
        transcript_path = str(tmp_path / "transcript.md")
        transcribe_result = transcribe_media(str(audio_file), output=transcript_path)
        assert transcribe_result.status == "success"
        assert Path(transcript_path).exists()

        # Step 3: Convert transcription to SRT
        srt_path = str(tmp_path / "final.srt")
        convert_result = convert_caption(
            input_file=transcript_path,
            output=srt_path,
            input_format="gemini",
            output_format="srt",
        )
        assert convert_result.status == "success"
        assert Path(srt_path).exists()

        srt_content = Path(srt_path).read_text()
        assert "Welcome" in srt_content
        assert "-->" in srt_content

    def test_convert_chain_srt_to_vtt_to_ass(self, tmp_path):
        """Chain conversions: SRT → VTT → ASS."""
        # Create initial SRT
        srt_file = tmp_path / "input.srt"
        srt_file.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nFirst line\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nSecond line\n"
        )

        # SRT → VTT
        vtt_file = str(tmp_path / "mid.vtt")
        r1 = convert_caption(str(srt_file), output=vtt_file, output_format="vtt")
        assert r1.status == "success"

        # VTT → ASS
        ass_file = str(tmp_path / "final.ass")
        r2 = convert_caption(vtt_file, output=ass_file, output_format="ass")
        assert r2.status == "success"

        ass_content = Path(ass_file).read_text()
        assert "[Script Info]" in ass_content
        assert "First line" in ass_content
        assert "Second line" in ass_content


class TestErrorPropagation:
    """Test that errors propagate correctly through the pipeline."""

    def test_convert_invalid_input_file(self):
        """Convert with non-existent file should fail gracefully."""
        result = convert_caption(
            input_file="/nonexistent/file.srt",
            output="/tmp/out.vtt",
        )
        assert result.status == "error"

    @patch("omnicaptions.services.transcribe.get_gemini_api_key", return_value=None)
    def test_transcribe_missing_key_stops_pipeline(self, mock_key):
        """Missing API key in transcribe should stop the pipeline."""
        result = transcribe_media("some_file.mp3")
        assert result.status == "error"
        assert result.error_code == ErrorCode.MISSING_API_KEY
        # Pipeline should check this before proceeding to convert

    def test_download_empty_url(self):
        """Download with empty URL should fail."""
        result = download_media("")
        assert result.status == "error"
        assert result.error_code == ErrorCode.INVALID_URL
