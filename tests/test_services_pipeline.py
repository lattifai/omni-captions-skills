"""Tests for the pipeline service."""

from unittest.mock import MagicMock, patch

from omnicaptions.services.base import ErrorCode, ToolResult
from omnicaptions.services.pipeline import PipelineConfig, run_pipeline


def _ok(msg="ok", **data):
    return ToolResult(status="success", message=msg, data=data)


def _err(msg="fail", code=ErrorCode.DOWNLOAD_FAILED):
    return ToolResult(status="error", message=msg, error_code=code)


@patch("omnicaptions.services.pipeline.convert_caption")
@patch("omnicaptions.services.pipeline.translate_caption")
@patch("omnicaptions.services.pipeline.download_media")
def test_pipeline_full_flow(mock_dl, mock_tl, mock_cv):
    """Full pipeline with captions found in download (skip transcribe)."""
    mock_dl.return_value = _ok(
        audio_path="/tmp/v.m4a",
        caption_path="/tmp/v.en.vtt",
        title="Test Video",
        video_id="abc123",
    )
    mock_tl.return_value = _ok(output_path="/tmp/v.en_Gemini_zh.vtt")
    mock_cv.return_value = _ok(output_path="/tmp/v.en_Gemini_zh.srt")

    config = PipelineConfig(url="abc123", skip_align=True)
    result = run_pipeline(config)

    assert result.status == "success"
    assert result.data["output_path"] == "/tmp/v.en_Gemini_zh.srt"
    assert result.data["title"] == "Test Video"

    steps = result.data["steps"]
    assert steps[0]["name"] == "download"
    assert steps[0]["status"] == "success"
    assert steps[1]["name"] == "transcribe"
    assert steps[1]["status"] == "skipped"


@patch("omnicaptions.services.pipeline.convert_caption")
@patch("omnicaptions.services.pipeline.translate_caption")
@patch("omnicaptions.services.pipeline.transcribe_media")
@patch("omnicaptions.services.pipeline.download_media")
def test_pipeline_transcribes_when_no_captions(mock_dl, mock_tr, mock_tl, mock_cv):
    """Should transcribe when download returns no caption_path."""
    mock_dl.return_value = _ok(
        audio_path="/tmp/v.m4a", caption_path=None, title="No Caps", video_id="x"
    )
    mock_tr.return_value = _ok(output_path="/tmp/v_GeminiUnd.md")
    mock_tl.return_value = _ok(output_path="/tmp/v_Gemini_zh.md")
    mock_cv.return_value = _ok(output_path="/tmp/v_Gemini_zh.srt")

    config = PipelineConfig(url="https://youtube.com/watch?v=x", skip_align=True)
    result = run_pipeline(config)

    assert result.status == "success"
    mock_tr.assert_called_once()
    steps = result.data["steps"]
    assert steps[1]["name"] == "transcribe"
    assert steps[1]["status"] == "success"


@patch("omnicaptions.services.pipeline.download_media")
def test_pipeline_error_propagation_download(mock_dl):
    """Download failure should stop pipeline immediately."""
    mock_dl.return_value = _err("network error", ErrorCode.NETWORK_TIMEOUT)

    config = PipelineConfig(url="bad-url")
    result = run_pipeline(config)

    assert result.status == "error"
    assert result.data["failed_step"] == "download"
    assert result.error_code == ErrorCode.NETWORK_TIMEOUT


@patch("omnicaptions.services.pipeline.convert_caption")
@patch("omnicaptions.services.pipeline.translate_caption")
@patch("omnicaptions.services.pipeline.download_media")
def test_pipeline_skip_translate_flag(mock_dl, mock_tl, mock_cv):
    """With skip_translate=True, translation should be skipped."""
    mock_dl.return_value = _ok(
        audio_path="/tmp/v.m4a", caption_path="/tmp/v.en.vtt", title="T", video_id="x"
    )
    mock_cv.return_value = _ok(output_path="/tmp/v.en.srt")

    config = PipelineConfig(url="x", skip_align=True, skip_translate=True)
    result = run_pipeline(config)

    assert result.status == "success"
    mock_tl.assert_not_called()
    steps = {s["name"]: s for s in result.data["steps"]}
    assert steps["translate"]["status"] == "skipped"


@patch("omnicaptions.services.pipeline.convert_caption")
@patch("omnicaptions.services.pipeline.translate_caption")
@patch("omnicaptions.services.pipeline.download_media")
def test_pipeline_on_step_callback(mock_dl, mock_tl, mock_cv):
    """on_step callback should fire for each pipeline step."""
    mock_dl.return_value = _ok(
        audio_path="/tmp/v.m4a", caption_path="/tmp/v.en.vtt", title="T", video_id="x"
    )
    mock_tl.return_value = _ok(output_path="/tmp/t.srt")
    mock_cv.return_value = _ok(output_path="/tmp/f.srt")

    callback = MagicMock()
    config = PipelineConfig(url="x", skip_align=True, on_step=callback)
    run_pipeline(config)

    assert callback.call_count >= 5  # at least one call per step


@patch("omnicaptions.services.pipeline.convert_caption")
@patch("omnicaptions.services.pipeline.translate_caption")
@patch("omnicaptions.services.pipeline.download_media")
def test_pipeline_output_data_structure(mock_dl, mock_tl, mock_cv):
    """Final ToolResult data should contain output_path, title, video_id, steps."""
    mock_dl.return_value = _ok(
        audio_path="/tmp/v.m4a",
        caption_path="/tmp/v.en.vtt",
        title="My Video",
        video_id="vid123",
    )
    mock_tl.return_value = _ok(output_path="/tmp/translated.srt")
    mock_cv.return_value = _ok(output_path="/tmp/final.srt")

    config = PipelineConfig(url="vid123", skip_align=True)
    result = run_pipeline(config)

    assert "output_path" in result.data
    assert "title" in result.data
    assert "video_id" in result.data
    assert "steps" in result.data
    assert isinstance(result.data["steps"], list)


@patch("omnicaptions.services.pipeline.convert_caption")
@patch("omnicaptions.services.pipeline.translate_caption")
@patch("omnicaptions.services.pipeline.forced_align")
@patch("omnicaptions.services.pipeline.download_media")
def test_pipeline_align_success(mock_dl, mock_align, mock_tl, mock_cv):
    """Alignment step should run when lattifai_api_key is provided."""
    mock_dl.return_value = _ok(
        audio_path="/tmp/v.m4a", caption_path="/tmp/v.en.vtt", title="T", video_id="x"
    )
    mock_align.return_value = _ok(output_path="/tmp/v.en_LaiCut.json")
    mock_tl.return_value = _ok(output_path="/tmp/translated.srt")
    mock_cv.return_value = _ok(output_path="/tmp/final.srt")

    config = PipelineConfig(url="x", lattifai_api_key="test-key")
    result = run_pipeline(config)

    assert result.status == "success"
    mock_align.assert_called_once()
    steps = {s["name"]: s for s in result.data["steps"]}
    assert steps["align"]["status"] == "success"
    assert steps["align"]["output_path"] == "/tmp/v.en_LaiCut.json"


@patch("omnicaptions.services.pipeline.convert_caption")
@patch("omnicaptions.services.pipeline.translate_caption")
@patch("omnicaptions.services.pipeline.forced_align")
@patch("omnicaptions.services.pipeline.download_media")
def test_pipeline_align_failure_is_nonfatal(mock_dl, mock_align, mock_tl, mock_cv):
    """Alignment failure should be non-fatal — pipeline continues with unaligned captions."""
    mock_dl.return_value = _ok(
        audio_path="/tmp/v.m4a", caption_path="/tmp/v.en.vtt", title="T", video_id="x"
    )
    mock_align.return_value = _err("alignment timeout", ErrorCode.ALIGNMENT_FAILED)
    mock_tl.return_value = _ok(output_path="/tmp/translated.srt")
    mock_cv.return_value = _ok(output_path="/tmp/final.srt")

    config = PipelineConfig(url="x", lattifai_api_key="test-key")
    result = run_pipeline(config)

    assert result.status == "success"  # Pipeline should still succeed
    steps = {s["name"]: s for s in result.data["steps"]}
    assert steps["align"]["status"] == "skipped"
    assert "alignment failed" in steps["align"]["reason"]
    # translate should use original caption, not aligned
    mock_tl.assert_called_once()
