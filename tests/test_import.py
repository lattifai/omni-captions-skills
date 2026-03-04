"""Test imports."""


def test_services_layer():
    """Test services layer imports."""
    from omnicaptions import ErrorCode, ToolResult

    assert ToolResult is not None
    assert ErrorCode is not None


def test_config_functions():
    """Test config functions import."""
    from omnicaptions import (
        clear_gemini_api_key,
        get_config_path,
        get_gemini_api_key,
        get_setup_instructions,
        set_gemini_api_key,
    )

    assert callable(get_gemini_api_key)
    assert callable(set_gemini_api_key)
    assert callable(clear_gemini_api_key)
    assert callable(get_config_path)
    assert callable(get_setup_instructions)


def test_lattifai_reexports():
    """Test lattifai-captions re-exports."""
    from omnicaptions import Caption, GeminiReader, GeminiSegment, GeminiWriter

    assert Caption is not None
    assert GeminiReader is not None
    assert GeminiWriter is not None
    assert GeminiSegment is not None


def test_download_result():
    """Test DownloadResult import."""
    from omnicaptions import DownloadResult

    r = DownloadResult()
    assert r.video_id == ""


def test_cli_module():
    """Test CLI module imports."""
    from omnicaptions.cli import main

    assert callable(main)
