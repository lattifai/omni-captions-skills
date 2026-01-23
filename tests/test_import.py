"""Test imports."""


def test_main_class():
    """Test GeminiCaption import."""
    from omnicaptions import GeminiCaption, GeminiCaptionConfig

    assert GeminiCaption is not None
    assert GeminiCaptionConfig is not None


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


def test_gemini_caption_instantiation():
    """Test GeminiCaption can be instantiated."""
    from omnicaptions import GeminiCaption

    gc = GeminiCaption()
    assert gc is not None
    assert hasattr(gc, "transcribe")
    assert hasattr(gc, "translate")
    assert hasattr(gc, "write")


def test_cli_module():
    """Test CLI module imports."""
    from omnicaptions.cli import main

    assert callable(main)
