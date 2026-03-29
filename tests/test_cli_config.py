"""Tests for the config subcommand."""

import json
import subprocess
import sys

from omnicaptions.config import list_config, mask_key

# ---- mask_key tests ----


def test_mask_key_long():
    """Key longer than 12 chars shows prefix...suffix."""
    key = "AIzaSyD1234567890abcdef7x9Q"
    result = mask_key(key)
    assert result == "AIza...7x9Q"
    assert "1234567890" not in result


def test_mask_key_short():
    """Short key (<=11 chars) shows ****."""
    assert mask_key("abc12345") == "****"
    assert mask_key("12345678901") == "****"


def test_mask_key_empty():
    """Empty string returns empty string."""
    assert mask_key("") == ""
    assert mask_key(None) == ""


# ---- list_config tests ----


def test_list_config_no_keys(monkeypatch, tmp_path):
    """Returns all keys with set=False when nothing configured."""
    import omnicaptions.config as cfg

    # Clear env vars
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LATTIFAI_API_KEY", raising=False)

    # Point config file to empty temp location
    empty_config = tmp_path / "config.json"
    monkeypatch.setattr(cfg, "CONFIG_FILE", empty_config)

    result = list_config()
    assert "gemini_api_key" in result
    assert "lattifai_api_key" in result
    for info in result.values():
        assert info["set"] is False
        assert info["source"] == "not set"
        assert info["value"] == ""


# ---- config set + get integration ----


def test_config_set_and_get(monkeypatch, tmp_path):
    """Set a key via save_config, then verify list_config sees it."""
    import omnicaptions.config as cfg

    # Clear env vars so config file is the only source
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LATTIFAI_API_KEY", raising=False)

    # Point config file to temp location
    config_dir = tmp_path / "omnicaptions"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    monkeypatch.setattr(cfg, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg, "CONFIG_FILE", config_file)

    # Set a key
    cfg.set_gemini_api_key("AIzaSyD_TEST_KEY_1234567890")

    # Verify via list_config
    result = list_config()
    info = result["gemini_api_key"]
    assert info["set"] is True
    assert info["source"] == "config"
    assert info["value"] == "AIza...7890"


# ---- CLI subprocess tests ----


def test_config_show_json(monkeypatch):
    """Run CLI with --json, verify JSON output structure."""
    result = subprocess.run(
        [sys.executable, "-m", "omnicaptions", "config", "show", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["status"] == "success"
    assert "gemini_api_key" in data["data"]
    assert "lattifai_api_key" in data["data"]
    for key_info in data["data"].values():
        assert "value" in key_info
        assert "source" in key_info
        assert "set" in key_info
