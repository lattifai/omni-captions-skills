"""Configuration management for gemini-caption-skills.

Handles API key storage and retrieval with the following priority:
1. Environment variable (GEMINI_API_KEY)
2. Project-level .env file
3. User config file (~/.config/gemini-caption/config.json)
"""

import json
import os
from pathlib import Path
from typing import Optional

# Config file location
CONFIG_DIR = Path.home() / ".config" / "omnicaptions"
CONFIG_FILE = CONFIG_DIR / "config.json"

# API Key setup URL
API_KEY_URL = "https://aistudio.google.com/apikey"


def get_config_path() -> Path:
    """Get the config file path."""
    return CONFIG_FILE


def load_config() -> dict:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(config: dict) -> None:
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def get_gemini_api_key() -> Optional[str]:
    """
    Get Gemini API key with priority:
    1. Environment variable GEMINI_API_KEY
    2. Project .env file (if python-dotenv available)
    3. User config file ~/.config/gemini-caption/config.json
    """
    # 1. Environment variable (highest priority)
    if api_key := os.getenv("GEMINI_API_KEY"):
        return api_key

    # 2. Try loading from .env file
    try:
        from dotenv import load_dotenv

        load_dotenv()
        if api_key := os.getenv("GEMINI_API_KEY"):
            return api_key
    except ImportError:
        pass

    # 3. User config file
    config = load_config()
    return config.get("GEMINI_API_KEY")


def set_gemini_api_key(api_key: str) -> None:
    """
    Save Gemini API key to user config file.

    Args:
        api_key: The Gemini API key to save
    """
    config = load_config()
    config["GEMINI_API_KEY"] = api_key
    save_config(config)


def clear_gemini_api_key() -> None:
    """Remove Gemini API key from user config file."""
    config = load_config()
    config.pop("GEMINI_API_KEY", None)
    save_config(config)


def get_setup_instructions() -> str:
    """Get instructions for setting up API key."""
    return f"Get key from {API_KEY_URL} then: export GEMINI_API_KEY=<paste-here>"


# ==================== LattifAI API Key ====================

LATTIFAI_API_KEY_URL = "https://lattifai.com/dashboard"


def get_lattifai_api_key() -> Optional[str]:
    """Get LattifAI API key from environment or config file."""
    # 1. Check environment variable
    if api_key := os.getenv("LATTIFAI_API_KEY"):
        return api_key

    # 2. Try loading from .env file
    try:
        from dotenv import load_dotenv

        load_dotenv()
        if api_key := os.getenv("LATTIFAI_API_KEY"):
            return api_key
    except ImportError:
        pass

    # 3. User config file
    config = load_config()
    return config.get("LATTIFAI_API_KEY")


def save_lattifai_api_key(key: str) -> None:
    """Save LattifAI API key to config file."""
    config = load_config()
    config["LATTIFAI_API_KEY"] = key
    save_config(config)


def get_lattifai_setup_instructions() -> str:
    """Return setup instructions for LattifAI API key."""
    return f"""
LattifAI API key not found.

Get your API key from: {LATTIFAI_API_KEY_URL}

Then either:
1. Set environment variable: export LATTIFAI_API_KEY=your-key
2. Pass directly: omnicaptions LaiCut --api-key your-key
"""
