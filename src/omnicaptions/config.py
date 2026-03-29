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


def set_lattifai_api_key(key: str) -> None:
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


# ==================== Config Management ====================

# Key name mapping: user-friendly -> internal
KEY_MAP = {
    "gemini_api_key": "GEMINI_API_KEY",
    "lattifai_api_key": "LATTIFAI_API_KEY",
}


def mask_key(key: str, visible_prefix: int = 4, visible_suffix: int = 4) -> str:
    """Mask an API key for display: 'AIza...7x9Q'"""
    if not key:
        return ""
    if len(key) <= visible_prefix + visible_suffix + 3:
        return "****"
    return f"{key[:visible_prefix]}...{key[-visible_suffix:]}"


def _resolve_key(env_name: str) -> tuple[str, str]:
    """Resolve a config key value and its source.

    Checks in priority order:
    1. Environment variable (source = "environment")
    2. .env file via python-dotenv (source = "dotenv")
    3. Config file ~/.config/omnicaptions/config.json (source = "config")
    4. Not found (source = "not set")

    Returns:
        (value, source) tuple
    """
    # 1. Check environment variable before dotenv
    pre_dotenv = os.environ.get(env_name)
    if pre_dotenv:
        return pre_dotenv, "environment"

    # 2. Load .env (if available) and check again
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass

    post_dotenv = os.environ.get(env_name)
    if post_dotenv:
        return post_dotenv, "dotenv"

    # 3. User config file
    config = load_config()
    val = config.get(env_name)
    if val:
        return val, "config"

    # 4. Not found
    return "", "not set"


def list_config() -> dict[str, dict]:
    """Return config listing with source info.

    Returns dict like:
    {
        "gemini_api_key": {"value": "AIza...7x9Q", "source": "config", "set": True},
        "lattifai_api_key": {"value": "", "source": "not set", "set": False},
    }
    """
    result = {}
    for friendly_name, env_name in KEY_MAP.items():
        value, source = _resolve_key(env_name)
        result[friendly_name] = {
            "value": mask_key(value) if value else "",
            "source": source,
            "set": bool(value),
        }
    return result
