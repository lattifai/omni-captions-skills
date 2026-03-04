"""Tests for CLI --json output mode."""

import json
import subprocess
import sys
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_VTT_FILE = TEST_DATA_DIR / "SA1.vtt"


def test_convert_json_output(tmp_path):
    """Test convert with --json produces valid JSON output."""
    output = tmp_path / "out.srt"
    result = subprocess.run(
        [sys.executable, "-m", "omnicaptions", "convert", str(TEST_VTT_FILE), "-o", str(output), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["status"] == "success"
    assert "output_path" in data["data"]
    assert data["error_code"] is None


def test_convert_json_error():
    """Test convert with --json on nonexistent file produces JSON error."""
    result = subprocess.run(
        [sys.executable, "-m", "omnicaptions", "convert", "/nonexistent.srt", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["status"] == "error"
    assert data["error_code"] is not None


def test_convert_human_output(tmp_path):
    """Without --json, should still work as before (silent on success)."""
    output = tmp_path / "out.srt"
    result = subprocess.run(
        [sys.executable, "-m", "omnicaptions", "convert", str(TEST_VTT_FILE), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
