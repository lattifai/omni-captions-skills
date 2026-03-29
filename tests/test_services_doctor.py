"""Tests for the doctor service."""

import json
import subprocess
import sys
from unittest.mock import patch

from omnicaptions.services.doctor import run_doctor


def test_doctor_returns_tool_result():
    result = run_doctor()
    assert result.status in ("success", "error")
    assert "checks" in result.data


def test_doctor_checks_structure():
    result = run_doctor()
    checks = result.data["checks"]
    check_names = [c["name"] for c in checks]
    assert "ffmpeg" in check_names
    assert "yt-dlp" in check_names
    assert "gemini_api_key" in check_names
    assert "lattifai_api_key" in check_names


def test_doctor_check_fields():
    result = run_doctor()
    for check in result.data["checks"]:
        assert "name" in check
        assert "status" in check
        assert check["status"] in ("ok", "missing")
        assert "help" in check


def test_doctor_missing_gemini_key():
    with patch("omnicaptions.services.doctor.get_gemini_api_key", return_value=None):
        result = run_doctor()
        checks = {c["name"]: c for c in result.data["checks"]}
        assert checks["gemini_api_key"]["status"] == "missing"
        assert "aistudio.google.com" in checks["gemini_api_key"]["help"]


def test_doctor_missing_lattifai_key():
    with patch("omnicaptions.services.doctor.get_lattifai_api_key", return_value=None):
        result = run_doctor()
        checks = {c["name"]: c for c in result.data["checks"]}
        assert checks["lattifai_api_key"]["status"] == "missing"
        assert "lattifai.com" in checks["lattifai_api_key"]["help"]


def test_doctor_cli_json():
    result = subprocess.run(
        [sys.executable, "-m", "omnicaptions", "doctor", "--json"],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    assert "status" in data
    assert "checks" in data["data"]


def test_doctor_cli_human():
    result = subprocess.run(
        [sys.executable, "-m", "omnicaptions", "doctor"],
        capture_output=True,
        text=True,
    )
    # Rich output goes to stderr
    output = result.stdout + result.stderr
    assert "ffmpeg" in output
    assert "yt-dlp" in output
