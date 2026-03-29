"""Tests for Rich UI helpers."""

from omnicaptions.ui import (
    StepTracker,
    get_console,
    print_success,
    print_table,
    step_progress,
    step_spinner,
)


def test_spinner_noop_json():
    """step_spinner yields without error when as_json=True."""
    with step_spinner("Working...", as_json=True):
        pass  # should not raise or produce output


def test_step_tracker_noop_json():
    """StepTracker methods work silently when as_json=True."""
    tracker = step_progress(["a", "b", "c"], as_json=True)
    assert isinstance(tracker, StepTracker)
    tracker.advance("a", "running")
    tracker.advance("b", "success")
    tracker.advance("c", "error")


def test_print_success_noop_json(capsys):
    """print_success produces no stdout when as_json=True."""
    print_success("done", as_json=True)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_table_noop_json(capsys):
    """print_table produces no stdout when as_json=True."""
    print_table(["A", "B"], [["1", "2"]], as_json=True)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_console_singleton():
    """get_console() returns the same instance on repeated calls."""
    c1 = get_console()
    c2 = get_console()
    assert c1 is c2
