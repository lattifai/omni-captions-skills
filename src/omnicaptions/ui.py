"""Rich console UI helpers for human-readable output.

All functions are no-ops when as_json=True.
All Rich output goes to stderr to never pollute stdout JSON.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


_console: Console | None = None


def get_console() -> "Console":
    """Get or create the shared Rich console (stderr)."""
    global _console
    if _console is None:
        from rich.console import Console

        _console = Console(stderr=True)
    return _console


@contextmanager
def step_spinner(message: str, as_json: bool = False):
    """Show a spinner during a long operation. No-op when as_json=True."""
    if as_json:
        yield
        return
    console = get_console()
    with console.status(message, spinner="dots"):
        yield


class StepTracker:
    """Track multi-step pipeline progress."""

    def __init__(self, steps: list[str], as_json: bool = False):
        self._steps = steps
        self._as_json = as_json
        self._console = None if as_json else get_console()

    def advance(self, step_name: str, status: str):
        """Update step status. status is 'running', 'success', 'skipped', 'error'."""
        if self._as_json:
            return
        icon = {
            "running": "\u280b",
            "success": "\u2713",
            "skipped": "\u25cb",
            "error": "\u2717",
        }.get(status, "?")
        style = {
            "running": "bold cyan",
            "success": "bold green",
            "skipped": "dim",
            "error": "bold red",
        }.get(status, "")
        self._console.print(
            f"  [{style}]{icon} {step_name}[/{style}]" if style else f"  {icon} {step_name}"
        )


def step_progress(steps: list[str], as_json: bool = False) -> StepTracker:
    """Create a multi-step progress tracker."""
    return StepTracker(steps, as_json=as_json)


def print_table(headers: list[str], rows: list[list[str]], as_json: bool = False):
    """Print a Rich table to stderr."""
    if as_json:
        return
    from rich.table import Table

    console = get_console()
    table = Table(show_header=True, header_style="bold", show_edge=False, pad_edge=False)
    for h in headers:
        table.add_column(h)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def print_success(message: str, as_json: bool = False):
    """Print success with green checkmark to stderr."""
    if as_json:
        return
    get_console().print(f"  [bold green]\u2713[/bold green] {message}")


def print_error(message: str, as_json: bool = False):
    """Print error with red X to stderr."""
    if as_json:
        return
    get_console().print(f"  [bold red]\u2717[/bold red] {message}")
