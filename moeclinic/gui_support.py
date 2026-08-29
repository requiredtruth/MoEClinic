"""Dependency-free command construction for the desktop control panel."""

from __future__ import annotations

import shlex


def demo_arguments(raw_arguments: str) -> tuple[str, ...]:
    """Return operator arguments or the deterministic bundled demo command."""
    parsed = tuple(shlex.split(raw_arguments))
    if parsed:
        return parsed
    return ("examples/healthy.jsonl", "--fail-on", "never")
