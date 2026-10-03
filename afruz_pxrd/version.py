"""Authoritative Afruz PXRD release identity.

Every user-visible version and exported software-version field must be derived
from this module.  Schema versions remain local to their respective formats.
"""

from __future__ import annotations


APP_NAME = "Afruz PXRD Analyzer"
APP_VERSION = "23.0.0"
APP_WINDOW_TITLE = f"{APP_NAME} — {APP_VERSION}"
APP_RELEASE = f"{APP_NAME} {APP_VERSION}"


def project_window_title(project_name: str | None = None) -> str:
    """Return the consistent main-window title for a project or empty session."""
    name = str(project_name or "").strip()
    return f"{name} — {APP_WINDOW_TITLE}" if name else APP_WINDOW_TITLE
