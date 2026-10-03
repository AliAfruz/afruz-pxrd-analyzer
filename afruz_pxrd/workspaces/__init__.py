"""Modular workspace contracts, adapters, registry, and GUI mixins."""

from .base import WorkspaceAdapter, WorkspaceValidation
from .registry import WorkspaceRegistry, WorkspaceWidgetSpec

__all__ = [
    "WorkspaceAdapter",
    "WorkspaceRegistry",
    "WorkspaceValidation",
    "WorkspaceWidgetSpec",
]
