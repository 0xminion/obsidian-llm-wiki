"""Visualization adapter — re-exports the legacy visualizer module."""

from __future__ import annotations

from pipeline.okf_visualizer import generate_visualization  # noqa: F401

__all__ = ["generate_visualization"]
