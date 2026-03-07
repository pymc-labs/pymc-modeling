"""Shared fixtures for plugin evals."""

from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).parent.parent


@pytest.fixture
def plugin_root():
    """Path to the plugin root directory."""
    return PLUGIN_ROOT
