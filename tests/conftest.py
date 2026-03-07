"""Shared fixtures for plugin evals."""

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).parent.parent


@pytest.fixture
def plugin_root():
    """Path to the plugin root directory."""
    return PLUGIN_ROOT


# ── MCP server data (session-scoped to avoid reloading large JSON files) ────

MCP_SERVER_SRC = PLUGIN_ROOT / "mcp-server" / "src"


@pytest.fixture(scope="session")
def _mcp_path():
    """Ensure MCP server source is on sys.path (once per session)."""
    src = str(MCP_SERVER_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


@pytest.fixture(scope="session")
def mcp_all_api(_mcp_path):
    """ALL_API list loaded once per session."""
    from pymc_docs_server.server import ALL_API
    return ALL_API


@pytest.fixture(scope="session")
def mcp_patterns(_mcp_path):
    """PATTERNS list loaded once per session."""
    from pymc_docs_server.server import PATTERNS
    return PATTERNS


@pytest.fixture(scope="session")
def mcp_error_patterns(_mcp_path):
    """ERROR_PATTERNS list loaded once per session."""
    from pymc_docs_server.server import ERROR_PATTERNS
    return ERROR_PATTERNS
