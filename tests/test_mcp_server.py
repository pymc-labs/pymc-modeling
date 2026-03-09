"""Evals for the MCP server tool functions — tests search, lookup, and error matching."""

import sys
from pathlib import Path

import pytest

# Add the MCP server source to the path so we can import directly
MCP_SERVER_SRC = Path(__file__).parent.parent / "mcp-server" / "src"
sys.path.insert(0, str(MCP_SERVER_SRC))

from pymc_docs_server.server import (
    ERROR_PATTERNS,
    ALL_API,
    PATTERNS,
    _keyword_match,
    _text_match,
    pymc_api_lookup,
    pymc_error_lookup,
    pymc_example_search,
)


# ── Data integrity ───────────────────────────────────────────────────────────

class TestDataIntegrity:
    def test_api_entries_have_required_fields(self, mcp_all_api):
        for entry in mcp_all_api:
            assert "name" in entry, f"API entry missing 'name': {entry}"
            assert "signature" in entry, f"API entry missing 'signature': {entry.get('name')}"
            assert "description" in entry, f"API entry missing 'description': {entry.get('name')}"

    def test_patterns_have_required_fields(self, mcp_patterns):
        for pattern in mcp_patterns:
            assert "keywords" in pattern, f"Pattern missing 'keywords': {pattern}"
            assert "title" in pattern, f"Pattern missing 'title': {pattern}"
            assert "code" in pattern, f"Pattern missing 'code': {pattern}"
            assert "explanation" in pattern, f"Pattern missing 'explanation': {pattern}"

    def test_error_patterns_have_required_fields(self, mcp_error_patterns):
        for err in mcp_error_patterns:
            assert "pattern" in err, f"Error pattern missing 'pattern'"
            assert "keywords" in err, f"Error pattern missing 'keywords': {err.get('pattern')}"
            assert "title" in err, f"Error pattern missing 'title': {err.get('pattern')}"
            assert "fix" in err, f"Error pattern missing 'fix': {err.get('pattern')}"

    def test_api_has_core_functions(self, mcp_all_api):
        names = {e["name"] for e in mcp_all_api}
        core = ["pm.sample", "pm.Model", "pm.Normal"]
        for fn in core:
            assert fn in names, f"Missing core API entry: {fn}"

    def test_patterns_not_empty(self, mcp_patterns):
        assert len(mcp_patterns) >= 3, f"Expected at least 3 patterns, got {len(mcp_patterns)}"

    def test_error_patterns_not_empty(self, mcp_error_patterns):
        assert len(mcp_error_patterns) >= 5, f"Expected at least 5 error patterns, got {len(mcp_error_patterns)}"

    def test_api_gotchas_are_lists(self, mcp_all_api):
        """gotchas field should be a list, not a string."""
        for entry in mcp_all_api:
            if "gotchas" in entry:
                assert isinstance(entry["gotchas"], list), (
                    f"gotchas should be a list in {entry['name']}, got {type(entry['gotchas'])}"
                )

    def test_api_see_also_are_lists(self, mcp_all_api):
        """see_also field should be a list when present."""
        for entry in mcp_all_api:
            if "see_also" in entry:
                assert isinstance(entry["see_also"], list), (
                    f"see_also should be a list in {entry['name']}, got {type(entry['see_also'])}"
                )

    def test_no_duplicate_api_names(self, mcp_all_api):
        """No duplicate name values across ALL_API."""
        names = [e["name"] for e in mcp_all_api]
        dupes = [n for n in names if names.count(n) > 1]
        assert not dupes, f"Duplicate API names: {set(dupes)}"

    def test_pattern_code_contains_pymc(self, mcp_patterns):
        """Pattern code fields should contain valid-looking Python."""
        for pattern in mcp_patterns:
            code = pattern["code"]
            assert "pm." in code or "az." in code or "import" in code or "pymc" in code.lower(), (
                f"Pattern '{pattern['title']}' code doesn't look like PyMC/ArviZ Python"
            )

    def test_error_pattern_keywords_nonempty(self, mcp_error_patterns):
        """Error pattern keywords lists should be non-empty."""
        for err in mcp_error_patterns:
            assert len(err["keywords"]) > 0, (
                f"Error pattern '{err['pattern']}' has empty keywords list"
            )


# ── Helper functions ─────────────────────────────────────────────────────────

class TestKeywordMatch:
    def test_exact_match(self):
        assert _keyword_match("divergence", ["divergence"]) == 1

    def test_case_insensitive(self):
        assert _keyword_match("DIVERGENCE", ["divergence"]) == 1

    def test_multiple_matches(self):
        assert _keyword_match("divergence and rhat", ["divergence", "rhat", "ess"]) == 2

    def test_no_match(self):
        assert _keyword_match("hello world", ["divergence", "rhat"]) == 0

    def test_partial_match(self):
        assert _keyword_match("divergent transitions", ["divergent"]) == 1


class TestTextMatch:
    def test_word_in_text(self):
        assert _text_match("sample", "Draw samples from the posterior")

    def test_no_match(self):
        assert not _text_match("banana", "Draw samples from the posterior")

    def test_multi_word_query(self):
        assert _text_match("posterior sample", "Draw samples from the posterior")


# ── pymc_api_lookup ──────────────────────────────────────────────────────────

class TestApiLookup:
    def test_exact_lookup(self):
        result = pymc_api_lookup("pm.sample")
        assert "pm.sample" in result
        assert "draws" in result or "posterior" in result.lower()

    def test_short_name_lookup(self):
        result = pymc_api_lookup("Normal")
        assert "Normal" in result

    def test_partial_match(self):
        result = pymc_api_lookup("sample_prior")
        assert "prior" in result.lower()

    def test_not_found(self):
        result = pymc_api_lookup("nonexistent_function_xyz")
        assert "No API entry found" in result or "no" in result.lower()

    def test_gotchas_included(self):
        result = pymc_api_lookup("pm.sample")
        assert "nutpie" in result.lower() or "Gotchas" in result

    def test_see_also_included(self):
        result = pymc_api_lookup("pm.sample")
        assert "See also" in result or "see_also" in result.lower()


# ── ArviZ API lookups ────────────────────────────────────────────────────────

class TestArviZApiLookup:
    def test_az_summary_lookup(self):
        result = pymc_api_lookup("az.summary")
        assert "summary" in result.lower()

    def test_az_loo_lookup(self):
        result = pymc_api_lookup("az.loo")
        assert "loo" in result.lower()

    def test_az_compare_lookup(self):
        result = pymc_api_lookup("az.compare")
        assert "compare" in result.lower()

    def test_all_api_contains_arviz(self, mcp_all_api):
        """ALL_API should contain both PyMC and ArviZ entries."""
        names = {e["name"] for e in mcp_all_api}
        arviz_names = [n for n in names if n.startswith("az.")]
        assert len(arviz_names) >= 3, f"Expected at least 3 ArviZ entries, found {len(arviz_names)}"


# ── pymc_example_search ──────────────────────────────────────────────────────

class TestExampleSearch:
    def test_hierarchical_search(self):
        result = pymc_example_search("hierarchical non-centered")
        assert "hierarchical" in result.lower() or "non-centered" in result.lower()
        assert "```python" in result

    def test_mixture_search(self):
        result = pymc_example_search("mixture model")
        assert "mixture" in result.lower()

    def test_horseshoe_search(self):
        result = pymc_example_search("horseshoe shrinkage")
        assert "horseshoe" in result.lower()

    def test_no_results(self):
        result = pymc_example_search("quantum computing blockchain")
        assert "No patterns found" in result

    def test_broad_search(self):
        result = pymc_example_search("regression")
        # Should find something related
        assert "```python" in result or "No patterns" in result

    def test_results_have_code_blocks(self):
        result = pymc_example_search("hierarchical")
        if "No patterns" not in result:
            assert "```python" in result

    def test_hierarchical_ranks_first(self):
        """The best match for 'hierarchical' should appear in the first result block."""
        result = pymc_example_search("hierarchical")
        if "No patterns" not in result:
            # The first pattern title in the output should contain 'hierarchical'
            first_section = result.split("---")[0] if "---" in result else result
            assert "hierarchical" in first_section.lower(), (
                "Expected 'hierarchical' pattern to rank first"
            )

    def test_mixture_ranks_in_top_results(self):
        """A search for 'mixture model' should rank mixture-related results in top 3."""
        result = pymc_example_search("mixture model")
        if "No patterns" not in result:
            sections = result.split("---")[:3]
            top_text = " ".join(sections).lower()
            assert "mixture" in top_text, (
                "Expected 'mixture' to appear in top 3 results"
            )


# ── pymc_error_lookup ────────────────────────────────────────────────────────

class TestErrorLookup:
    def test_divergence_lookup(self):
        result = pymc_error_lookup("I'm getting divergent transitions")
        assert "divergen" in result.lower()
        assert "non-centered" in result.lower() or "target_accept" in result.lower()

    def test_rhat_lookup(self):
        result = pymc_error_lookup("r_hat values are above 1.05")
        assert "r-hat" in result.lower() or "rhat" in result.lower() or "r_hat" in result.lower()

    def test_shape_mismatch_lookup(self):
        result = pymc_error_lookup("ValueError: shape mismatch in my model")
        assert "shape" in result.lower() or "dimension" in result.lower()

    def test_sampling_error_lookup(self):
        result = pymc_error_lookup("SamplingError: bad initial energy")
        assert "initial" in result.lower() or "sampling" in result.lower()

    def test_nutpie_log_likelihood_lookup(self):
        result = pymc_error_lookup("nutpie log_likelihood missing for LOO")
        assert "nutpie" in result.lower() or "compute_log_likelihood" in result.lower()

    def test_label_switching_lookup(self):
        result = pymc_error_lookup("label switching in mixture model")
        assert "label" in result.lower() or "ordered" in result.lower()

    def test_theano_import_lookup(self):
        result = pymc_error_lookup("import theano ModuleNotFoundError")
        assert "pytensor" in result.lower() or "theano" in result.lower()

    def test_slow_sampling_lookup(self):
        result = pymc_error_lookup("sampling is very slow takes hours")
        assert "nutpie" in result.lower() or "speed" in result.lower() or "slow" in result.lower()

    def test_unknown_error_gets_general_tips(self):
        result = pymc_error_lookup("bananas and coconuts")
        assert "General debugging" in result or "summary" in result.lower()

    @pytest.mark.parametrize("error_pattern", ERROR_PATTERNS, ids=lambda e: e["pattern"])
    def test_each_error_pattern_findable(self, error_pattern):
        """Each defined error pattern should be findable via at least one of its keywords."""
        keyword = error_pattern["keywords"][0]
        result = pymc_error_lookup(keyword)
        assert error_pattern["title"] in result, (
            f"Error pattern '{error_pattern['pattern']}' not found via keyword '{keyword}'"
        )

    def test_divergence_ranks_first(self):
        """Divergence error should be the first match for 'divergent transitions'."""
        result = pymc_error_lookup("I'm getting divergent transitions")
        first_section = result.split("---")[0] if "---" in result else result
        assert "divergen" in first_section.lower(), (
            "Expected divergence fix to rank first"
        )


# ── FastMCP registration ─────────────────────────────────────────────────────

class TestFastMCPRegistration:
    def test_mcp_object_exists(self):
        """The FastMCP mcp object should be importable."""
        from pymc_docs_server.server import mcp
        assert mcp is not None
        assert mcp.name == "pymc-docs"

    def test_mcp_lists_tools(self):
        """FastMCP should register the 3 expected tools."""
        import asyncio
        from pymc_docs_server.server import mcp

        async def _list():
            return await mcp.list_tools()

        tools = asyncio.run(_list())
        tool_names = {t.name for t in tools}
        expected = {"pymc_api_lookup", "pymc_example_search", "pymc_error_lookup"}
        assert expected == tool_names, f"Expected tools {expected}, got {tool_names}"
