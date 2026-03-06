"""PyMC documentation search MCP server.

Provides three tools:
- pymc_api_lookup: Look up PyMC/ArviZ function signatures and gotchas
- pymc_example_search: Search for code examples matching a query
- pymc_error_lookup: Look up common PyMC errors and their fixes
"""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pymc-docs")

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Load data files at module import time
with open(DATA_DIR / "pymc_api.json") as f:
    PYMC_API = json.load(f)

with open(DATA_DIR / "arviz_api.json") as f:
    ARVIZ_API = json.load(f)

with open(DATA_DIR / "patterns.json") as f:
    PATTERNS = json.load(f)

# Combine API entries for unified search
ALL_API = PYMC_API + ARVIZ_API

# Common error patterns and fixes
ERROR_PATTERNS = [
    {
        "pattern": "divergence",
        "keywords": ["divergence", "divergent", "diverging"],
        "title": "Divergent transitions after tuning",
        "fix": (
            "1. Try non-centered parameterization for hierarchical models\n"
            "2. Increase target_accept: pm.sample(target_accept=0.95)\n"
            "3. Increase tune steps: pm.sample(tune=2000)\n"
            "4. Reparameterize: use offset + scale instead of direct sampling\n"
            "5. Check az.plot_pair(idata, divergences=True) to identify problematic parameters"
        ),
    },
    {
        "pattern": "bad_energy",
        "keywords": ["energy", "BFMI", "low BFMI", "bad energy"],
        "title": "Low Bayesian Fraction of Missing Information (BFMI)",
        "fix": (
            "1. Check az.plot_energy(idata) — gap between distributions indicates problems\n"
            "2. Reparameterize the model (non-centered parameterization)\n"
            "3. Use more informative priors to constrain the posterior"
        ),
    },
    {
        "pattern": "rhat",
        "keywords": ["r_hat", "rhat", "convergence", "not converged", "1.05", "1.1"],
        "title": "R-hat values above 1.01",
        "fix": (
            "1. Run more chains and draws: pm.sample(draws=2000, tune=2000, chains=4)\n"
            "2. Check for multimodality in the posterior\n"
            "3. Use more informative priors\n"
            "4. Check az.plot_trace(idata) for chain mixing issues\n"
            "5. Try different initialization: pm.sample(initvals=...)"
        ),
    },
    {
        "pattern": "shape_mismatch",
        "keywords": ["shape", "dimension", "mismatch", "broadcast", "ValueError", "shape mismatch"],
        "title": "Shape/dimension mismatch errors",
        "fix": (
            "1. Use coords and dims consistently\n"
            "2. Check that observed data shape matches the model's expected shape\n"
            "3. For matrix operations, verify dimensions with pm.math.dot\n"
            "4. Use pm.model_to_graphviz(model) to visualize the model structure"
        ),
    },
    {
        "pattern": "sampling_error",
        "keywords": ["SamplingError", "sampling error", "bad initial energy", "initial point"],
        "title": "SamplingError / bad initial energy",
        "fix": (
            "1. Check for invalid prior values (e.g., negative scale parameters)\n"
            "2. Set explicit initvals: pm.sample(initvals={'param': value})\n"
            "3. Verify observed data doesn't contain NaN or inf\n"
            "4. Scale your data (standardize predictors)\n"
            "5. Use pm.find_MAP() first to find a good starting point"
        ),
    },
    {
        "pattern": "nutpie_log_likelihood",
        "keywords": ["nutpie", "log_likelihood", "loo", "waic", "compute_log_likelihood"],
        "title": "Missing log_likelihood with nutpie sampler",
        "fix": (
            "nutpie doesn't store log_likelihood automatically.\n"
            "After sampling, call:\n"
            "  pm.compute_log_likelihood(idata)\n"
            "Then you can use az.loo(idata) or az.compare()."
        ),
    },
    {
        "pattern": "label_switching",
        "keywords": ["label switching", "label swap", "mixture", "identifiability", "multimodal"],
        "title": "Label switching in mixture models",
        "fix": (
            "1. Apply ordered transform to component means:\n"
            "   mu = pm.Normal('mu', 0, 10, dims='component',\n"
            "                  transform=pm.distributions.transforms.ordered)\n"
            "2. Use ordered(Dirichlet) for mixture weights\n"
            "3. Set informative initvals for component locations\n"
            "4. Consider using pm.Marginalized for discrete latent variables"
        ),
    },
    {
        "pattern": "memory_error",
        "keywords": ["memory", "MemoryError", "OOM", "out of memory", "killed"],
        "title": "Out of memory during sampling",
        "fix": (
            "1. Reduce number of draws or chains\n"
            "2. Use nutpie which is more memory-efficient: nuts_sampler='nutpie'\n"
            "3. For GPs, use HSGP approximation instead of full GP\n"
            "4. For large datasets, subsample or use minibatch approaches\n"
            "5. Avoid storing unnecessary groups: idata.posterior only"
        ),
    },
    {
        "pattern": "dims_conflict",
        "keywords": ["dims", "coords", "dimension", "conflict", "cutpoints", "already exists"],
        "title": "Dimension name conflicts",
        "fix": (
            "1. Don't use the same name for both a variable and a dimension\n"
            "   BAD:  cutpoints = pm.Normal('cutpoints', dims='cutpoints')\n"
            "   GOOD: cutpoints = pm.Normal('kappa', dims='cutpoint_dim')\n"
            "2. Define all coords before the pm.Model context\n"
            "3. Check that dim names don't clash with PyMC internals"
        ),
    },
    {
        "pattern": "theano_pytensor",
        "keywords": ["theano", "aesara", "pytensor", "import error", "module not found"],
        "title": "Theano/Aesara/PyTensor import errors",
        "fix": (
            "PyMC 5+ uses PyTensor (not Theano or Aesara).\n"
            "Replace:\n"
            "  import theano.tensor as tt  ->  import pytensor.tensor as pt\n"
            "  import aesara.tensor as at  ->  import pytensor.tensor as pt\n"
            "Math operations: pt.dot, pt.exp, pt.log, pt.switch, etc."
        ),
    },
    {
        "pattern": "arviz_datatree",
        "keywords": ["DataTree", "datatree", "ArviZ", "1.0", "groups", "xarray"],
        "title": "ArviZ 1.0 DataTree changes",
        "fix": (
            "ArviZ 1.0 uses xarray DataTree instead of InferenceData.\n"
            "Most functions work the same, but access patterns change:\n"
            "  idata.posterior  (still works)\n"
            "  idata.groups()   (lists available groups)\n"
            "New LOO API: az.loo_expectations, az.loo_metrics, az.loo_r2"
        ),
    },
    {
        "pattern": "slow_sampling",
        "keywords": ["slow", "takes long", "performance", "speed", "hours"],
        "title": "Slow MCMC sampling",
        "fix": (
            "1. Use nutpie: pm.sample(nuts_sampler='nutpie') — 2-5x faster\n"
            "2. Reduce data size if possible (subsample for prototyping)\n"
            "3. Use HSGP instead of full GP for n > 500\n"
            "4. Non-centered parameterization reduces geometry complexity\n"
            "5. Standardize predictors to improve sampler efficiency\n"
            "6. Consider Laplace approximation for quick exploration"
        ),
    },
]


def _keyword_match(query: str, keywords: list[str]) -> int:
    """Count how many keywords match the query (case-insensitive)."""
    query_lower = query.lower()
    return sum(1 for kw in keywords if kw.lower() in query_lower)


def _text_match(query: str, text: str) -> bool:
    """Check if any word in query appears in text (case-insensitive)."""
    query_words = query.lower().split()
    text_lower = text.lower()
    return any(word in text_lower for word in query_words)


@mcp.tool()
def pymc_api_lookup(function_name: str) -> str:
    """Look up PyMC/ArviZ function signatures, descriptions, and gotchas.

    Args:
        function_name: Function name to look up (e.g., 'pm.sample', 'az.loo', 'Normal')
    """
    query = function_name.lower().strip()
    matches = []

    for entry in ALL_API:
        name_lower = entry["name"].lower()
        # Exact match
        if query == name_lower or query == name_lower.split(".")[-1]:
            matches.insert(0, entry)
        # Partial match
        elif query in name_lower or name_lower.split(".")[-1] in query:
            matches.append(entry)

    if not matches:
        # Fuzzy: search in descriptions
        for entry in ALL_API:
            if _text_match(function_name, entry["description"]):
                matches.append(entry)

    if not matches:
        return f"No API entry found for '{function_name}'. Try a different name or search pymc_example_search for patterns."

    result_parts = []
    for entry in matches[:5]:
        parts = [
            f"## {entry['name']}",
            f"```python\n{entry['signature']}\n```",
            f"{entry['description']}",
        ]
        if entry.get("gotchas"):
            parts.append("\n**Gotchas:**")
            for g in entry["gotchas"]:
                parts.append(f"- {g}")
        if entry.get("see_also"):
            parts.append(f"\n**See also:** {', '.join(entry['see_also'])}")
        result_parts.append("\n".join(parts))

    return "\n\n---\n\n".join(result_parts)


@mcp.tool()
def pymc_example_search(query: str) -> str:
    """Search for PyMC code examples and patterns matching a query.

    Args:
        query: Search query (e.g., 'hierarchical non-centered', 'mixture model', 'LOO comparison')
    """
    scored = []
    for pattern in PATTERNS:
        score = _keyword_match(query, pattern["keywords"])
        if score == 0:
            # Check title and explanation
            if _text_match(query, pattern["title"]) or _text_match(query, pattern["explanation"]):
                score = 1
        if score > 0:
            scored.append((score, pattern))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return f"No patterns found for '{query}'. Try broader keywords like 'hierarchical', 'mixture', 'GP', 'time series'."

    result_parts = []
    for _, pattern in scored[:5]:
        parts = [
            f"## {pattern['title']}",
            f"```python\n{pattern['code']}\n```",
            f"{pattern['explanation']}",
            f"*Keywords: {', '.join(pattern['keywords'])}*",
        ]
        result_parts.append("\n".join(parts))

    return "\n\n---\n\n".join(result_parts)


@mcp.tool()
def pymc_error_lookup(error_message: str) -> str:
    """Look up common PyMC errors and their fixes.

    Args:
        error_message: Error message or description (e.g., 'divergences', 'shape mismatch', 'slow sampling')
    """
    scored = []
    for err in ERROR_PATTERNS:
        score = _keyword_match(error_message, err["keywords"])
        if score == 0 and _text_match(error_message, err["title"]):
            score = 1
        if score > 0:
            scored.append((score, err))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return (
            f"No specific error pattern found for '{error_message}'.\n\n"
            "General debugging tips:\n"
            "1. Check az.summary(idata) for r_hat and ESS\n"
            "2. Look at az.plot_trace(idata) for mixing\n"
            "3. Check for divergences in idata.sample_stats\n"
            "4. Try simplifying the model first, then add complexity\n"
            "5. Use pm.model_to_graphviz(model) to verify structure"
        )

    result_parts = []
    for _, err in scored[:3]:
        parts = [
            f"## {err['title']}",
            f"\n{err['fix']}",
        ]
        result_parts.append("\n".join(parts))

    return "\n\n---\n\n".join(result_parts)


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")
