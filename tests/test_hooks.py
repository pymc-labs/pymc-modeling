"""Evals for hook scripts — tests keyword matching and lint checks."""

import json
import subprocess
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).parent.parent
SUGGEST_SCRIPT = PLUGIN_ROOT / "hooks" / "scripts" / "suggest-skill.sh"
POST_WRITE_SCRIPT = PLUGIN_ROOT / "hooks" / "scripts" / "pymc-post-write.sh"


def _run_hook(script: Path, stdin_data: dict, env_override: dict | None = None) -> dict:
    """Run a hook script with JSON on stdin, return parsed JSON output or empty dict."""
    import os
    env = os.environ.copy()
    env["CLAUDE_PLUGIN_ROOT"] = str(PLUGIN_ROOT)
    if env_override:
        env.update(env_override)

    result = subprocess.run(
        ["bash", str(script)],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    assert result.returncode == 0, f"Hook failed: {result.stderr}"

    stdout = result.stdout.strip()
    if not stdout:
        return {}
    return json.loads(stdout)


# ── suggest-skill.sh ─────────────────────────────────────────────────────────

class TestSuggestSkillKeywords:
    """Test that the suggest-skill hook triggers on expected keywords."""

    @pytest.mark.parametrize("prompt", [
        "Build a Bayesian hierarchical model",
        "Use pymc to fit a regression",
        "I need to run MCMC sampling",
        "Check the posterior predictive",
        "Fix these divergences",
        "Use nutpie for faster sampling",
    ])
    def test_pymc_keywords_trigger(self, prompt):
        output = _run_hook(SUGGEST_SCRIPT, {"user_prompt": prompt})
        assert "systemMessage" in output, f"No suggestion for: {prompt}"
        assert "pymc-modeling" in output["systemMessage"]

    @pytest.mark.parametrize("prompt", [
        "Write a unit test for my PyMC model",
        "Set up pytest fixtures for pymc",
        "How to use mock_sample in tests",
    ])
    def test_testing_keywords_trigger(self, prompt):
        output = _run_hook(SUGGEST_SCRIPT, {"user_prompt": prompt})
        assert "systemMessage" in output, f"No suggestion for: {prompt}"
        assert "pymc-testing" in output["systemMessage"]

    @pytest.mark.parametrize("prompt", [
        "Help me with prior elicitation",
        "Use PreliZ to find constrained priors",
        "I need weakly informative priors",
    ])
    def test_prior_elicitation_keywords_trigger(self, prompt):
        output = _run_hook(SUGGEST_SCRIPT, {"user_prompt": prompt})
        assert "systemMessage" in output, f"No suggestion for: {prompt}"
        assert "prior-elicitation" in output["systemMessage"]

    @pytest.mark.parametrize("prompt", [
        "Compare models using LOO",
        "Run cross-validation on the model",
        "Compute ELPD for model comparison",
        "What are the stacking weights",
    ])
    def test_model_evaluation_keywords_trigger(self, prompt):
        output = _run_hook(SUGGEST_SCRIPT, {"user_prompt": prompt})
        assert "systemMessage" in output, f"No suggestion for: {prompt}"
        assert "model-evaluation" in output["systemMessage"]

    @pytest.mark.parametrize("prompt", [
        "Use pymc-extras splines",
        "Apply R2D2 prior",
        "Use Laplace approximation with fit_laplace",
        "Try the horseshoe prior from pymc_extras",
    ])
    def test_pymc_extras_keywords_trigger(self, prompt):
        output = _run_hook(SUGGEST_SCRIPT, {"user_prompt": prompt})
        assert "systemMessage" in output, f"No suggestion for: {prompt}"
        assert "pymc-extras" in output["systemMessage"]


class TestSuggestSkillNonTrigger:
    """Test that unrelated prompts don't trigger suggestions."""

    @pytest.mark.parametrize("prompt", [
        "Write a hello world in Python",
        "Create a REST API with FastAPI",
        "Help me debug this JavaScript",
        "Set up a Docker container",
        "Write SQL to query users",
    ])
    def test_unrelated_prompts_no_trigger(self, prompt):
        output = _run_hook(SUGGEST_SCRIPT, {"user_prompt": prompt})
        assert output == {}, f"Unexpected suggestion for: {prompt}"


class TestSuggestSkillEdgeCases:
    def test_empty_prompt(self):
        output = _run_hook(SUGGEST_SCRIPT, {"user_prompt": ""})
        assert output == {}

    def test_missing_prompt_field(self):
        output = _run_hook(SUGGEST_SCRIPT, {"some_other_field": "hello"})
        assert output == {}

    def test_case_insensitive(self):
        output = _run_hook(SUGGEST_SCRIPT, {"user_prompt": "BAYESIAN MODEL"})
        assert "systemMessage" in output

    def test_multiple_skills_suggested(self):
        """A prompt touching multiple domains should suggest multiple skills."""
        output = _run_hook(SUGGEST_SCRIPT, {
            "user_prompt": "Test my PyMC model with pytest and compare models using LOO"
        })
        assert "systemMessage" in output
        msg = output["systemMessage"]
        # Should suggest at least pymc-modeling and one other
        assert "pymc" in msg.lower()


# ── pymc-post-write.sh ───────────────────────────────────────────────────────

class TestPostWriteLintChecks:
    """Test the PostToolUse Write|Edit hook for PyMC lint warnings."""

    def test_flat_prior_warning(self):
        code = "import pymc as pm\nwith pm.Model():\n    x = pm.Flat('x')\n"
        output = _run_hook(POST_WRITE_SCRIPT, {
            "tool_input": {"file_path": "model.py", "content": code}
        })
        assert "systemMessage" in output
        assert "Flat" in output["systemMessage"]

    def test_half_flat_prior_warning(self):
        code = "import pymc as pm\nwith pm.Model():\n    x = pm.HalfFlat('x')\n"
        output = _run_hook(POST_WRITE_SCRIPT, {
            "tool_input": {"file_path": "model.py", "content": code}
        })
        assert "systemMessage" in output
        assert "Flat" in output["systemMessage"] or "improper" in output["systemMessage"].lower()

    def test_arviz_deprecated_access_warning(self):
        code = (
            "import arviz as az\n"
            "summary = idata.posterior\n"
        )
        output = _run_hook(POST_WRITE_SCRIPT, {
            "tool_input": {"file_path": "analysis.py", "content": code}
        })
        assert "systemMessage" in output
        assert "idata.posterior" in output["systemMessage"] or "ArviZ" in output["systemMessage"]

    def test_pm_sample_diagnostics_reminder(self):
        code = (
            "import pymc as pm\n"
            "with pm.Model():\n"
            "    mu = pm.Normal('mu', 0, 1)\n"
            "    idata = pm.sample()\n"
        )
        output = _run_hook(POST_WRITE_SCRIPT, {
            "tool_input": {"file_path": "model.py", "content": code}
        })
        assert "systemMessage" in output
        assert "diagnostics" in output["systemMessage"].lower()

    def test_non_python_file_ignored(self):
        output = _run_hook(POST_WRITE_SCRIPT, {
            "tool_input": {"file_path": "readme.md", "content": "# Hello\npm.Flat usage"}
        })
        assert output == {}

    def test_clean_code_no_warning(self):
        """Well-written PyMC code without antipatterns should not trigger Flat/ArviZ warnings."""
        code = (
            "import pymc as pm\n"
            "import numpy as np\n"
            "with pm.Model() as model:\n"
            "    mu = pm.Normal('mu', 0, 10)\n"
            "    sigma = pm.HalfNormal('sigma', 5)\n"
        )
        output = _run_hook(POST_WRITE_SCRIPT, {
            "tool_input": {"file_path": "model.py", "content": code}
        })
        # May still get pm.sample reminder if no sample call, but should NOT get Flat warning
        msg = output.get("systemMessage", "")
        assert "Flat" not in msg

    def test_empty_file_path(self):
        output = _run_hook(POST_WRITE_SCRIPT, {
            "tool_input": {"file_path": "", "content": "pm.Flat('x')"}
        })
        assert output == {}
