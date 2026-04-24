"""End-to-end integration test for the benchmark pipeline.

Creates synthetic run data, scores it, loads scores, and generates a report.
No Claude CLI calls — tests the full scoring/analysis pipeline.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.analysis import compute_effect_sizes, generate_report, load_scores
from src.scorer import score_run


def _create_synthetic_run(
    run_dir: Path,
    task_id: str,
    condition: str,
    rep: int,
    n_chains=4,
    n_draws=500,
    quality="good",
):
    """Create a complete synthetic run directory with model.py, results.nc, metadata.json, turns.jsonl."""
    import arviz as az

    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42 + rep)

    # Write model.py with varying quality
    if quality == "good":
        code = """
import pymc as pm
import arviz as az
import numpy as np

with pm.Model(coords={"school": range(8)}) as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    offset = pm.Normal("offset", mu=0, sigma=1, dims="school")
    theta = pm.Deterministic("theta", mu + sigma * offset, dims="school")

    effects = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    std = np.array([15, 10, 16, 11, 9, 11, 10, 18])
    y = pm.Normal("y", mu=theta, sigma=std, observed=effects, dims="school")

    prior = pm.sample_prior_predictive(draws=100, random_seed=42)
    idata = pm.sample(1000, nuts_sampler="nutpie", random_seed=42)

idata.to_netcdf("results.nc")

summary = az.summary(idata)
print(summary)
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div}")
"""
    else:
        code = """
import pymc as pm
import numpy as np

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    idata = pm.sample(200)
idata.to_netcdf("results.nc")
"""
    (run_dir / "model.py").write_text(code)

    # Create synthetic InferenceData
    if quality == "good":
        mu = rng.normal(8, 2, (n_chains, n_draws))
        sigma = np.abs(rng.normal(5, 1, (n_chains, n_draws)))
        offset = rng.normal(0, 1, (n_chains, n_draws, 8))
        theta = mu[:, :, None] + sigma[:, :, None] * offset

        posterior_vars = {"mu": mu, "sigma": sigma, "offset": offset, "theta": theta}
        dims = {
            "mu": [],
            "sigma": [],
            "offset": ["school"],
            "theta": ["school"],
            "diverging": [],
        }
        coords = {
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "school": np.arange(8),
        }
    else:
        posterior_vars = {"mu": rng.normal(0, 1, (n_chains, n_draws))}
        dims = {"mu": [], "diverging": []}
        coords = {"chain": np.arange(n_chains), "draw": np.arange(n_draws)}

    diverging = np.zeros((n_chains, n_draws), dtype=bool)

    idata = az.from_dict(
        {"posterior": posterior_vars, "sample_stats": {"diverging": diverging}},
        coords=coords,
        dims=dims,
    )
    idata.to_netcdf(str(run_dir / "results.nc"))

    # Write metadata.json
    metadata = {
        "task_id": task_id,
        "condition": condition,
        "rep": rep,
        "success": True,
        "wall_time": 45.0 + rng.normal(0, 5),
        "num_turns": 6 + int(rng.integers(0, 4)),
        "input_tokens": 5000,
        "output_tokens": 2000,
        "total_input_tokens": 5000,
        "cache_creation_tokens": 3000,
        "cache_read_tokens": 0,
        "cost_usd": 0.05,
        "tool_calls": [],
        "error": "",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Write turns.jsonl
    turns = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"file_path": "/tmp/work/model.py"},
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "python model.py"},
                    }
                ]
            },
        },
    ]
    with open(run_dir / "turns.jsonl", "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")


def _mock_appropriateness_llm(run_dir, task_id):
    """Mock for score_model_appropriateness_llm that returns a fixed score."""
    model_py = run_dir / "model.py"
    if not model_py.exists():
        return 0, {"reason": "no model.py", "method": "mock"}
    return 3, {"method": "mock", "reasoning": "mocked for integration test"}


class TestBenchmarkIntegration:
    def test_full_pipeline(self, tmp_path):
        """Full pipeline: create runs -> score -> load scores -> generate report."""
        runs_dir = tmp_path / "runs"
        scores_dir = tmp_path / "scores"
        analysis_dir = tmp_path / "analysis"

        task_id = "T1_hierarchical"

        # Create synthetic runs for both conditions
        for condition in ["no_skill", "with_skill"]:
            for rep in range(3):
                quality = "good" if condition == "with_skill" else "good"
                run_dir = runs_dir / f"{task_id}_{condition}_rep{rep}"
                _create_synthetic_run(run_dir, task_id, condition, rep, quality=quality)

        # Score each run (mock the LLM judge to avoid Claude CLI calls)
        scores_dir.mkdir(parents=True, exist_ok=True)
        with patch(
            "src.scorer.score_model_appropriateness_llm", _mock_appropriateness_llm
        ):
            for run_dir in sorted(runs_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                meta = json.loads((run_dir / "metadata.json").read_text())
                result = score_run(
                    run_dir, meta["task_id"], meta["condition"], meta["rep"]
                )

                # Verify score has all criteria
                assert result.total > 0
                assert 0 <= result.model_produced <= 5
                assert 0 <= result.convergence <= 5
                assert 0 <= result.best_practices <= 5
                assert 0 <= result.workflow <= 5
                assert 0 <= result.parameter_recovery <= 5

                # Save score
                score_file = (
                    scores_dir
                    / f"{meta['task_id']}_{meta['condition']}_rep{meta['rep']}.json"
                )
                score_file.write_text(
                    json.dumps(
                        {
                            "task_id": meta["task_id"],
                            "condition": meta["condition"],
                            "rep": meta["rep"],
                            "model_produced": result.model_produced,
                            "convergence": result.convergence,
                            "model_appropriateness": result.model_appropriateness,
                            "best_practices": result.best_practices,
                            "workflow": result.workflow,
                            "parameter_recovery": result.parameter_recovery,
                            "total": result.total,
                            "passed": result.passed,
                            "retries": result.retries,
                        },
                        indent=2,
                    )
                )

        # Load scores
        df = load_scores(scores_dir)
        assert len(df) == 6  # 2 conditions * 3 reps
        assert "total" in df.columns
        assert "passed" in df.columns

        # Compute effect sizes
        effects = compute_effect_sizes(df)
        assert not effects.is_empty()

        # Generate report
        report = generate_report(scores_dir=scores_dir, output_dir=analysis_dir)
        assert "Analysis Report" in report
        assert "T1_hierarchical" in report
        assert "Cohen's d" in report
        assert (analysis_dir / "report.md").exists()
        assert (analysis_dir / "summary.csv").exists()

    def test_pipeline_with_failed_run(self, tmp_path):
        """Pipeline handles a mix of passed and failed runs."""
        runs_dir = tmp_path / "runs"
        scores_dir = tmp_path / "scores"

        task_id = "T1_hierarchical"

        # Create one good and one bad run
        good_dir = runs_dir / f"{task_id}_with_skill_rep0"
        _create_synthetic_run(good_dir, task_id, "with_skill", 0, quality="good")

        bad_dir = runs_dir / f"{task_id}_no_skill_rep0"
        _create_synthetic_run(bad_dir, task_id, "no_skill", 0, quality="bad")

        # Score both (mock the LLM judge)
        with patch(
            "src.scorer.score_model_appropriateness_llm", _mock_appropriateness_llm
        ):
            for run_dir in sorted(runs_dir.iterdir()):
                meta = json.loads((run_dir / "metadata.json").read_text())
                result = score_run(
                    run_dir, meta["task_id"], meta["condition"], meta["rep"]
                )

                scores_dir.mkdir(parents=True, exist_ok=True)
                score_file = (
                    scores_dir
                    / f"{meta['task_id']}_{meta['condition']}_rep{meta['rep']}.json"
                )
                score_file.write_text(
                    json.dumps(
                        {
                            "task_id": meta["task_id"],
                            "condition": meta["condition"],
                            "rep": meta["rep"],
                            "model_produced": result.model_produced,
                            "convergence": result.convergence,
                            "model_appropriateness": result.model_appropriateness,
                            "best_practices": result.best_practices,
                            "workflow": result.workflow,
                            "parameter_recovery": result.parameter_recovery,
                            "total": result.total,
                            "passed": result.passed,
                            "retries": result.retries,
                        },
                        indent=2,
                    )
                )

        df = load_scores(scores_dir)
        assert len(df) == 2
