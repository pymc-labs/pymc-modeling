# Manual PyMC tracking

Generic tracking needs no PyMC-Marketing dependency. This example targets
[PyMC 6.3.2](https://github.com/pymc-devs/pymc/releases/tag/v6.3.2),
[ArviZ 1.3.0](https://github.com/arviz-devs/arviz/releases/tag/v1.3.0) and
[MLflow 3.16.1](https://github.com/mlflow/mlflow/releases/tag/v3.16.1).
Install those packages plus `h5netcdf` and `h5py` in an isolated environment.
Native PyMC NUTS needs no optional sampler. This is tracking infrastructure,
not native model restoration, registry publication or a serving endpoint.

## Complete local round trip

The seeded synthetic Gaussian regression uses four chains, 500 warmup and 500
retained draws per chain. Change the sampling configuration for the estimand's
exploration and precision requirements; no draw-count threshold certifies inference.
The scratch directory persists after Python exits and is outside the checkout.
For actual work use managed durable storage instead of OS-cleanable temporary
storage. Never commit the SQLite database, downloaded artifacts or source data.

```python
from hashlib import sha256
from importlib.metadata import distributions, version
from pathlib import Path
import platform
import tempfile

import arviz as az
import mlflow
import numpy as np
import pymc as pm
import xarray as xr
from mlflow.tracking import MlflowClient

root = Path(tempfile.mkdtemp(prefix="pymc-mlflow-tracking-")).resolve()
artifacts = root / "artifacts"
artifacts.mkdir()
mlflow.set_tracking_uri(f"sqlite:///{root / 'tracking.db'}")
client = MlflowClient()
experiment_id = client.create_experiment(
    "gaussian-regression-demo", artifact_location=artifacts.as_uri(),
)
seed = 20260911
rng = np.random.default_rng(seed)
x = np.linspace(-1.0, 1.0, 80)
y = 0.5 + 1.2 * x + rng.normal(0.0, 0.5, x.size)
inputs = root / "inputs.npz"
np.savez(inputs, x=x, y=y, obs=np.arange(x.size))
config = dict(draws=500, tune=500, chains=4, cores=1,
              target_accept=0.9, nuts_sampler="pymc", random_seed=seed)
with pm.Model(coords={"obs": np.arange(x.size)}) as model:
    alpha = pm.Normal("alpha", 0.0, 1.0)
    beta = pm.Normal("beta", 0.0, 1.0)
    pm.Normal("response", alpha + beta * x, 0.5, observed=y, dims="obs")

with mlflow.start_run(experiment_id=experiment_id,
                      run_name="manual-gaussian", log_system_metrics=False) as run:
    run_id = run.info.run_id
    print("run_id:", run_id, "local storage:", root)
    mlflow.set_tags({"run_type": "demo", "inference_type": "mcmc",
                     "model_family": "gaussian_regression",
                     "evaluation_target": "exchangeable_row"})
    mlflow.log_params({**config, "n_observations": x.size,
                       "model_spec": "alpha,beta~Normal(0,1); y~Normal(alpha+beta*x,0.5)",
                       "data_sha256": sha256(inputs.read_bytes()).hexdigest(),
                       **{f"version_{name}": version(name)
                          for name in ("pymc", "arviz", "mlflow")}})
    mlflow.log_dict({"python": platform.python_version(),
                     "platform": platform.platform(),
                     "packages": {d.metadata["Name"]: d.version
                                  for d in distributions()}}, "provenance/environment.json")
    mlflow.log_artifact(str(inputs), artifact_path="data")  # Synthetic, authorized data.
    idata = pm.sample(model=model, **config, progressbar=False)
    raw = root / "idata_raw.nc"
    idata.to_netcdf(raw, engine="h5netcdf")  # Preserve the unprocessed fit first.
    mlflow.log_artifact(str(raw), artifact_path="inference")

    pm.compute_log_likelihood(idata, model=model, progressbar=False)
    pp = pm.sample_posterior_predictive(
        idata, model=model, var_names=["response"], random_seed=seed, progressbar=False,
    )
    idata["posterior_predictive"] = pp["posterior_predictive"]
    enriched = root / "idata_enriched.nc"
    idata.to_netcdf(enriched, engine="h5netcdf")
    mlflow.log_artifact(str(enriched), artifact_path="inference")
    print("groups:", idata.groups)

    summary = az.summary(idata, var_names=["alpha", "beta"], ci_prob=0.89,
                         ci_kind="eti", round_to="none")
    summary_path = root / "arviz_summary.csv"
    summary.to_csv(summary_path)
    mlflow.log_artifact(str(summary_path), artifact_path="diagnostics")
    print(summary)
    loo = az.loo(idata, var_name="response", pointwise=True)
    flagged = (loo.pareto_k > loo.good_k) | ~np.isfinite(loo.pareto_k)
    pointwise = xr.Dataset({"elpd_i": loo.elpd_i, "pareto_k": loo.pareto_k})
    pointwise_path = root / "loo_pointwise.nc"
    pointwise.to_netcdf(pointwise_path, engine="h5netcdf")
    mlflow.log_artifact(str(pointwise_path), artifact_path="diagnostics")
    posterior = idata["posterior"]
    metrics = {
        "retained_draws_total": posterior.sizes["chain"] * posterior.sizes["draw"],
        "divergences": idata["sample_stats"]["diverging"].sum().item(),
        "rhat_max": np.max(summary["r_hat"].values),
        "ess_bulk_min": np.min(summary["ess_bulk"].values),
        "ess_tail_min": np.min(summary["ess_tail"].values),
        "mcse_mean_max": np.max(summary["mcse_mean"].values),
        "loo_elpd": loo.elpd, "loo_se": loo.se, "loo_p": loo.p,
        "loo_good_k": loo.good_k, "pareto_k_max": np.max(loo.pareto_k.values),
        "pareto_k_flagged": flagged.sum().item(), "loo_warning": int(loo.warning),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        mlflow.set_tag("diagnostics_status", "nonfinite")
        raise ValueError("Nonfinite diagnostics: inspect preserved inference and summaries.")
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
    mlflow.set_tag("diagnostics_status", "recorded")  # Not a scientific pass/fail.
    print("LOO:", loo.elpd, loo.p, loo.se, "good_k:", loo.good_k,
          "flagged:", metrics["pareto_k_flagged"], "warning:", loo.warning)

    downloaded = Path(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="inference/idata_enriched.nc",
        dst_path=str(root / "downloads"),
    ))
    assert sha256(downloaded.read_bytes()).digest() == sha256(enriched.read_bytes()).digest()
    with xr.open_datatree(downloaded, engine="h5netcdf") as reopened:
        assert set(reopened.groups) == set(idata.groups)
        for group in idata.groups:
            xr.testing.assert_equal(reopened[group].to_dataset(), idata[group].to_dataset())
        print("downloaded artifact:", downloaded, "round-trip groups:", reopened.groups)

matches = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=("tags.run_type = 'demo' AND metrics.retained_draws_total >= 1000 "
                   f"AND attributes.run_id = '{run_id}'"),
)
assert run_id in matches["run_id"].values
print("numeric metric filter matched:", matches["run_id"].tolist())
```

The checks compare uploaded/downloaded bytes and actual group variables and
coordinates, not just filenames. Expect `posterior`, `log_likelihood` and
`posterior_predictive` alongside observed data and sampler statistics. Successful
sampling, artifact retrieval and a numeric filter establish infrastructure only.
Inspect divergences, rank R-hat, bulk/tail ESS and estimand-specific MCSE together
with chain/rank behavior; investigate warnings rather than changing seeds to pass.
An 89% ETI describes posterior uncertainty, not Monte Carlo error. In-sample PPCs
are not held-out validation. See [ArviZ diagnostics](../../arviz-diagnostics/SKILL.md)
for actual predictive criticism, plots and high-Pareto-k remedies.

## Metadata, comparisons and failures

- Log tags, immutable sampling parameters and provenance **before** fitting.
  Parameter values cannot be changed within a run: a changed configuration gets
  a new run, not an overwritten parameter. Use lowercase classification values
  (`demo`, `exploratory`, `reportable`, `mcmc`, `mock`); a `reportable` tag records
  intended use, not an automatic certification.
- Parameters and tags are strings: search `params.chains = '4'`, not a numeric
  inequality. Metrics are numeric: use `metrics.ess_bulk_min > 400` as a screening
  query, not a validity guarantee. Search supports `AND`, not SQL `OR`;
  **`params.* IN (...)` is unsupported**. For multiple parameter choices, issue
  separate equality searches and deduplicate runs, or filter returned rows locally.
  Backtick field names containing spaces or punctuation.
- Compare LOO only for the same response, density scale, observation identities,
  ordering and prediction target. Preserve pointwise values and report paired
  ELPD-difference uncertainty, PSIS warnings and substantive relevance, not only
  ranks. These rows support exchangeable-row LOO, not future-time or new-group
  claims. Genuine held-out evaluation fits preprocessing and the model on training
  data only, then scores unseen outcomes on an aligned, documented split. Record
  split identities, score definition and uncertainty; never label training PPCs
  or training RMSE as held-out performance.
- Let sampling, diagnostic, serialization and upload errors propagate. The run
  context marks uncaught exceptions as failed; the on-disk raw fit survives later
  postprocessing failures. Do not log missing diagnostics as zero or manufacture
  sampler statistics. MLflow tracking is not an atomic transaction: already logged
  metadata/artifacts remain available on a failed run.
- Save model source/revision, environment lock, data identity and preprocessing
  alongside seeds, sampler, chain count, warmup, retained draws and adaptation
  settings in real projects. The demo logs synthetic inputs and installed versions;
  versions alone are not a complete environment lock. Seeds do not guarantee
  bitwise reproduction across hardware, package versions or sampling backends.

## Storage, resource metrics and mocks

Use consistent names such as `inference/idata_raw.nc`,
`inference/idata_enriched.nc`, `diagnostics/arviz_summary.csv` and
`diagnostics/loo_pointwise.nc`. Serialize to a real local path, then upload that
path; `log_artifact` does not serialize a DataTree. Reopen the local path returned
by `download_artifacts`, not a `runs:/` URI with an xarray reader. DataTrees retain
inference results but do not reconstruct the executable PyMC model. SQLite stores
metadata; artifact storage is separate and must also be backed up and retained.
Observed/constant data and coordinates can contain sensitive information: require
upload authorization, access controls and retention rules, and never log secrets.

For optional CPU/memory/disk/network monitoring, install `psutil` and change the
run to `log_system_metrics=True`. NVIDIA GPU monitoring additionally needs
`nvidia-ml-py`; AMD/HIP uses `pyrsmi`. Metrics live under `system/`; default polling
is every 10 seconds, so short runs may yield none. Set sampling interval and samples
before logging deliberately; wall-clock resource use is not an inference diagnostic.

Mock runs belong in scoped tests with an explicit `inference_type=mock` tag, never
in posterior-quality comparisons. Prior-generated mock draws and repeated chain
labels cannot justify R-hat, ESS, LOO or posterior claims. Do not globally replace
`pm.sample`; use the existing [model-testing reference](../../pymc-modeling/references/model-testing.md).

## Primary API sources

- PyMC 6.3.2: [sampling](https://github.com/pymc-devs/pymc/blob/v6.3.2/pymc/sampling/mcmc.py), [likelihood enrichment](https://github.com/pymc-devs/pymc/blob/v6.3.2/pymc/stats/log_density.py).
- ArviZ-Stats 1.3.0: [summary](https://github.com/arviz-devs/arviz-stats/blob/v1.3.0/src/arviz_stats/summary.py), [LOO fields](https://github.com/arviz-devs/arviz-stats/blob/v1.3.0/src/arviz_stats/loo/loo.py).
- MLflow 3.16.1: [fluent tracking API](https://github.com/mlflow/mlflow/blob/v3.16.1/mlflow/tracking/fluent.py), [search syntax](https://mlflow.org/docs/3.16.1/ml/search/search-runs/), [artifact download](https://github.com/mlflow/mlflow/blob/v3.16.1/mlflow/artifacts/__init__.py), [system metrics](https://mlflow.org/docs/3.16.1/ml/tracking/system-metrics/).
- xarray: [DataTree NetCDF writing](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.to_netcdf.html), [opening a DataTree](https://docs.xarray.dev/en/stable/generated/xarray.open_datatree.html).
