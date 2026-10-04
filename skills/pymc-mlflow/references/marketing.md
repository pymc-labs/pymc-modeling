# PyMC-Marketing autologging

Target [PyMC-Marketing 1.2.0](https://github.com/pymc-labs/pymc-marketing/releases/tag/1.2.0), not an older PyMC 5 integration. Its [dependency declarations](https://github.com/pymc-labs/pymc-marketing/blob/1.2.0/pyproject.toml) require Python >=3.12, PyMC >=6.3.1,<6.4, ArviZ/arviz-stats/arviz-plots >=1.2,<2, NumPy >=2,<2.5 and pymc-extras >=0.15,<0.16. Install MLflow separately: there is **no `mlflow` extra**. The reference environment uses PyMC 6.3.2, ArviZ 1.3.0 and MLflow 3.16.1.

Reuse the consuming project's compatible environment and package manager; do not
override declared pins. If using a uv-managed virtual environment that needs
these dependencies, the version-matched installation is:

```sh
uv pip install "pymc-marketing==1.2.0" "pymc==6.3.2" "arviz==1.3.0" "mlflow==3.16.1" "h5netcdf>=1.8.1" "h5py>=3.16.0"
```

NetCDF persistence needs a working xarray backend; the command supplies h5netcdf/h5py. Graph artifacts additionally need the Python `graphviz` package and the Graphviz system executable. Graph rendering is optional upstream and may log a message instead of producing a PDF.

## Generic PyMC: tracking, not model serving

Run this standalone block once in a **fresh Python process and disposable working directory**. It creates a local SQLite tracking store and local artifact directory; no server or GUI is required. The real but deliberately short sampling run demonstrates logging and round-trip infrastructure, not computational or scientific adequacy.

```python
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import numpy as np
import pymc as pm
import pymc_marketing.mlflow as pmm_mlflow
import xarray as xr

root = Path.cwd().resolve()
mlflow.set_tracking_uri(f"sqlite:///{root / 'autolog.db'}")
client = MlflowClient()
experiment_id = client.create_experiment(
    "generic-pymc-autolog", artifact_location=(root / "artifacts").as_uri()
)
rng = np.random.default_rng(20261004)
x = np.linspace(-1, 1, 24)
y = 0.4 + 0.8 * x + rng.normal(0, 0.3, len(x))

# Enable once; unrelated domain fit wrappers are not needed here.
pmm_mlflow.autolog(log_mmm=False, log_clv=False, log_bass=False)
with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.set_tags({"run_type": "demo", "inference_type": "mcmc"})
    mlflow.log_param("random_seed", 20261004)
    with pm.Model(coords={"observation": np.arange(len(x))}) as model:
        # A real data container provides nonempty constant_data input metadata.
        predictor = pm.Data("predictor", x, dims="observation")
        intercept = pm.Normal("intercept", 0, 1)
        slope = pm.Normal("slope", 0, 1)
        pm.Normal("y", intercept + slope * predictor, 0.3,
                  observed=y, dims="observation")
        idata = pm.sample(draws=100, tune=100, chains=2, cores=1,
                          nuts_sampler="pymc", random_seed=20261004,
                          progressbar=False)
        idata.to_netcdf(root / "idata_raw.nc", engine="h5netcdf")
        prior = pm.sample_prior_predictive(draws=30, random_seed=11)
        idata["prior"] = prior["prior"]
        idata["prior_predictive"] = prior["prior_predictive"]
        pp = pm.sample_posterior_predictive(
            idata, random_seed=12, progressbar=False,
        )
        idata["posterior_predictive"] = pp["posterior_predictive"]
        pm.compute_log_likelihood(idata, model=model, progressbar=False)
    # Generic autolog does NOT create idata.nc. Save the enriched groups explicitly.
    required_groups = {
        "posterior", "sample_stats", "constant_data", "observed_data",
        "prior", "prior_predictive", "posterior_predictive", "log_likelihood",
    }
    assert required_groups <= set(idata.children)
    pmm_mlflow.log_inference_data(idata)
    run_id = run.info.run_id

record = client.get_run(run_id)
artifacts = {item.path for item in client.list_artifacts(run_id)}
print("run:", run_id)
print("parameters:", record.data.params)
print("metrics:", record.data.metrics)
print("artifacts:", sorted(artifacts))
assert {"summary.html", "idata.nc", "model_repr.txt"} <= artifacts
assert "total_divergences" in record.data.metrics
assert record.data.params["draws"] == "100"
assert record.data.params["chains"] == "2"
assert record.data.params["likelihood"] == "Normal"
download_dir = root / "downloaded"
download_dir.mkdir()
idata_path = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path="idata.nc", dst_path=str(download_dir)
)
restored = xr.open_datatree(idata_path, engine="h5netcdf")
xr.testing.assert_equal(
    idata["posterior"].to_dataset(), restored["posterior"].to_dataset()
)
assert required_groups <= set(restored.children)
restored.close()
print("downloaded:", idata_path)
```

[Released integration source](https://github.com/pymc-labs/pymc-marketing/blob/1.2.0/pymc_marketing/mlflow.py) is the authority for logged names. `pm.sample` logging includes versions, model-derived parameters, input metadata, `total_divergences`, available sampling/time-per-draw metrics and the **`summary.html`** artifact. It stamps `idata.attrs["mlflow_run_id"]`. Autolog owns parameters such as `likelihood`, `draws`, `chains`, `posterior_samples` and sampler information: do not pre-log conflicting values. Use separate runs for different fits.

Generic autolog neither saves `idata.nc` automatically nor logs scalar ESS/R-hat metrics. Its summary artifact is not a scalar metric series. For additional scalar diagnostics, compute real ArviZ summaries and explicitly log finite values for selected estimands; do not invent unavailable diagnostics or interpret a short fit as converged. Follow [arviz-diagnostics](../../arviz-diagnostics/SKILL.md) for ESS, rank R-hat, MCSE and predictive assessment.

## Instrumentation lifecycle and file safety

Enable once per process, not once per run or notebook cell. Domain switches
`log_mmm`, `log_clv` and `log_bass` select which fit methods are wrapped.
**Released 1.2.0 has no unpatch branch for `disable=True`**: it directly wraps
callables again, so a disable call is not cleanup. Stop instrumentation by
ending the Python process; restart with the desired switches. Do not repeatedly
call `autolog` or rely on `silent=True` to hide errors.

Several helpers create then delete named files in the current directory: `summary.html`, graph files and `idata.nc`. Existing files can be overwritten or removed; concurrent runs sharing a directory can collide. Use an isolated writable working directory per process and protect user files. Passing an explicit unique path to `log_inference_data` changes the logged artifact basename; native MMM restoration specifically expects root artifact `idata.nc`.

`create_log_callback` is optional and only supports the native `nuts_sampler="pymc"` callback interface, not nutpie/NumPyro/BlackJAX. Its logged variables must be scalar; callback traces are not convergence diagnostics. MLflow system metrics are separately optional (`mlflow.start_run(log_system_metrics=True)` with MLflow's monitoring dependencies). They measure host resources, not posterior reliability.

## CLV: current constructor and native persistence

[BetaGeoModel's constructor](https://github.com/pymc-labs/pymc-marketing/blob/1.2.0/pymc_marketing/clv/models/beta_geo.py) takes keyword-only configuration, not training data. Pass data to `fit(data=..., method="mcmc")`; the [fit implementation](https://github.com/pymc-labs/pymc-marketing/blob/1.2.0/pymc_marketing/model_builder.py) uses `method`, not `fit_method`. `method="map"` is a point estimate and does not provide MCMC ESS/R-hat.

The following **separate standalone process** builds seeded synthetic repeat-purchase summaries with valid schema, fits real MCMC and compares native restored predictions. It does not demonstrate CLV pyfunc serving; this release supplies an MMM wrapper, not a `CLVWrapper`. As above, this is an infrastructure illustration only. Metadata logging is disabled because this CLV graph has no feature data containers; this is not a generic substitute for supplying `pm.Data`.

```python
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pymc_marketing.mlflow as pmm_mlflow
from pymc_marketing.clv import BetaGeoModel
import xarray as xr

root = Path.cwd().resolve()
mlflow.set_tracking_uri(f"sqlite:///{root / 'clv.db'}")
client = MlflowClient()
experiment_id = client.create_experiment(
    "clv-native-persistence", artifact_location=(root / "clv-artifacts").as_uri()
)
rng = np.random.default_rng(42)
frequency = rng.poisson(2, size=40)
T = rng.uniform(8, 12, size=40)
recency = np.where(frequency > 0, T * rng.uniform(0.1, 0.9, size=40), 0)
data = pd.DataFrame({
    "customer_id": np.arange(40), "frequency": frequency, "recency": recency, "T": T,
})
model = BetaGeoModel()
pmm_mlflow.autolog(log_mmm=False, log_clv=True, log_bass=False,
                   log_metadata_info=False)
with mlflow.start_run(experiment_id=experiment_id) as run:
    idata = model.fit(
        data=data, method="mcmc", random_seed=42, progressbar=False,
        draws=100, tune=100, chains=2, cores=1, nuts_sampler="pymc",
    )
    model.save(str(root / "clv-native.nc"))
    run_id = run.info.run_id
restored_model = BetaGeoModel.load(str(root / "clv-native.nc"))
xr.testing.assert_equal(
    model.idata["posterior"].to_dataset(),
    restored_model.idata["posterior"].to_dataset(),
)
xr.testing.assert_allclose(
    model.expected_purchases(future_t=4),
    restored_model.expected_purchases(future_t=4),
)
assert "idata.nc" in {a.path for a in client.list_artifacts(run_id)}
print("CLV native round trip:", run_id)
```

For MMM run artifacts, native reconstruction and the registry/serving boundary, continue to [MMM persistence](mmm-persistence.md).
