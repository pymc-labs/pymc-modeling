# MMM persistence and the pyfunc boundary

Target [PyMC-Marketing 1.2.0](https://github.com/pymc-labs/pymc-marketing/releases/tag/1.2.0) with the [dependencies in the autologging reference](marketing.md). Consult the released [MLflow integration](https://github.com/pymc-labs/pymc-marketing/blob/1.2.0/pymc_marketing/mlflow.py), [MMM implementation](https://github.com/pymc-labs/pymc-marketing/blob/1.2.0/pymc_marketing/mmm/mmm.py) and [inherited prediction implementation](https://github.com/pymc-labs/pymc-marketing/blob/1.2.0/pymc_marketing/model_builder.py), rather than copying stale docstring examples.

**Release limitation:** the built-in `MMMWrapper` always forwards `original_scale` when predicting. The current MMM posterior-predictive API has no such argument and forwards unknown keywords to `pm.sample_posterior_predictive`. Consequently the released built-in pyfunc posterior/point-prediction route is incompatible with the current MMM implementation. Logging/loading/registering the wrapper is not evidence that its prediction or serving works. Native MMM restoration and prediction remain a separate route. Do not suppress the error, alter global sampling, or silently replace the wrapper with a custom serving adapter.

## Standalone native round trip and pyfunc preparation

Run the first block in a fresh Python process and disposable working directory. It requires no external data or server. A seeded synthetic one-channel dataset and short real fit exercise tracking, serialization, reconstruction and next-period predictions; **they do not establish convergence, causal attribution, forecasting quality or budget recommendations**. See [arviz-diagnostics](../../arviz-diagnostics/SKILL.md) before interpreting a real fit.

```python
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pymc_marketing.mlflow as pmm_mlflow
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MMM
import xarray as xr

root = Path.cwd().resolve()
mlflow.set_tracking_uri(f"sqlite:///{root / 'mmm.db'}")
client = MlflowClient()
experiment_id = client.create_experiment(
    "mmm-native-roundtrip", artifact_location=(root / "mmm-artifacts").as_uri()
)
rng = np.random.default_rng(42)
X = pd.DataFrame({
    "date": pd.date_range("2025-01-06", periods=24, freq="W-MON"),
    "media": rng.uniform(10, 100, size=24),
})
y = pd.Series(20 + 0.3 * X["media"] + rng.normal(0, 2, size=24), name="y")
X_pred = pd.DataFrame({
    "date": pd.date_range(X["date"].iloc[-1] + pd.Timedelta(weeks=1),
                          periods=4, freq="W-MON"),
    "media": rng.uniform(10, 100, size=4),
})
mmm = MMM(
    date_column="date", channel_columns=["media"],
    adstock=GeometricAdstock(l_max=2), saturation=LogisticSaturation(),
)
# Enable once. MMM.fit additionally logs the fully attributed idata.nc.
pmm_mlflow.autolog(log_mmm=True, log_clv=False, log_bass=False)
with mlflow.start_run(experiment_id=experiment_id) as run:
    idata = mmm.fit(
        X=X, y=y, method="mcmc", draws=100, tune=100, chains=2, cores=1,
        nuts_sampler="pymc", random_seed=42, progressbar=False,
    )
    # The actual default path is 'model'; log_mmm returns None, not ModelInfo.
    pmm_mlflow.log_mmm(mmm, include_last_observations=True)
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

artifacts = {a.path for a in client.list_artifacts(run_id)}
assert "idata.nc" in artifacts
# Loads the built-in wrapper, but does not certify its incompatible predict route.
pyfunc_model = pmm_mlflow.load_mmm(run_id=run_id, artifact_path="model")
assert isinstance(pyfunc_model, mlflow.pyfunc.PyFuncModel)

native = pmm_mlflow.load_mmm(
    run_id=run_id, full_model=True, keep_idata=False,
    dst_path=str(root / "native-download"),
)
assert isinstance(native, MMM)
xr.testing.assert_equal(
    mmm.idata["posterior"].to_dataset(), native.idata["posterior"].to_dataset()
)
assert "fit_data" in native.idata.children

# Native current API accepts X, not X_pred as a keyword, and no original_scale.
# Include training carry-in only because these are immediately following dates.
prediction_kwargs = {
    "extend_idata": False, "include_last_observations": True,
    "random_seed": 99, "progressbar": False,
}
original_prediction = mmm.predict(X=X_pred, **prediction_kwargs)
restored_prediction = native.predict(X=X_pred, **prediction_kwargs)
assert original_prediction.shape == (len(X_pred),)
assert np.isfinite(restored_prediction).all()
np.testing.assert_allclose(original_prediction, restored_prediction)
print("run:", run_id)
print("pyfunc URI (prediction limitation applies):", model_uri)
print("native predictions in model target scale:", restored_prediction)
print("run artifacts:", sorted(artifacts))
```

Verification contract: the block must log `idata.nc`, load both representations, compare actual native posterior datasets and make seeded predictions with the original and reconstructed native MMM. Equality checks concern persistence, not fit adequacy. `summary.html` and available sampler metrics are supplied by generic autolog; MMM autolog does not automatically call `log_mmm`.

`load_mmm`'s actual defaults are `full_model=False`, `keep_idata=False`, `artifact_path="model"` (some released docstrings disagree). The two persistence paths are distinct:

| Path | Contents and use |
|---|---|
| `log_mmm` then `load_mmm(run_id)` | MLflow pyfunc wrapper; suitable for model artifact/registry preparation, subject to the released prediction incompatibility. Do not claim posterior-free storage merely because it is wrapped as pyfunc. |
| MMM autolog or explicit `log_inference_data(mmm.idata)` then `load_mmm(run_id, full_model=True)` | Downloads root artifact `idata.nc` and invokes `MMM.load` to rebuild the native model, training data, configuration and posterior. |

`full_model=True` requires the **separate root `idata.nc` artifact**; a registered pyfunc alone does not supply this reconstruction path. If adding groups after fit, log the enriched `mmm.idata` explicitly inside its run. Preserve reconstruction attrs and `fit_data`; a generic posterior-only NetCDF is not a native MMM checkpoint. `keep_idata=False` eagerly loads groups then removes the downloaded file; it does **not** discard the posterior from the returned model. Supply a dedicated empty `dst_path`; upstream removal may warn if it cannot remove that directory. Helpers write/delete `idata.nc` in the working directory, so isolate concurrent fits and avoid existing files.

## Prediction semantics and upstream incompatibility

The wrapper's actual PythonModel signature is `predict(self, context, model_input: pandas.DataFrame, params: dict | None = None)`. MLflow supplies `context`; users call `PyFuncModel.predict(data, params=None)`, not a context argument. For a functioning point-prediction route, the inherited MMM `predict` would return a NumPy array of posterior-predictive means, not a DataTree, interval or uncertainty distribution. No generic PyMC serving flavor or CLV wrapper is supplied by this integration.

In the current native MMM, `sample_posterior_predictive(X=..., ...)` returns an xarray Dataset, with `combined=True` combining chain/draw into sample. `X_pred` in the demo is just a local variable name. `original_scale` is not a native posterior-prediction keyword in this release. Native `y` predictions are in the model's scaled target units; do not label them revenue units. For original-unit draws, explicitly create `y_original_scale` using `mmm.add_original_scale_contribution_variable(["y"])` and request that variable through `var_names` on native posterior-predictive sampling. `original_scale` is valid on other APIs such as `sample_saturation_curve`, not an interchangeable keyword across methods.

The wrapper exposes prediction-method overrides, but `log_mmm` does not declare an MLflow parameter schema; do not assume `PyFuncModel.predict(..., params=...)` transports method switches unchanged. The wrapper also defaults an absent/empty params dictionary to `{"predict_method": "predict"}`. Neither is a repair for the `original_scale` mismatch.

The following **intentional failure reproduction**, not a success/demo check, continues with `pyfunc_model` and `X_pred` from the first block. It invokes the actual built-in route without catching or suppressing the exception. Run separately after the native checks; on released 1.2.0 its `original_scale` forwarding is unsupported.

```python
# Continuation: pyfunc_model and X_pred from the standalone block.
pyfunc_prediction = pyfunc_model.predict(X_pred)
```

## Registry continuation is not serving

The following optional **continuation** uses `mlflow`, `model_uri`, `X_pred`, `root` and the existing SQLite tracking store from the first block. Registration creates a named version; a `runs:/...` URI identifies a run's model, whereas `models:/name/version` identifies a registry version. `log_mmm(..., registered_model_name=...)` can also register directly, but still returns None. Register only trusted artifacts: pyfunc deserialization loads executable Python objects.

```python
# Continuation: run this in the same process after the standalone block.
mlflow.set_registry_uri(f"sqlite:///{root / 'mmm.db'}")
registered = mlflow.register_model(model_uri=model_uri, name="synthetic-mmm")
registry_uri = f"models:/{registered.name}/{registered.version}"
registered_pyfunc = mlflow.pyfunc.load_model(registry_uri)
assert isinstance(registered_pyfunc, mlflow.pyfunc.PyFuncModel)
print("registered wrapper (prediction limitation still applies):", registry_uri)
```

Neither registration nor local `load_model` starts a service. Actual serving additionally requires a working prediction implementation, validated request/response schema and units, compatible locked runtime, transport/deployment configuration and an exercised request. The released wrapper mismatch blocks a truthful built-in serving demonstration; retain the native reconstruction route rather than claiming an endpoint exists.
