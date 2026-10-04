---
name: pymc-mlflow
description: >-
  Track PyMC experiments with MLflow, record sampling configuration and diagnostics,
  save and retrieve DataTree artifacts, and compare runs. Use for
  pymc_marketing.mlflow autologging, MMM model persistence and MLflow registry
  preparation, or CLV fit tracking. Distinguishes experiment tracking, native
  model restoration and prediction wrappers from deployed services.
---

# PyMC experiment tracking with MLflow

Record how a fit was produced, preserve its inference output, and retrieve it
without changing the consuming project's modeling environment. `pymc-mlflow`
is this skill's name; the optional integration module is
`pymc_marketing.mlflow`, not a separate `pymc-mlflow` package.

## Choose the integration

| Need | Reference |
|---|---|
| Track ordinary PyMC models without installing PyMC-Marketing | [Manual tracking and DataTree artifacts](references/tracking.md) |
| Use maintained PyMC-Marketing autologging or track CLV fits | [Autologging and domain-model tracking](references/marketing.md) |
| Save/restore MMM models or log a prediction wrapper to MLflow | [MMM persistence and registry preparation](references/mmm-persistence.md) |

Examples target PyMC 6/DataTree and modern ArviZ. PyMC-Marketing 1.2.0 requires
PyMC `>=6.3.1,<6.4.0` and ArviZ `>=1.2.0,<2.0`; MLflow is a separate optional
installation. Inspect installed versions and signatures before enabling
integration. Do not downgrade or replace a working modeling environment just to
use autologging. Direct MLflow tracking does not require PyMC-Marketing.

## Preserve the fit before interpreting it

1. **Identify the run.** Set a tracking URI and project experiment. Log source/data
   identities, package versions, seed and actual sampler configuration. Use
   consistent lowercase tags for exploratory, reportable and mock runs; tags
   describe provenance, not proof of inferential quality.
2. **Fit inside an explicit run.** Use `with mlflow.start_run():` so exceptions
   mark failed runs. Log configuration before expensive work. MLflow parameters
   cannot change within a run; do not overwrite integration-owned parameters
   such as `likelihood`, `draws` or `chains` with different values.
3. **Persist actual inference.** Save a local raw DataTree before diagnostics or
   enrichment. Generic `pymc_marketing.mlflow.autolog()` does not save `idata.nc`:
   explicitly serialize/upload it or call `log_inference_data`. Preserve the
   groups needed by the downstream task; adding likelihoods or predictions later
   requires saving the enriched output too.
4. **Log diagnostics, not reassurance.** Preserve chain identities and actual
   draws. Missing/nonfinite diagnostics are unresolved, not zero. A draw-count
   threshold, run tag or attractive summary is not convergence evidence. Use
   `arviz-diagnostics` for computation, predictive checks and reliable comparison.
5. **Retrieve and check.** Download artifacts through MLflow to local paths,
   reopen the DataTree, and check groups, coordinates and required variables.
   A remote artifact URI is not a local NetCDF filename.
6. **Compare like with like.** Numeric filtering uses metrics, not string
   parameters. Compare predictive scores only for the same response scale,
   observations and validation target; report uncertainty and Pareto-k warnings.
   Training PPCs are not predictions for held-out inputs.

## Keep tracking distinct from serving

- A posterior NetCDF file is inference output, not an executable generic PyMC
  model, preprocessing pipeline or prediction endpoint.
- PyMC-Marketing supports native model restoration. MMM additionally has an
  MLflow prediction wrapper, but release 1.2.0's built-in point/posterior-prediction
  route fails on an unsupported `original_scale` argument. Use the documented
  native restoration route; do not claim the wrapper is ready for serving.
  CLV tracking does not supply an equivalent CLV serving wrapper.
- A `runs:/` URI addresses a run artifact; a `models:/` URI addresses a registered
  model. Logging or registering either does not deploy a service. Validate the
  selected prediction method, input schema, scale and runtime dependencies before
  treating a wrapper as usable for serving.

## Avoid workflow contamination

Do not leave `pm.sample` globally replaced with `mock_sample`. Mock output is
prior-generated infrastructure data, not posterior inference, and may have no
sampler statistics. Scope test patches and restore the original callable; see
[meaningful model testing](../pymc-modeling/references/model-testing.md).
Enable autologging once per process, not repeatedly in a notebook loop.

Keep tracking databases, posterior files and raw data out of source control.
Artifacts and autologged dataset inputs can contain observed values, predictors
and identifying coordinates; use authorized storage and access controls. Do not
upload credentials or unapproved data. Diagnose full unthinned chains before
making any explicitly labeled reduced-storage artifact.

For model construction and predictions, use `pymc-modeling`; for convergence,
PPCs and LOO interpretation, use `arviz-diagnostics`. This skill records those
workflows; it does not replace them.
