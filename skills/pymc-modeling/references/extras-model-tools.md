# Extras model tools preserve explicit modeling choices

Use these tools when configuration, reuse or deployment requires them. A plain
`with pm.Model()` block is usually simpler for one model. APIs below target extras
0.15.1; start with [compatibility and feature selection](pymc-extras.md).

## Prior factories separate configuration from model creation

`Prior` describes a distribution before a model exists; `create_variable(name)`
registers it in the active model. Nested factories create named hyperparameters.
Use declared units and prior-predictive checks just as with direct PyMC calls.

```python
import json
import numpy as np
import pymc as pm
from pymc_extras.prior import Prior

coefficient_prior = Prior(
    "Normal", mu=Prior("Normal", mu=0, sigma=1),
    sigma=Prior("HalfNormal", sigma=0.5),
    dims="group", centered=False,
)
restored_prior = Prior.from_dict(json.loads(json.dumps(coefficient_prior.to_dict())))
with pm.Model(coords={"group": ["a", "b", "c"]}) as hierarchical:
    beta = restored_prior.create_variable("beta")
    pm.Normal("y", mu=beta, sigma=0.3, observed=np.array([0.1, -0.2, 0.4]),
              dims="group")
    prior_predictions = pm.sample_prior_predictive(draws=100, random_seed=42)
```

- In this release non-centering supports Normal, StudentT and ZeroSumNormal;
  it is not an arbitrary-family option. Include required location/scale parameters.
- Scalar hyperpriors are shared; a nested prior with its own dims creates separate
  hyperparameters over those axes. Do not accidentally change the pooling model.
- Ordinary tensors still have positional axes. `handle_dims` and
  `create_dim_handler` transpose/add axes by dimension names; they do not join or
  reorder observations by coordinate values. Validate alignment yourself.
- `create_variable(..., xdist=True)` opts into experimental `pymc.dims`
  distributions. Give named `xarray.DataArray` parameters, explicit `core_dims`
  for multivariate support, and verify operator support before mixing tensor types.
- `Prior("Normal", sigma=...).create_likelihood_variable(name, mu=..., observed=...)`
  works for families with a `mu` parameter and requires `mu` not already configured.
  It is not a generic likelihood interface for every parameterization.
- `Prior(transform="sigmoid", ...)` creates a transformed deterministic from a
  raw draw. This is a modeling transformation, not the sampler's unconstraining
  transform. It changes the induced prior; inspect it on the target scale.
- JSON serialization stores distributions and parameter configuration, not a
  compiled graph, fitted optimizer or all custom transform implementations.
  Re-register custom transforms/deserializers in the loading process. Do not
  assume arbitrary symbolic expressions can be serialized.
  In 0.15.1 `to_dict`/`from_dict` omit `core_dims`; store and restore that
  metadata separately for experimental multivariate `pymc.dims` factories.

`Censored(Prior(...), lower=..., upper=...)` represents clipping with boundary
point masses; it is not truncation or a model for values absent through selection.
It requires a centered, untransformed base Prior in 0.15.1. `Scaled(Prior(...),
factor=...)` multiplies a draw and registers the result as a deterministic;
scale likelihoods and other terms consistently if it is only a units change.

```python
from pymc_extras.prior import Censored, Scaled, sample_prior

censored = Censored(Prior("Normal", mu=0, sigma=1), lower=0)
scaled = Scaled(Prior("Normal", mu=0, sigma=1), factor=10)
censored_draws = sample_prior(censored, draws=200, random_seed=42)
scaled_draws = sample_prior(scaled, draws=200, random_seed=43)
```

`sample_prior` returns an **xarray Dataset**, not a DataTree. Provide coords for
factory dims; `wrap=True` records a custom factory's returned expression when it
is not already a registered variable. Factory draws alone do not test a model's
prior-predictive outcomes. For elicitation and `Prior.constrain`/PreliZ use
[prior elicitation](../../prior-elicitation/SKILL.md).

## as_model is a model factory, not a fitting decorator

The decorated function constructs variables; calling it returns a `pm.Model`,
not its Python return value. Tensor-convertible arguments become `pm.Data`,
including scalar arguments and defaults. Keep Python control-flow configuration
outside those arguments, and reserve `coords` for the factory's coordinate input.
Pass coords at each invocation: the 0.15.1 implementation consumes decorator
coords on its first call rather than retaining them for subsequent calls.

```python
import pymc_extras as pmx

@pmx.as_model()
def regression(x, outcome):
    intercept = pm.Normal("intercept", 0, 1)
    slope = pm.Normal("slope", 0, 1)
    pm.Normal("y", intercept + slope * x, 0.5, observed=outcome, dims="obs")

factory_model = regression(
    np.array([-1.0, 0.0, 1.0]), np.array([-0.6, 0.1, 0.8]),
    coords={"obs": ["a", "b", "c"]},
)
assert np.isfinite(factory_model.compile_logp()(factory_model.initial_point()))
```

Use `pm.set_data(..., coords=...)` for new predictors/observation lengths under
that model. Preserve predictor meaning and registration names; do not treat
automatic data wrapping as identity alignment or data validation.

## ModelBuilder requires a real model lifecycle

`from pymc_extras.model_builder import ModelBuilder` provides fit, posterior
prediction and NetCDF persistence. It is a base class, not a ready-made regression.
Implement `build_model`, `_generate_and_preprocess_model_data`, `_data_setter`,
`output_var`, both default-config methods and `_serializable_model_config`.
Use it only when consumers need this interface; do not wrap every model in a class.
Grouped NetCDF persistence also needs a backend such as `h5netcdf[h5py]` or
`netCDF4`; it is not supplied by every extras installation.

A complete single-predictor example retains labeled inputs, supports changed
prediction lengths and reconstructs the same model on load:

```python
import pandas as pd
from pymc_extras.model_builder import ModelBuilder

class GaussianRegression(ModelBuilder):
    _model_type = "GaussianRegression"
    version = "1"

    @property
    def output_var(self):
        return "y"

    @staticmethod
    def get_default_model_config():
        return {"coefficient_sd": 1.0, "noise_sd": 0.5}

    @staticmethod
    def get_default_sampler_config():
        return {"chains": 2, "cores": 1, "draws": 100, "tune": 100,
                "nuts_sampler": "pymc"}

    @property
    def _serializable_model_config(self):
        return self.model_config

    def _generate_and_preprocess_model_data(self, X, y):
        if list(X.columns) != ["x"] or not X.index.equals(pd.RangeIndex(len(X))):
            raise ValueError("Expected one x column and a zero-based RangeIndex")
        values = X["x"].to_numpy(dtype=float)
        outcome = np.asarray(y, dtype=float)
        if outcome.shape != (len(X),) or not np.isfinite(values).all() or not np.isfinite(outcome).all():
            raise ValueError("Training inputs and outcomes must be aligned and finite")
        self.X, self.y = X.copy(), outcome

    def build_model(self, X, y, **kwargs):
        self._generate_and_preprocess_model_data(X, y)
        with pm.Model(coords={"obs": np.arange(len(X))}) as self.model:
            x = pm.Data("x_data", self.X["x"].to_numpy(), dims="obs")
            outcome = pm.Data("y_data", self.y, dims="obs")
            intercept = pm.Normal("intercept", 0, self.model_config["coefficient_sd"])
            slope = pm.Normal("slope", 0, self.model_config["coefficient_sd"])
            pm.Normal(self.output_var, intercept + slope * x,
                      self.model_config["noise_sd"], observed=outcome, dims="obs")

    def _data_setter(self, X, y=None):
        if list(X.columns) != ["x"]:
            raise ValueError("Expected the training predictor x")
        values = X["x"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Prediction inputs must be finite")
        outcome = np.zeros(len(X)) if y is None else np.asarray(y, dtype=float)
        if outcome.shape != (len(X),) or not np.isfinite(outcome).all():
            raise ValueError("Outcomes must match prediction rows")
        with self.model:
            pm.set_data({"x_data": values, "y_data": outcome},
                        coords={"obs": np.arange(len(X))})

estimator = GaussianRegression()
fitted = estimator.fit(
    pd.DataFrame({"x": [-1.0, 0.0, 1.0]}), np.array([-0.6, 0.1, 0.8]),
    progressbar=False, random_seed=42,
)
predictive = estimator.predict_posterior(
    pd.DataFrame({"x": [0.5, 1.5]}), extend_idata=False, combined=False,
    random_seed=43, progressbar=False,
)
estimator.save("regression.nc")
loaded = GaussianRegression.load("regression.nc")
```

This fit is an execution example, not evidence of reliable posterior inference.
Increase the budget and diagnose before reporting estimates. Prediction dummies
supply the observed container's shape; no fitting takes place against them.
`predict` averages outcome draws, whereas `predict_posterior` retains uncertainty.
`predict_proba` is an alias for predictive draws, not necessarily class
probabilities. Extract latent means explicitly when noisy outcomes are not wanted.

In 0.15.1, fit persistence combines X with a newly constructed RangeIndex outcome
frame. Align training rows before `fit`; do not let Pandas concatenate different
indices. Prediction mutates model data even with `extend_idata=False`; restore
training data before training PPC/log-likelihood computations. `extend_idata=True`
can replace predictive groups; save distinct forecast scenarios separately.
`save` includes training `fit_data` and configurations: consider data authorization
and storage exposure. `load` requires the same subclass/code/configuration and
rebuilds the model; NetCDF is not a portable serialization of arbitrary Python.
If preprocessing learns transformations, store and restore them and never refit
preprocessing on prediction rows.

## VIP chooses geometry, not a different scientific prior

`vip_reparametrize` returns a new model and a helper. In 0.15.1 dispatch supports
Normal and Exponential RVs, not arbitrary distributions; transformed Normals are
unsupported. For Normal effects lambda=1 is centered, lambda=0 non-centered.

```python
from pymc_extras.model.transforms.autoreparam import vip_reparametrize

with pm.Model(coords={"group": ["a", "b", "c"]}) as centered:
    mu = pm.Normal("mu", 0, 1)
    scale = pm.HalfNormal("scale", 0.5)
    effect = pm.Normal("effect", mu, scale, dims="group")
    pm.Normal("y", effect, 0.3, observed=np.array([0.1, -0.2, 0.4]), dims="group")

reparameterized, vip = vip_reparametrize(centered, ["effect"])
with reparameterized:
    vip.set_all_lambda(0.5)
    exploratory_vi = vip.fit(n=500, random_seed=42, progressbar=False)
learned_lambda = vip.get_lambda()
# Freeze these shared parameterization values before starting posterior sampling.
with reparameterized:
    posterior = pm.sample(draws=100, tune=100, chains=2, cores=1,
                          nuts_sampler="pymc", random_seed=43, progressbar=False)
```

VI here learns parameterization; it does not validate the final posterior. Save
lambda and compare geometry/ESS per unit cost with manual non-centering. Changing
lambda, including truncating it with `truncate_all_lambda`, requires a fresh fit:
old sampler adaptation and auxiliary-variable draws use the old coordinates.
Inspect original effects, not only auxiliary variables. Unsupported graphs need
an explicit manual parameterization, not a silently changed model.

## Posterior-as-prior transfer is approximate updating

`from pymc_extras.utils.prior import prior_from_idata` constructs a joint
MvNormal approximation from selected posterior variables in the active model.
For example, `prior_from_idata(previous, var_names=["intercept", "slope"])`
returns a name-to-variable dictionary. Positive scales need a log transform;
simplexes need an appropriate transform and matching coords. Name/dims overrides
are supported. Inspect installed help before configuring constrained variables.

Retaining joint covariance is better than independent marginal fits, but the
Gaussian can miss multimodality, tails and nonlinear dependence. Validate its
prior predictions against the original joint draws. Use only new, nonoverlapping
data in the next likelihood; reusing original observations double-counts evidence.
Changes to parameter meaning, units, covariate preprocessing or selection require
a new modeling argument, not just matching variable names.

## Inspection utilities expose graphs, not scientific equivalence

`from pymc_extras.printing import model_table` provides a Rich model summary.
`from pymc_extras.utils.model_equivalence import equivalent_models` compares model
graphs, optionally with canonicalization. These help inspect refactors; graph
comparison is not proof of equal predictive behavior under arbitrary
reparameterizations, nor of causal identification. Check numeric log densities,
transformed quantities and predictions on a small case too.

## Primary sources

- [Prior/factory source, 0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/prior.py).
- [Model factory source, 0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/model/model_api.py).
- [ModelBuilder source, 0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/model_builder.py).
- [VIP source, 0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/model/transforms/autoreparam.py).
- [Posterior-prior source, 0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/utils/prior.py).
- [Model equivalence source, 0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/utils/model_equivalence.py).
