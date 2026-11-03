# Bayesian state-space models with pymc-extras

Use the [canonical compatibility guide](pymc-extras.md#resolve-compatibility-before-choosing-an-api) for the checked environment and dependency bounds.
Older version cautions retain their stated scope.
The official stable documentation can change independently.

The examples are small runnable workflows, not evidence that all model families or numerical options work correctly.

## Choose the model, not just an available constructor

The equations below describe the common linear Gaussian representation:
`x[t+1] = c[t] + T[t] @ x[t] + R[t] @ eta[t]`,
`y[t] = d[t] + Z[t] @ x[t] + epsilon[t]`, with innovation covariance `Q[t]`
and measurement covariance `H[t]`.
The initial state has mean `x0` and covariance `P0`.

Kalman filtering integrates the latent state sequence for these models.
`BayesianETS` also supports a direct innovations likelihood under the conditions below.
Parameter inference still uses PyMC priors and the selected likelihood.

This is not a generic nonlinear or non-Gaussian interface.

| Scientific structure | Release class | Important choices |
|---|---|---|
| One series with lagged dynamics, innovations and optional seasonality/differencing | `BayesianSARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,S))` | Specify stationarity, MA invertibility/identification, initialization, calendar period and exogenous regression. `trend="c"` is a constant in the differenced ARMA equation, hence a drift when differencing. |
| Several series whose histories jointly predict each other | `BayesianVARMAX(order=(p,q), endog_names=[...])` | Regularize cross-series coefficients and innovation covariance. A VARMA is not automatically identified by choosing p and q. Exogenous inputs enter the transition by default; `exog_in_observation=True` instead gives regression with VARMA errors. These coefficients have different interpretations. |
| Exponential smoothing with additive Gaussian errors | `BayesianETS(order=("A","N","N"), endog_names=["y"])`, with additive/damped trend and additive seasonality options | This is the additive-error subset, not arbitrary multiplicative ETS. With complete data, no measurement error and `stationary_initialization=True`, the likelihood uses an innovations recursion. Other cases use Kalman filtering. Match smoothing priors to `use_transformed_parameterization`. |
| Many observed series driven by fewer dynamic latent factors | `BayesianDynamicFactor` | Choose factor number/order and idiosyncratic error structure. Standardized factor innovations do not resolve every loading sign/rotation ambiguity; impose scientifically defensible identification before interpreting individual factors. Optional regression coefficients can evolve through `exog_innovations`. |
| Interpretable level, trend, seasonal, cyclic and regression contributions | `structural` components composed with `+`, then `.build()` | Specify which components vary, their phase and shared versus series-specific states, and distinguish process shocks from measurement noise. |

See [model release exports](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/__init__.py),
[SARIMAX source](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/SARIMAX.py),
[VARMAX source](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/VARMAX.py),
[`BayesianETS` source, 0.16.0](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/ETS.py),
and [DynamicFactor source](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/DFM.py).

### ETS initialization in 0.16.0

The constructor defaults to `stationary_initialization=True` and `measurement_error=False`.
The default fixes `P0=R @ Q @ R.T` and needs no distribution for `P0`.
This is the single-source-of-error prediction covariance, not a stationary distribution for the level or trend.

Declare distributions for the initial level, trend and seasonal values from `param_info`.
With `stationary_initialization=False`, declare `P0`.
The constructor no longer accepts `initialization_dampening`.

The fixed covariance is singular.
Use the default `mvn_method="svd"` for conditional sampling.
Do not select `"cholesky"` or `"eigh"` for that singular covariance.

See [ETS innovations change, PR 777](https://github.com/pymc-devs/pymc-extras/pull/777).

The default `use_transformed_parameterization=False` uses `alpha`, `beta` and `gamma` between zero and one.
In that form, the trend gain is `alpha*beta` and the seasonal gain is `(1-alpha)*gamma`.

With `use_transformed_parameterization=True`, use `0<alpha<1`, `0<beta<alpha` and `0<gamma<1-alpha`.
Only declare coefficients for the selected trend and seasonality.

## Register priors and build the likelihood

Construct the state-space object first. Inspect `ss.param_names`, `ss.param_info`,
`ss.param_dims`, `ss.coords`, `ss.observed_states` and, where relevant,
`ss.data_names`/`ss.data_info`. Component names affect parameter names; copy the
actual metadata instead of adapting a stale example's names. Use
`pm.Model(coords=ss.coords)` and register a variable with each required name and
shape. A named `pm.Deterministic` may supply a constrained transformation or a
fixed covariance. Use `ss.param_dims.get(name)`; scalar parameter names can be
absent from this dictionary. Do not unpack dictionary values by positional order.
Constructor constraint descriptions are guidance, not automatically enforced prior support.

Call `ss.build_statespace_graph(train)` inside that model after declaring priors
and required `pm.Data` inputs. It registers the training observation container
`data` and likelihood `obs`; do not add a second independent likelihood for the
same observations or reserve these names for unrelated variables. Then use
ordinary `pm.sample` and the [sampling workflow](sampling.md). Post-estimation
methods recover fit coordinates, inputs and parameter dimensions from the saved
DataTree, so preserve `observed_data` and `constant_data` as well as `posterior`.
Do not save only an unlabeled parameter array.

For stationary SARIMAX initialization, the priors must produce stable dynamics.
For AR(1), a transformed coefficient in (-1,1) suffices; separately bounding every
AR(p) coefficient does not. Check roots or the companion matrix's spectral
radius. Nonstationary initialization does not make an explosive process sensible.
With `d>0` or `D>0`, [0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/statespace/models/SARIMAX.py) needs `stationary_initialization=False` and the `"fast"` representation for internal differencing.
Register `x0` and `P0`.
Manual differencing needs saved ordered level/seasonal history and draw-wise
inversion. Stable transitions with an arbitrary `P0` are not stationary
initialization. Initial covariance is in squared outcome/state units, not SD.

For structural models, `.build()` adds the required `P0` parameter. Component
initial parameters give initial-state means; `P0` is conditional covariance
around those means. A prior on the initial mean and nonzero `P0` represent two
sources of uncertainty. Specify both deliberately rather than using a huge
covariance as an unexplained numerical convenience.

Sources: [base interface and graph construction](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/core/statespace.py),
[structural assembly](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/structural/core.py),
[official SARIMAX API](https://www.pymc.io/projects/extras/en/stable/statespace/generated/pymc_extras.statespace.models.BayesianSARIMAX.html).

## Complete local-level example: fit, simulate and forecast

Here `y[t] = level[t] + measurement_error[t]` and
`level[t+1] = level[t] + process_shock[t]`. It is deliberately nonstationary.
The synthetic data are in modest, known units. Replace these scales with
scientifically elicited priors for real data, not scales estimated from a future
holdout. The short two-chain fit is an API/exploration example, **not** sufficient
for reportable inference; assess divergences, R-hat, ESS and MCSE before relying
on parameter or forecast summaries.

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc_extras.statespace import structural as st

rng = np.random.default_rng(2026)
n, horizon = 40, 5
index = pd.date_range("2025-01-01", periods=n, freq="D")
level = np.r_[0.0, np.cumsum(rng.normal(0.0, 0.15, n - 1))]
y = level + rng.normal(0.0, 0.25, n)

ss = (
    st.LevelTrend(order=1, innovations_order=1, name="level")
    + st.MeasurementError(name="measurement")
).build(verbose=False)
train = pd.DataFrame(y[:, None], index=index, columns=list(ss.observed_states))
assert train.index.is_unique and train.index.is_monotonic_increasing
assert train.shape == (n, ss.k_endog)
assert np.isfinite(train.to_numpy()).all()

with pm.Model(coords=ss.coords) as model:
    pm.Normal("initial_level", mu=0.0, sigma=1.0,
              dims=ss.param_dims["initial_level"])
    pm.HalfNormal("sigma_level", sigma=0.3,
                  dims=ss.param_dims["sigma_level"])
    pm.HalfNormal("sigma_measurement", sigma=0.5,
                  dims=ss.param_dims.get("sigma_measurement"))
    pm.Deterministic("P0", pt.eye(ss.k_states) * 0.25**2,
                     dims=ss.param_dims["P0"])
    ss.build_statespace_graph(train)
    # Draw parameters only; use the unconditional simulator for prior paths.
    prior = pm.sample_prior_predictive(
        draws=20, var_names=list(ss.param_names), random_seed=2027
    )

prior_paths = ss.sample_unconditional_prior(
    prior, use_data_time_dim=True, random_seed=2028, progressbar=False
)
assert prior_paths["prior_observed"].sizes["time"] == n
assert np.isfinite(prior_paths["prior_observed"].values).all()

with model:
    idata = pm.sample(
        draws=100, tune=100, chains=2, cores=1,
        target_accept=0.9, random_seed=2029, progressbar=False
    )

# Retrospective state uncertainty conditional on the training observations.
conditional = ss.sample_conditional_posterior(
    idata, random_seed=2030, progressbar=False
)
smoothed = conditional["smoothed_posterior"]
assert smoothed.sizes["time"] == n
assert smoothed.sizes["state"] == ss.k_states
assert np.isfinite(smoothed.values).all()

# Replications from the initial-state law, not continuations from the endpoint.
replications = ss.sample_unconditional_posterior(
    idata, use_data_time_dim=True, random_seed=2031, progressbar=False
)
assert replications["posterior_observed"].sizes["time"] == n
assert np.isfinite(replications["posterior_observed"].values).all()

# Forecast from the final training filtered distribution, with new innovations.
future = ss.forecast(
    idata, start=train.index[-1], periods=horizon,
    filter_output="filtered", random_seed=2032, progressbar=False
)
future_y = future["forecast_observed"]
future_x = future["forecast_latent"]
expected_index = pd.date_range(index[-1], periods=horizon + 1, freq="D")[1:]
assert future_y.dims == ("chain", "draw", "time", "observed_state")
assert future_x.dims == ("chain", "draw", "time", "state")
assert future_y.shape == (2, 100, horizon, 1)
assert future_x.shape == (2, 100, horizon, ss.k_states)
assert np.array_equal(future_y["time"].values, expected_index.values)
assert np.isfinite(future_y.values).all() and np.isfinite(future_x.values).all()
forecast_mean = future_y.mean(dim=("chain", "draw"))
forecast_eti = future_y.quantile([0.055, 0.945], dim=("chain", "draw"))
```

`future`, `conditional`, `prior_paths` and `replications` are the returned
predictive **child** DataTrees in 0.16.0. Access their variables directly, as
above, not `future["posterior_predictive"]["forecast_observed"]`. The fit itself
is a root DataTree: parameters are in `idata["posterior"]`.
Unconditional `steps=k` produces `k+1` time positions including the initial
state; `use_data_time_dim=True` matches the training span. Forecast
`periods=horizon` instead produces exactly `horizon` future positions and
excludes the conditioning origin. `forecast_latent` is the full state vector,
not necessarily a same-shaped noiseless observation; use the design matrix to
map a multicomponent state to the observation scale.

The assertions check identity, shape and finiteness, not forecast calibration
or Monte Carlo reliability. The ETI is a pointwise 89% equal-tail interval, not
a simultaneous path band. Observation forecasts include new measurement noise;
latent state forecasts do not.

Sources: [LevelTrend parameters](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/structural/components/level_trend.py),
[MeasurementError parameters](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/structural/components/measurement_error.py),
[forward sampling and return groups, 0.16.0](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/core/forward_sampling.py),
[forecast indexing and simulation, 0.16.0](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/core/forecast.py).

## Construct richer structural models

Use current names from `pymc_extras.statespace.structural`, not deprecated
`*Component` aliases. Add components, then inspect the resulting object's
requirements before creating priors. Use explicit unique `name=` values and
consistent `observed_state_names` across components.

- `LevelTrend(order=1, innovations_order=1)` is a random-walk level;
  `order=2, innovations_order=2` is a local linear trend;
  `order=2, innovations_order=[0,1]` gives a smooth trend with stochastic slope
  but no direct level shock. `innovations_order=0` makes the selected derivatives
  deterministic conditional on their initial states.
- `TimeSeasonality(season_length=7, name="weekly", innovations=False)` adds a
  fixed seven-position seasonal pattern. Set `start_state` to the first
  observation's phase. The default `remove_first_state=True` imposes the
  sum-to-zero identification by deriving one effect; with `False`, enforce that
  constraint yourself. `duration` counts equally spaced observations per seasonal
  position; a fixed duration does not encode unequal calendar-month lengths.
- `FrequencySeasonality` uses harmonic seasonal states; choose enough harmonics
  to represent the scientific pattern without duplicating a trend or cycle.
- `Cycle` describes an oscillation; its period and damping need distinct prior
  justification from deterministic calendar seasonality.
- `Autoregressive` adds residual lag dependence; stationarity still requires a
  valid joint coefficient parameterization.
- `Regression(name="drivers", state_names=["price", "weather"],
  innovations=False)` adds constant coefficients; `innovations=True` lets them
  follow random walks and requires priors on `sigma_beta_drivers`. The required
  input container is `data_drivers`, not the observed `data`; coefficient means
  are `beta_drivers`. A stochastic coefficient process requires horizon-specific
  prior-predictive checks and may be weakly identified against a flexible trend.
- `MeasurementError(name="measurement")` contributes observation noise but no
  hidden state. Omitting it conditions on noiseless measurements under that model;
  do not omit it solely to make fitting easier.

For example, `(st.LevelTrend(order=2, name="trend") +
st.TimeSeasonality(season_length=7, name="weekly", innovations=False) +
st.MeasurementError(name="measurement")).build()` constructs the system, but
must be followed by priors for **all** names in its own `param_info`, including
`P0`. Do not reuse the local-level example's prior names after changing components.
For a fitted structural model, select only the desired latent output before
extracting contributions, for example
`ss.extract_components_from_idata(conditional["smoothed_posterior"].to_dataset(name="smoothed_posterior"))`.
This avoids mistaking same-width observation variables for state variables. Choose
filtering or smoothing according to the scientific question before extraction.

Sources: [official structural component documentation](https://www.pymc.io/projects/extras/en/stable/statespace/models/structural.html),
[release component exports](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/structural/__init__.py),
[seasonality implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/structural/components/seasonality.py),
[regression implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/models/structural/components/regression.py).

## Conditioning: filtering, smoothing, replication and forecasting

- **Predicted states** at t use earlier observations but not y[t]; **filtered
  states** incorporate y[t]; **smoothed states** also use later training
  observations. Filter means/covariances are conditional moments for each
  parameter draw, not draws of the state itself.
- `sample_conditional_prior(prior)` uses prior parameter draws but conditions
  states on observed training outcomes. It is not an outcome-free prior
  predictive check. Use `sample_unconditional_prior(prior)` for that purpose.
- `sample_conditional_posterior(idata)` returns `filtered_posterior`,
  `predicted_posterior`, `smoothed_posterior` and corresponding `_observed`
  variables. Observed outputs are noisy replications conditional on the state
  information, not recovered missing measurements known with certainty.
- In [0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/statespace/core/forward_sampling.py), `joint_smoothed_draws=True` (default) uses a simulation smoother that preserves cross-time posterior state covariance.
  With `False`, smoothed draws
  have correct per-time marginal distributions but not joint path dependence.
  Filtered/predicted state draws are likewise per-time marginals, not joint paths.
  Do not use marginal draws to assess path sums or simultaneous excursions.
- `sample_unconditional_posterior(idata)` uses fitted parameters but restarts at
  the initial-state law. It checks model replications, not endpoint-conditioned
  future predictions. `forecast` propagates a chosen origin distribution with
  fresh process and observation innovations, retaining parameter uncertainty.
- Pass `filter_output="filtered"` explicitly for forecasting. The release default
  is `"smoothed"`. At the final training endpoint the two condition on the same
  observations; at an earlier origin smoothing leaks later training outcomes.
  Even a filtered earlier state from a full-series parameter fit leaks through
  the parameter posterior. Holdout forecasts need prefix-only preprocessing,
  priors chosen without holdout outcomes and prefix-only fitting/refitting.

For a numerical oracle, compare a small independent Kalman recursion for Kalman-based models.
Match initialization and jitter.
For the ETS innovations path, compare forecast errors and summed Normal innovation log densities with an independent ETS recursion.

For a zero-mean latent AR(1) with terminal filter `(mT,VT)`, conditional on
parameters, horizon-h observation mean and variance are
`phi**h*mT` and
`phi**(2*h)*VT + sigma_state**2*sum(phi**(2*j) for j in range(h)) + sigma_obs**2`.
Check temporal covariance and independence of initial-state and innovation draws,
not only marginal variances. A healthy parameter sampler alone does not establish
correct simulation, seasonal phase or forecast alignment.

## Missing observations and future exogenous scenarios

Keep missing measurements as NaNs at their original clock positions.
Kalman likelihoods mask missing measurements and propagate state uncertainty through gaps.
ETS with missing data also uses the Kalman path.

These likelihoods do not add ordinary PyMC imputation variables.
Validate unexpected infinities yourself:
data preprocessing masks nonfinite values, which must not disguise a corrupt
measurement as routine missingness. Declare assumptions about informative
missingness; a missing-data mask does not model the selection process. Use a
regular, ordered, unique time index and explicit DataFrame columns in
`ss.observed_states` order. Dimensions label positions; they do not join rows.
A DateTime forecast requires a regular frequency. Do not drop dates or rely on a
generated range index to establish scientific clock alignment. Panel MultiIndex
input is not supported by this data interface.

Register required training exogenous containers before building the graph. For
SARIMAX with `exog_state_names=["price","weather"]`, use a `pm.Data` named
`exogenous_data` with dimensions `ss.data_info["exogenous_data"]["dims"]`,
training-time rows and those two columns, plus a prior named `beta_exog`.
For the structural Regression above, use `data_drivers` and `beta_drivers`
with their own metadata dimensions. If exogenous data introduce the `time`
coordinate first, supply the exact same training index later to graph construction.

Custom exogenous models in 0.16.0:
A `PyMCStateSpace` subclass can declare `Data(..., is_exogenous=True)` through `set_data_info()`.
The base class now derives its exogenous-data flag from that metadata.
[PR 794](https://github.com/pymc-devs/pymc-extras/pull/794) fixes the missing flag for such custom subclasses in 0.15.1.

This activates the forecast scenario path.
It does not establish scenario dates, column order or scientific alignment.
See the [0.16.0 base implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/core/statespace.py).

Forecasts of models fitted with exogenous inputs require scenarios for **every**
required input. With multiple containers, pass a dictionary keyed by
`ss.data_names`; using that dictionary also makes single-container intent clear:
`scenario={"exogenous_data": future_drivers}`. Construct `future_drivers` with
exact forecast dates, training column labels/order and H rows, then pass it to
`ss.forecast(idata, start=origin, periods=H, filter_output="filtered", scenario=scenario)`.
Assert dates and column order before the call.
Forecast preparation can replace the supplied scenario index with the generated forecast index, including in 0.16.0.
This does not prove that the original dates matched.

Keep these alignment checks after the exogenous-data flag fix.
`use_scenario_index=True`
uses the scenario's regular index instead; it ignores explicit start/end/periods
and takes the preceding fitted date as the conditioning origin. Verify that date
and frequency yourself. Greater-than-two-dimensional exogenous inputs are not
supported by forecast scenario validation.

A fixed scenario conditions on its supplied path. Forecast intervals do not
include uncertainty in future weather, prices or interventions unless you model
or integrate over that uncertainty. Scenario contrasts are not automatically
causal effects. Select scenarios known or credibly modeled at the forecast
origin; never insert realized future outcomes or holdout-derived regressors.
Time-varying seasonal/trend/exogenous models need explicit phase and scenario
checks, particularly when requesting origins before the final fitted date.

Sources: [data preparation and missing masking](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/utils/data_tools.py),
[scenario validation and forecast construction, 0.16.0](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/core/forecast.py).

## Numerical choices without changing the scientific model

The release implementation accepts `filter_type="standard"`, `"univariate"`,
`"cholesky"` or `"convergent"`; some docstrings still list older labels such as
`"single"` and `"steady_state"`, which are not factory keys in 0.15.1. Start
with `"standard"`; compare an alternative's covariance behavior on the actual
system rather than assuming universal speedups. `smoother_type="disturbance"`
is the default; `"rts"` provides an alternative recursion for cross-checks.
The convergent filter requires time-invariant matrices and rejects missing
observations; do not select it merely because a series is long.
`mvn_method="svd"` is the auxiliary simulation default; `"cholesky"` is faster
when covariances are well-conditioned positive definite but is less forgiving
of singular state covariances. Degenerate latent components need special care.

For Kalman filtering, constructor `cov_jitter` affects fitting and post-estimation graphs.
Its default is 1e-8 (1e-6 for float32).
Record it and check sensitivity, since it adds
to covariance diagonals and is not a scientifically specified measurement
variance. Constructor `missing_fill_value` defaults to -9999.0; change it if this
is a legitimate measurement, not by modifying the observed data to avoid errors.
The constructor's `mode` configures auxiliary sampling/forecast compilation,
not `pm.sample`; sampling methods can override it through `compile_kwargs`.
Keep sampler/backend compatibility distinct from these graph compilation options.

`ss.make_filter_outputs(names="filtered_states")` inside the built model context
returns symbolic conditional moments to derive a deterministic quantity;
`ss.sample_filter_outputs(idata, filter_output_names=["filtered_states",
"filtered_covariances"])` evaluates moments across saved parameter draws and
returns a **root** DataTree with a `posterior_predictive` child. This differs from
the direct predictive child returned by the trajectory and forecast methods.
`ss.unpack_statespace()` returns parameter-bound `x0,P0,c,d,T,Z,R,H,Q` inside the
model context, useful for stability or covariance checks. These are inspection
interfaces, not a substitute for predictive criticism or temporal holdouts.

Sources: [filter factory, jitter, compilation and inspection methods](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/core/statespace.py),
[filter-output sampling](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/core/forward_sampling.py),
[simulation implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/statespace/filters/distributions.py),
[older filter-label caution, 0.15.1](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/statespace/core/statespace.py).
