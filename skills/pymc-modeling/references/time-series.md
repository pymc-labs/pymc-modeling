# Time series, differential equations and state-space forecasting

## Define time and information

Preserve `(series,time)` identity and physical clock units. Validate ordered unique
times, component labels and nonoverlapping future coordinates. Choose forecast
origins before fitting; only the available prefix informs preprocessing, priors,
parameters and filtered states. Smoothed states use later observations; choosing
an earlier filtered state from a full-series fit still leaks through parameters.
Use rolling/expanding-origin refits or a justified contiguous future block, not
shuffled CV or ordinary rowwise LOO as substitutes for multi-step forecasting.

Declare initial-state distribution versus fixed conditioning and observed process
versus noisy measurement of latent state. For noisy observations, forecast from
a filtered state distribution, not the last measurement as an exact state.
Future weather/prices need known-at-origin scenarios or their own uncertainty;
future outcomes never become convenient predictors.

Differencing requires saved last levels, and seasonal differencing requires lagged
seasonal history with its calendar map. Invert forecasts draw-wise by cumulative
sums to retain temporal covariance. Missing observations retain clock positions;
dropping them and treating survivors as unit-spaced changes the model.

## Autoregression

A stationary AR(1) has |rho|<1 and initial law
`y0 ~ Normal(0, sigma/sqrt(1-rho²))` for zero intercept. Stable dynamics with an
arbitrary initial distribution is not stationary initialization. For AR(p),
coefficient-wise bounds are insufficient: inspect roots/use a stationary
parameterization and specify the **joint** initial p-state law.

```python
# rho and sigma have scientifically chosen priors inside the active model.
pm.AR("response", rho=pt.stack([rho]), sigma=sigma,
      constant=False, ar_order=1,
      init_dist=pm.Normal.dist(0, sigma/pt.sqrt(1-rho**2)),
      observed=train_y, dims="time")
```

`constant=True` makes the first rho coefficient an intercept:
`[intercept,phi]` is AR(1), `[phi]` alone is not. `steps` counts innovations;
total length is `steps+ar_order`. Explicit ar_order avoids ambiguous symbolic shape.

For an observed zero-intercept AR(1), conditional forecast mean/variance at horizon
h are `rho**h*yT` and `sigma²*sum(rho**(2*j),j=0..h-1)`. Generate fresh innovations
for each posterior draw. A fixed-origin continuation contains the origin plus H
future points; exclude the origin from forecast scoring. For p lags retain the
needed ordered history, not one terminal value.

## Random walks

| Interface | Increment contract |
|---|---|
| RandomWalk | Independent unnamed initial and innovation distributions of equal support dimension. |
| GaussianRandomWalk | Normal increments, mu is drift per step. |
| MvGaussianRandomWalk | Vector increments with cov/tau/chol for within-step dependence. |
| MvStudentTRandomWalk | Vector t increments; covariance is nu/(nu-2)*scale only for nu>2. |

Steps counts increments, yielding steps+1 time points before event axes. Unnamed
distributions are cloned; this does not share a registered latent initial draw.
Build explicit conditional graphs when initial/innovation dependence is needed.
For exact vector conditioning, add cumulative vector innovations to the known
terminal vector, not tiny artificial initial covariance.

With known innovation SD and uncertain drift, the N-1 training differences give
a conjugate Normal reference. Conditional forecast mean is last+h*drift and
variance h*sigma²; integrating drift adds h² Var(drift). Shared uncertain drift
can produce implausibly wide long paths even with small one-step scales. Check
prior trajectories over the actual horizon, not just increments. Preserve
cross-component and cross-time covariance in multivariate predictions.

## GARCH

For GARCH(1,1), `v_t=omega+alpha*y_(t-1)²+beta*v_(t-1)` and
`y_t ~ Normal(0,sqrt(v_t))`. `initial_vol` is SD, omega is variance-recursion
intercept, not unconditional variance. Positive coefficients and alpha+beta<1
are finite-variance modeling constraints; do not assume constructors enforce all
of them. Long-run variance is omega/(1-alpha-beta). Fixing initial volatility
conditions the process rather than drawing from its stationary law.

Reconstruct terminal variance from training returns. Forecast each posterior
parameter draw with fresh returns/innovations, never realized future squared
returns. Expected first variance is omega+alpha*yT²+beta*vT; later expectations
follow omega+(alpha+beta)*previous expectation. These moment formulas are checks,
not replacements for random volatility paths in observation simulation.

## SDEs and ODEs

Euler–Maruyama represents a discretized stochastic process. Its callback takes
state plus `sde_pars` and returns drift/diffusion with compatible batch/time shape.
For OU drift -rate*x and diffusion s, the Euler transition mean/SD are
`(1-rate*dt)*x` and `s*sqrt(dt)`. The scalar stability range is 0<rate*dt<2, not a
universal step-size rule. Exact OU uses exp(-rate*dt) and variance
`s²*(1-exp(-2*rate*dt))/(2*rate)`. Compare common physical horizons as dt decreases.
Accurate inference for the discretized model is not inference for the exact SDE;
discretization, measurement noise and Monte Carlo error are different uncertainties.

`pm.ode.DifferentialEquation` describes deterministic dynamics with an explicitly
separate observation likelihood. Its callback is `(y,t,theta)` with n_states
outputs. `return_sens=True` exposes `(n_times,n_states,n_states+n_theta)`
sensitivities; n_p includes initial states and dynamic parameters. Check analytic
small systems, parameter and initial-state derivatives, and observation noise.
A noisy last observation is not an exact ODE restart; integrate from the known
initial condition or an inferred state distribution.

Op shape/type/gradient and aliasing fields are compiler contracts, not alternative
statistical models. Do not mutate metadata to evade mismatches. Smooth scalar
success does not establish stiff-system, event/discontinuity, high-dimensional
sensitivity or higher-derivative accuracy. Preserve solver warnings and inspect
PyTensor derivative-protocol compatibility in the installed release.

## State-space models

pymc-extras provides BayesianSARIMAX, VARMAX, ETS, DynamicFactor and composable
level/trend/seasonal/cycle/regression models. Choose from the scientific target;
check identification, phase, initial state and measurement noise explicitly.
Extras 0.14.0 requires PyMC <6.3/PyTensor <3.3; respect dependency bounds.

For SARIMAX, construct priors with `ss.param_dims`, use the exact
`ss.observed_states` columns and a validated time-indexed training DataFrame, then:

1. `build_statespace_graph` on training data only.
2. Draw prior parameters and `sample_unconditional_prior` trajectories for prior
   plausibility. A conditional prior filter has already used outcomes.
3. Fit/save parameter posterior, diagnose it, and check a small independent Kalman
   recursion for filtered means/covariances with matched jitter/initial law.
4. `sample_conditional_posterior` yields retrospective filtered/predicted/smoothed
   states; `sample_unconditional_posterior` yields model replications. Neither is
   a future forecast.
5. `forecast(idata,start=origin,periods=H,filter_output="filtered")` requests the
   intended conditioning explicitly (the 0.14 default is smoothed). That version
   returns the predictive child directly: `forecast["forecast_observed"]`, not
   another posterior_predictive group. Check that H outputs exclude the origin.

For latent AR(1) terminal filter `(mT,VT)`, future observation mean/variance are
`phi**h*mT` and
`phi**(2*h)*VT + sigma_state²*sum(phi**(2*j)) + sigma_measurement²`.
Check both means and covariance, with initial-state and future innovations
independent under this model. Extras 0.14 simulation RNG dependencies deserve
inspection: sharing an unadvanced generator between initial draw and scan could
correlate them. Treat this as a version-specific numerical check, not permission
to modify the target variance. A valid parameter fit does not prove state-simulation
or forecast accuracy; broadcasting failures are separate from sampler health.

Differencing d/D>0 requires nonstationary initialization. In extras 0.14 the fast
representation supports internal integrated states while interpretable does not.
Align exogenous scenarios by clock/identity before forecasting; row counts or an
automatically generated index do not establish alignment. Seasonal, MA and
multivariate structures need their own forecast-state and scenario checks.

Sources: [temporal distributions](https://www.pymc.io/projects/docs/en/stable/api/distributions/timeseries.html),
[temporal source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/timeseries.py),
[ODE source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/ode/ode.py),
[SARIMAX](https://www.pymc.io/projects/extras/en/stable/statespace/generated/pymc_extras.statespace.models.BayesianSARIMAX.html),
[state-space implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.14.0/pymc_extras/statespace/core/statespace.py),
[state simulation](https://github.com/pymc-devs/pymc-extras/blob/v0.14.0/pymc_extras/statespace/filters/distributions.py).
