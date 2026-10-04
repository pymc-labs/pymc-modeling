# Approximate inference: measure error for the intended use

## Select the method and compatibility deliberately

| Method | API and main limitation |
|---|---|
| Mean-field ADVI | `pm.ADVI` or `pm.fit(method="advi")`: Gaussian in sampling coordinates with diagonal covariance. |
| Full-rank ADVI | `pm.FullRankADVI` or `pm.fit(method="fullrank_advi")`: dense Gaussian, not arbitrary skewness or multimodality. |
| Grouped VI | `pm.Group`, `pm.variational.Approximation`, `pm.variational.KLqp`: independent groups lose cross-group dependence even if each is full rank. |
| SVGD | Empirical particles, kernel/temperature and finite-particle sensitivity; not independent MCMC samples. |
| ASVGD, ImplicitGradient | Experimental generator/Stein methods; PyMC 6.3.1 source discourages routine use. Keep warnings visible. |
| MAP | Core `pm.find_MAP` returns a point; extras `pmx.find_MAP` returns a DataTree with optimizer diagnostics. Neither supplies posterior uncertainty. |
| Laplace | `pmx.fit(method="laplace", model=model, ...)`: local mode/Hessian approximation; coordinates and curvature accuracy matter. |
| DADVI | `pmx.fit(method="dadvi", model=model, ...)`: deterministic finite-sample mean-field objective, not exact inference. |
| Pathfinder | `pmx.fit(method="pathfinder", model=model, ...)`: quasi-Newton Gaussian paths plus selection/reweighting. |
| INLA | `fit_INLA(x, Q, model=model, ...)`: experimental latent-Gaussian elimination followed by MCMC over remaining variables, not a generic Laplace alias. |

These extras calls target **pymc-extras 0.15.1**, which requires Python >=3.12,
PyMC >=6.3,<6.4, PyTensor >=3.3,<3.4, ArviZ >=1.2,<2 and PreliZ >=0.27,<0.29.
Do not override dependency bounds. See [extras selection and installation](pymc-extras.md).
Core `pm.fit` accepts advi/fullrank_advi/svgd/asvgd or an Inference object, not
extras methods. The separate `pmx.fit` dispatcher accepts `"laplace"`,
`"pathfinder"`, `"dadvi"` and case-sensitive `"INLA"`; its kwargs go to the
selected function. Direct imports are available from `pymc_extras.inference`;
`fit_dadvi` and `fit_INLA` are not exported at the package root.

PyTensor-backed MAP, Laplace, DADVI and multi-path Pathfinder do not require
BlackJAX. `gradient_backend="jax"` on MAP/Laplace/DADVI needs a compatible
JAX/JAXlib installation and JAX-translatable model. The separate
`fit_blackjax_pathfinder` needs BlackJAX and its compatible JAX stack. The release
lists `blackjax>=0.12,<2` in its `dev` extra, not its `complete` extra.
CPU execution does not establish accelerator compatibility. Full-rank DADVI and
arbitrary flow/NumPyro integrations are not supplied by these interfaces.

```python
with model:
    inference = pm.ADVI()
    approximation = inference.fit(n=optimization_steps)
    approximate_draws = approximation.sample(draws=draws, random_seed=42)
approximate_draws.to_netcdf("approximation.nc")
```

Save fitted parameters and loss history before downstream drawing/plotting when
recovery matters. Parameter arrays alone are not a resumable optimizer checkpoint:
accumulators, ordering, transforms and RNG state can matter.

## Concrete extras fits and their outputs

This small Gaussian case has an analytic posterior: for the data below,
`theta | y ~ Normal(sum(y)/5, sqrt(1/5))`. Use that oracle to inspect location,
spread, interval endpoints and Monte Carlo error, not as evidence for a harder
model's accuracy.

```python
import numpy as np
import pymc as pm
import pymc_extras as pmx

y = np.array([-0.2, 0.1, 0.4, 0.3])
with pm.Model() as model:
    theta = pm.Normal("theta", 0.0, 1.0)
    pm.Normal("y", theta, 1.0, observed=y)

map_fit = pmx.find_MAP(
    model=model, method="L-BFGS-B", initvals={"theta": 0.0},
    progressbar=False, random_seed=41,
)
print(map_fit["posterior"]["theta"])  # One optimized point, not posterior draws.
print(map_fit["optimizer_result"].to_dataset()[["success", "message", "fun"]])

laplace = pmx.fit(
    method="laplace", model=model, optimize_method="trust-exact",
    initvals={"theta": 0.0}, draws=500, random_seed=42, progressbar=False,
)
print(laplace["fit"].to_dataset()[["mean_vector", "covariance_matrix"]])
print(laplace["optimizer_result"].to_dataset()[["success", "message", "jac"]])
oracle_mean, oracle_variance = y.sum() / 5, 1 / 5
np.testing.assert_allclose(
    map_fit["posterior"]["theta"].values, oracle_mean, rtol=0, atol=1e-7,
)
np.testing.assert_allclose(
    laplace["fit"]["mean_vector"].values, [oracle_mean], rtol=0, atol=1e-7,
)
np.testing.assert_allclose(
    laplace["fit"]["covariance_matrix"].values, [[oracle_variance]],
    rtol=0, atol=1e-7,
)

dadvi = pmx.fit(
    method="dadvi", model=model, n_fixed_draws=100, n_draws=500,
    optimizer_method="trust-ncg", random_seed=43, progressbar=False,
)
print(dadvi["optimizer_result"].to_dataset()[["success", "message", "fun"]])

pathfinder = pmx.fit(
    method="pathfinder", model=model, num_paths=4, num_draws=500,
    num_draws_per_path=200, num_elbo_draws=20,
    importance_sampling="psis", parallel=False,
    random_seed=44, progressbar=False,
)
print(pathfinder["pathfinder"])
print(pathfinder["lbfgs"])
for name, result in [("laplace", laplace), ("dadvi", dadvi),
                     ("pathfinder", pathfinder)]:
    draws = result["posterior"]["theta"]
    print(name, float(draws.mean()), float(draws.std()))
    result.to_netcdf(f"{name}.nc")
```

Extras `find_MAP(method=..., *, model=..., ...)` is a different interface from
core `pm.find_MAP`: in 0.15.1 it returns a **DataTree**, not a point dictionary or
`(point, OptimizeResult)` tuple, despite a stale return annotation in the source.
It supports partial `initvals`, explicit `jitter_rvs`, gradient/Hessian/Hessian-vector
controls and SciPy optimizer options as kwargs. Inspect `optimizer_result` success,
message, objective, gradient when available and sensitivity to independent starts.
No automatic restart scheme establishes global optimality. `compute_hessian=True`
also requests curvature in `fit`; dense curvature costs quadratic storage.

Laplace uses `optimize_method`, `draws` and nested `optimizer_kwargs`; model is
keyword-only, so `fit_laplace(model)` misbinds it. Supplying `chains` raises.
The example uses `trust-exact` to request Hessian-based optimization; the default
`BFGS` can instead reuse approximate inverse-BFGS curvature. Do not interpret
`optimizer_result["hess_inv"]` as an independently checked exact covariance.

DADVI uses **`n_fixed_draws`** for the finite objective and **`n_draws`** for
output simulation. Increasing output draws cannot improve the fitted objective.
`optimizer_result["x"]` contains variational locations and log scales (with
coordinate labels), not a parameter MAP point. Compare independent seeds and
larger fixed samples as well as optimizer status; even a perfectly optimized
finite objective retains family and finite-objective error.

Multi-path Pathfinder uses `num_paths`, `num_draws_per_path`, `num_elbo_draws`
and final `num_draws`; tune its optimizer through
`from pymc_extras.inference.pathfinder import LBFGSConfig` and
`lbfgs_config=LBFGSConfig(maxiter=...)`, not old backend-specific kwargs.
Keep `jacobian_correction=True` for transformed variables. Inspect
`pathfinder["path_status_counts"]`, `pathfinder["pareto_k"]` when present,
the group's warning attributes and `lbfgs["status_counts"]`/`["niter"]`.
Retain failed-path information; aggregate counts alone are not proof every path
found an adequate proposal. `"psis"` and `"psir"` use importance resampling;
`"identity"` uses unsmoothed importance weights for resampling, while `None`
returns unweighted per-path draws without correction. Do not call raw paths
posterior samples merely because they occupy a `posterior` group.
`fit_blackjax_pathfinder(model=model, num_draws=..., random_seed=...)` is a
separate **single-path** API, without multi-path aggregation or jitter retries.

## INLA: select a latent-Gaussian structure, not an arbitrary model

The intended structure is hyperparameters `theta`, a Gaussian latent vector `x`
with known-form positive-definite prior precision `Q(theta)`, and observations
linked to a linear predictor `A @ x`. A non-Gaussian response likelihood is
possible; the Gaussian requirement is on the field, not on the response.
In 0.15.1, `fit_INLA` constructs a Laplace-marginalized model then calls
`pm.sample` for the remaining variables. It does **not** implement generic
deterministic integration over hyperparameters. Its MCMC diagnostics assess the
approximate marginal target, not the eliminated-field approximation.

```python
import numpy as np
import pymc as pm
import pymc_extras as pmx

Q = np.eye(2)  # Full prior precision for the two-dimensional Gaussian field.
with pm.Model() as latent_model:
    theta = pm.Normal("theta", 0.0, 1.0)
    x = pm.MvNormal("x", mu=theta * np.ones(2), tau=Q)
    pm.Poisson("counts", mu=pm.math.exp(x), observed=np.array([1, 3]))

inla = pmx.fit(
    method="INLA", model=latent_model, x=x, Q=Q,
    return_latent_posteriors=False,
    minimizer_kwargs={"method": "L-BFGS-B",
                      "optimizer_kwargs": {"tol": 1e-8}},
    draws=500, tune=500, chains=2, cores=1, random_seed=45,
)
inla.to_netcdf("inla.nc")
```

This is an experimental interface: keep its warning visible. The inner optimizer
and dense Hessian can dominate repeated log-density evaluations; sparse precision
input is not a promise of a complete sparse-INLA solver. Supply the actual prior
precision, not the posterior precision or a convenient scalar stand-in.
Check conditional modes, curvature, inner-optimization sensitivity and comparison
with the original joint model or numerical integration. Multimodal, weakly
identified, skewed or boundary-concentrated latent conditionals need special caution.
`minimizer_seed` is currently unused (initialization is deterministic);
`return_latent_posteriors=True` raises `NotImplementedError`.
Laplace field recovery via `recover`/`conditional` is also unsupported; see
[marginalization and prediction](mixtures.md#laplace-elimination-is-approximate).
Do not select INLA if those missing outputs are required.

## Declare intended accuracy

Mean screening, uncertainty and future-response moments are separate uses. Compare
parameters, scientifically important contrasts, interval endpoints and event
probabilities to an analytic reference or well-diagnosed alternative. Observation
noise may mask poor parameter uncertainty. Set tolerances in meaningful units or
reference SDs before seeing results. Include optimizer behavior and multiple
independent starts without selecting only favorable runs. MCMC is not obligatory
when a stronger analytic reference exists, but a small Gaussian success does not
justify an arbitrary model's uncertainty claims.

## Separate family, optimization and output-draw errors

For a Gaussian posterior `p=N(m,C)` with precision `P=C^-1`, reverse-KL mean-field
has optimum `q*=N(m,diag(1/P_jj))`, **not** `N(m,diag(C))`. It underestimates marginal
variances under correlation; contrast variance error can go either way. Full-rank
VI contains this target only at the correct fitted parameters.

\[
\mathrm{KL}\{N(a,S)\Vert N(m,C)\}
=\tfrac12[\operatorname{tr}(C^{-1}S)+(a-m)^TC^{-1}(a-m)-d+\log|C|-\log|S|].
\]

This supplies an analytic family-error floor. More optimizer steps or output
draws cannot remove it. A moment-matched Gaussian KL for particles is not the
empirical measure's KL.

A flat noisy ADVI loss or small parameter change does not establish an optimum.
Excess KL over the family floor separates optimization quality from the family
restriction where an oracle exists. A Stein objective is not negative ELBO;
a Gaussian quality distance cannot be relabelled as its optimization objective.

DADVI holds base Normal draws fixed while optimizing a sample-average objective.
For a Gaussian target let `zbar=mean(epsilon)`, `V` be the empirical covariance
using divisor N, and `A=P*V` elementwise. Its finite-objective optimum satisfies

\[
a_{\rm SAA}=m-s_{\rm SAA}\odot\bar z,\qquad
s_{\rm SAA}=\arg\min_{s>0}\{\tfrac12s^TAs-\sum_j\log s_j\}.
\]

An independent convex calculation can distinguish optimizer error from finite
objective error and the infinite-objective family floor. Reconstruct the same
base sample only for checking that specific finite objective, not for claiming
independent replication.

Pathfinder selects proposals along optimization paths with estimated ELBO and
can use PSIS importance reweighting. Inspect path failures, importance Pareto k,
final moments and intervals. This k concerns proposal importance weights, not LOO.
Multiple paths are not converged MCMC chains; optimization, proposal, weight and
resampling errors are not fully separable from final draws alone.

For iid draws from a **fixed fitted Gaussian**, mean MCSE is
`sqrt(diag(S)/N)`. It can be tiny around a biased answer. This conditional MCSE
excludes family and optimizer error and cannot be assigned unchanged to dependent
Stein particles or weighted/resampled outputs. Never reshape approximation draws
or optimization restarts into chains and report R-hat/ESS as MCMC convergence.

## Laplace coordinates and uncertainty

A local quadratic is exact only for a genuinely Gaussian target in the coordinates
of the Hessian, at the exact mode with exact curvature. Approximate inverse BFGS
curvature can still be inaccurate. Align returned `mean_vector` and
`covariance_matrix` by dimension labels, not assumed variable order.

In extras 0.15.1 the MAP helper evaluates `model.logp(jacobian=False)` in value
coordinates before transforming draws back. For nonlinear transformations this
is not the same local quadratic as the Jacobian-adjusted unconstrained density,
nor an original-space Gaussian moment calculation. DADVI and default Pathfinder
use the Jacobian-adjusted unconstrained target instead. These distinctions matter
even when outputs share a `posterior` group. Check boundary modes, skewness,
funnels, tails and missing modes; more observations alone proves neither speed
nor accuracy.

## Advanced VI without confusing state and inference

- `KLqp(approx,beta=1)` is ordinary reverse-KL; changing beta changes the target
  objective. Core `fit` shortcut `start_sigma` is mean-field ADVI-specific.
- `fit` mutates/returns the approximation. `refine` reuses a fitted step function;
  `run_profiling` runs optimization updates, so profile a disposable fit.
- Groups cover each free variable once; one `Group(None,...)` can absorb remaining
  variables. Supply either `vfam` or complete `params`, not both. Custom parameter
  dictionaries need correct shapes and `more_obj_params` for optimization.
- `MeanField` uses diagonal scales; `FullRank` uses mu/L_tril; `Empirical` stores
  particles with `has_logq=False`. In PyMC 6.3.1 Empirical initialization from a
  DataTree is unimplemented; inspect its supported initialization rather than
  silently treating draws as a full-rank density.
- Flattened ordering and transforms matter. Nonlinearly transformed locations/
  scales (including `state.std`) are not exact original-space moments.
- Prefer seeded `approx.sample`. `sample_node(expression,size=N)` draws symbolic
  expressions; `deterministic=True` substitutes a base location, not E[f(theta)].
  `evaluate_over_trace` reuses particles and adds no information.
- PyMC 6.3.1 grouped wrappers have an RNG/`step_function` compatibility boundary.
  Inspect current source before using them. An explicit `objective.updates` graph
  compiled with `pm.compile(..., random_seed=...)` is an advanced alternative only
  when it preserves the chosen objective, update rules and RNG semantics.
- Symbolic replacement/size protocols require actual dimensions and replacements
  before evaluation. OPVI density-normalization views are objective scaling, not
  estimates of marginal likelihood or calibrated probabilities.
- Custom Operator/KL/KSD and TestFunction extensions must satisfy density/kernel
  requirements. `obj_n_mc` and `tf_n_mc` estimate different expectations; optimizer
  parameter lists and updates specify what mutates. KSD has a distinct loss contract.
- `Tracker` callbacks accept `(approx, loss, iteration)` (or a nullary function);
  convergence callbacks stop on parameter changes, not target-distribution accuracy.
- SGD, momentum/Nesterov, Adagrad, RMSProp, Adadelta, Adam and Adamax supply update
  rules, not universal tuning recommendations. Norm constraints control tensor/
  combined gradient norms; vector `norm_constraint` needs explicit axes in 6.3.1.
  Clipping does not fix an unstable model or justify hiding exploding gradients.

Sources: [core VI](https://www.pymc.io/projects/docs/en/stable/api/vi.html),
[OPVI](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/variational/opvi.py),
[core inference](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/variational/inference.py),
[extras inference](https://www.pymc.io/projects/extras/en/stable/api/inference.html),
[release dispatcher](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/inference/fit.py),
[MAP and Laplace source](https://github.com/pymc-devs/pymc-extras/tree/v0.15.1/pymc_extras/inference/laplace_approx),
[DADVI source](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/inference/dadvi/dadvi.py),
[Pathfinder source and output schema](https://github.com/pymc-devs/pymc-extras/tree/v0.15.1/pymc_extras/inference/pathfinder),
[INLA source](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/inference/INLA/inla.py),
[release dependency metadata](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pyproject.toml),
[ADVI](https://jmlr.org/papers/v18/16-107.html),
[DADVI](https://www.jmlr.org/papers/v25/23-1015.html),
[Pathfinder](https://jmlr.org/papers/v23/21-0889.html).
