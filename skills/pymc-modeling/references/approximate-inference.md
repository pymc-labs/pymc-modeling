# Approximate inference: measure error for the intended use

## Select the method and compatibility deliberately

| Method | API and main limitation |
|---|---|
| Mean-field ADVI | `pm.ADVI` or `pm.fit(method="advi")`: Gaussian in sampling coordinates with diagonal covariance. |
| Full-rank ADVI | `pm.FullRankADVI` or `pm.fit(method="fullrank_advi")`: dense Gaussian, not arbitrary skewness or multimodality. |
| Grouped VI | `pm.Group`, `pm.variational.Approximation`, `pm.variational.KLqp`: independent groups lose cross-group dependence even if each is full rank. |
| SVGD | Empirical particles, kernel/temperature and finite-particle sensitivity; not independent MCMC samples. |
| ASVGD, ImplicitGradient | Experimental generator/Stein methods; PyMC 6.3.1 source discourages routine use. Keep warnings visible. |
| MAP | `pm.find_MAP(..., return_raw=True)`: one point, no posterior uncertainty and no universal NUTS initialization prescription. |
| Laplace | `pymc_extras.inference.fit_laplace(model=model, ...)`: local mode/Hessian approximation; coordinates and curvature accuracy matter. |
| DADVI | `pymc_extras.inference.fit_dadvi(model=model, ...)`: deterministic finite-sample mean-field objective, not exact inference. |
| Pathfinder | `pymc_extras.inference.fit_pathfinder(...)`: quasi-Newton Gaussian paths plus selection/reweighting. |

In extras 0.14.0, dependencies require PyMC <6.3 and PyTensor <3.3; use a compatible
project environment, not overridden bounds. `fit_dadvi` is not exported at the
package root; `pmx.fit(method="dadvi", ...)` is a separate extras dispatcher.
Core `pm.fit` accepts advi/fullrank_advi/svgd/asvgd or an Inference object, not
extras methods. BlackJAX Pathfinder needs its own compatible JAX stack; CPU
execution does not establish accelerator compatibility. Full-rank DADVI and
arbitrary flow/NumPyro integrations are not supplied by these interfaces. Extras
`fit_INLA(x,Q,...)` is specialized linear-mixed-model inference, not a Laplace alias.

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

In extras 0.14.0 `fit_laplace` begins with `optimize_method`; model is keyword-only.
`fit_laplace(model)` misbinds the model. `chains=` is rejected as deprecated;
the result is a DataTree, not a fitted Approximation. Its MAP helper evaluates
`model.logp(jacobian=False)` in value coordinates before transforming draws back.
For nonlinear transformations this is not the same local quadratic as the
Jacobian-adjusted unconstrained density, nor an original-space Gaussian moment
calculation. Check boundary modes, skewness, funnels, tails and missing modes;
more observations alone proves neither speed nor accuracy.

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
[Laplace source](https://github.com/pymc-devs/pymc-extras/tree/v0.14.0/pymc_extras/inference/laplace_approx),
[ADVI](https://jmlr.org/papers/v18/16-107.html),
[DADVI](https://www.jmlr.org/papers/v25/23-1015.html),
[Pathfinder](https://jmlr.org/papers/v23/21-0889.html).
