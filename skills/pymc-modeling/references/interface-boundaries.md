# Advanced numerical and extension interfaces

Prefer high-level model and distribution APIs. Use lower-level protocols when
extending an optimizer, distribution or compiler, not merely because a symbol is
importable. Inspect version-matched source before relying on internals.

## Numerical points and gradients

`DictToArrayBijection.map` returns `RaveledVars`: flattened data plus
`point_map_info` containing names, shapes, sizes and dtypes. `rmap` restores these;
`start_point` preserves entries outside the mapped subset. `mapf` adapts a
point-input function. Preserve order and metadata; an unlabelled vector is not
portable between models. Dtype restoration does not excuse lossy mixed-type casts.

`ValueGradFunction` evaluates a scalar cost and gradient. `set_weights` leaves the
first cost weight at one; changed weights define a changed objective, not the
original posterior. Extra fixed inputs must be initialized with `set_extra_values`.
Inputs need unique names and compatible floating dtypes; inspect `ravel_inputs`
for flattened versus separate inputs. In PyMC 6.3.1 a `copy()` resets the extra-value
initialization guard, so initialize the copy before calling it. Prefer
`model.compile_logp`, `compile_dlogp` and `compile_fn` for ordinary inspection.

Sources: [bijection](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/blocking.py),
[model numerical functions](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/model/core.py).

## Approximation state

Inference objects' `.approx` exposes an approximation, not healthy MCMC output.
Inspect family, optimizer behavior and reference accuracy before interpreting draws.

- `model`, `ndim`/`ddim` and `params` identify the graph and optimizer parameters.
- `has_logq` indicates a density representation; empirical particles need not have
  a normalized smooth density. `joint_histogram` requires all groups to be empirical.
- `sample_dict_fn` produces mapped approximation points, not independent chains.
- Group `mean`, `std`, `cov`, `mean_data` and `std_data` describe optimizer-space
  quantities. Nonlinear inverse transforms of locations/scales are not generally
  the induced original-space means/SDs. Use draws or analytic moment formulas.

Symbolic OPVI objectives, replacement maps, shared buffers and Stein kernel
matrices implement optimization; they are not pointwise observation likelihoods
or marginal likelihood estimates.

Sources: [OPVI](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/variational/opvi.py),
[Stein optimization](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/variational/stein.py).

## Distribution and step-method extensions

Step `vars` identifies inference-space variables; compound steps aggregate them.
Step names label algorithms/statistics, not model parameters. Class defaults are
not universal sampling recommendations; configure the fit through supported APIs.

Distribution constructors register RVs; `.dist()` creates an unregistered RV.
`rv_op`/`rv_type` bind distribution classes to random operations.
`SymbolicRandomVariable.extended_signature` describes RNG/size and input/output
roles; `inline_logprob` controls symbolic probability derivation. Do not guess
signature grammar from another release. Check event/batch shape, RNG updates,
log density, derivatives and predictive draws when extending these protocols.

Inner inputs/outputs, `fn`, `itypes`/`otypes` and `view_map`/`destroy_map` are compiler
contracts. Check aliasing and permitted mutation when implementing them.
Transform `name`/`ndim_supp` describes inference-space support handling;
internal bound-input indices are not user-specified numeric bounds.

Source: [distribution infrastructure](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/distribution.py).

For warnings and failed calls, start with [Troubleshooting](troubleshooting.md).
