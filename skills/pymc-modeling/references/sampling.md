# Sampling algorithms, backends and storage

Separate the scientific model, inference algorithm, and numerical/storage backend.
Compilation failure does not authorize deleting discrete latents, changing an
ordered prior or weakening the likelihood. Exact marginalization can preserve a
model, but must be justified explicitly. Select an alternative backend deliberately,
not as a hidden fallback.

## Algorithm selection

| Algorithm | Appropriate use and diagnostics |
|---|---|
| NUTS | Continuous differentiable value variables. Diagnose divergences, tree depth, energy/BFMI, rank R-hat, bulk/tail ESS and estimand MCSE. Native `init_nuts`/`init` does not initialize every external backend. |
| HamiltonianMC | Fixed trajectories with gradient/mass adaptation. Trajectory length depends on scale; zero divergences does not imply efficient movement. |
| Slice | Evaluable log density without derivatives; coordinate updates can mix poorly under correlation. Width adaptation is not convergence. |
| Metropolis | Gradient-free random walks; scale and proposal geometry matter. Heavy-tailed proposals are not universally efficient. |
| DEMetropolisZ | Within-chain history proposals; tuning/history reset is not posterior thinning. |
| DEMetropolis | Interacting population members need other members for difference proposals. They are not independent chains: use independent populations and dependence-aware precision checks. |
| BinaryMetropolis, BinaryGibbsMetropolis | Binary latents, not arbitrary counts. Conditional updates are not independent joint-posterior draws. |
| CategoricalGibbsMetropolis | Discrete categories; inspect category occupancy/transitions and event probabilities, not numeric category means alone. |
| CompoundStep | Exactly one compatible update block per free RV, e.g. native NUTS plus discrete kernels sharing the same joint target. |
| SMC (IMH/MH) | Likelihood tempering, resampling and particle mutation. Check final beta=1, weights, acceptance and between-independent-population error. Particle index is not MCMC time. |

Custom step lifecycle, seeding, initialization, population linking and tuning
transitions are driver responsibilities. `Competence` ranks support suitability,
not scientific quality. Stored `sampling_state` is not a posterior or a guarantee
of resuming different graph/code/RNG/adaptation versions.

## Proposals and support

Normal, Uniform, Laplace and Cauchy random-walk proposals are symmetric; a
multivariate proposal needs a positive-definite covariance of the correct dimension.
A centered proposal is not necessarily symmetric: `PoissonProposal(s)` produces
`Poisson(s)-s` in PyMC 6.3.1. Asymmetric proposals need forward/reverse Hastings
corrections and matching support. `metrop_select` consumes a supplied log ratio;
that release rejects all nonfinite ratios, including positive infinity.

For general discrete random walks, invalid proposals must reach a `-inf` target
rather than crash from an out-of-range tensor index. Support-safe algebra can
preserve the model on valid states; clipping proposals changes the kernel.
Enumerate small finite targets independently when checking category transitions.
Version caution: PyMC 6.3.1 proportional categorical proposals require scrutiny of
their normalization/Hastings calculation; for proposals excluding the current
category the acceptance ratio uses original conditional masses
`(1-p_current)/(1-p_proposed)`. Inspect the installed
[Metropolis source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/step_methods/metropolis.py)
and an independent finite-state reference rather than infer correctness from R-hat.

## Backend and dtype choices

Prefer nutpie when installed and compatible; no universal speed ratio follows.
`nuts_sampler` selects implementation; numerical `backend` selects compilation.
Native C/CVM can be requested with `Mode(linker="cvm", optimizer="fast_run")`;
`FAST_RUN` with an automatic linker need not mean C. External NUTS generally
requires continuous differentiable graphs and does not support native trace/callback
options or every native initialization form.

```python
with model:
    idata = pm.sample(nuts_sampler="nutpie", tune=tune, draws=draws,
                      chains=chains, random_seed=42)
idata.to_netcdf("posterior.nc")
```

NumPyro/BlackJAX require compatible JAX graph lowerings. Set x64/device policy before
creating arrays if precision/device matters. Vectorized chains need not correspond
to distinct devices. CPU operation is not proof of GPU support or acceleration.
In PyMC 6.3.1 use `nuts={...}` for supported options rather than deprecated
`nuts_sampler_kwargs`; implementations do not necessarily accept identical options.
The PyMC 6.3.1/BlackJAX 1.6.2 adaptation/progress interfaces differ: check wrapper
compatibility rather than assuming disabling progress repairs an injected keyword.

For low-level [JAX adapters](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/sampling/jax.py),
verify sign conventions with a known density: the 6.3.1 `get_jaxified_logp` flag
`negative_logp=True` returns log density despite the surprising name. Some assertion
lowerings return their input without a runtime check; validate data/support before
relying on the compiled graph. Float64 inputs alone do not repair already-created
float32 constants. Compare density/gradients under the chosen construction policy.

## Budgets, diagnostics and failure diagnosis

Distinguish **exploratory computation** from **reportable inference**. Short pilots
or rough approximations can expose bugs, geometry problems and gross misfit without
meeting final precision targets. Keep warnings visible and label those outputs
preliminary; use them to decide what to investigate, not to claim convergence or
reliable posterior probabilities. Escalate computation when the intended inference
needs it, rather than fully fitting every model that may soon be abandoned.

Measure a feasibility pilot's compilation, sampling and concurrent-chain/storage
memory. Allocate warmup from adaptation and geometry and retained draws from
precision, without universal ceilings. For a mean, approximately
`MCSE/SD = 1/sqrt(ESS)`; ESS 400 gives roughly 5% of a posterior SD. Quantiles,
tails and rare events need their own precision. Several long enough independent
chains are preferable to treating many short fragments as independent exploration.

For reportable MCMC inference, use rank R-hat <1.01 and bulk/tail ESS >=400 as
screening targets, not guarantees.
Specify tail probabilities for reported intervals, inspect every divergence,
energy and depth saturation where available, and visually assess traces/ranks.
Missing backend statistics are unavailable, not zero. For discrete variables,
constant quantile indicators can make tail diagnostics degenerate; inspect actual
category probabilities/visits. Check independent reference agreement when possible.

SMC particles within a population share resampling ancestry. Check terminal
annealing and finite final statistics, and estimate uncertainty across independent
populations; a handful of replicates provides noisy precision estimates. Nonterminal
missing marginal-likelihood entries differ from invalid final values. Neither SMC
nor differential evolution guarantees multimodal exploration.

Distinguish construction/support, density compilation, initialization, adaptation,
retained sampling, persistence and diagnostic failures. External wrappers can fuse
lazy compilation, warmup and conversion: do not assign a traceback to a phase
without actual information. Preserve warnings and raw results. A cold cache may
explain compilation cost, not prove sampler failure or successful compatibility.

Performance comparisons must control target, dtype, hardware/thread topology,
cache state, chain layout, compilation amortization and attained precision, with
replicates. Whole-driver time is not isolated sampling time. Do not search seeds
or weaken diagnostic criteria to obtain a favorable report.

## Storage and conversion

Use DataTree bracket groups. Save the raw result before plots, predictions and
explicit `pm.compute_log_likelihood`; enrich a separate output. Keep warmup separate
where supplied. Reopen important files and compare values, coordinates and draw
identities, not merely file existence. Persistence failures are not nonconvergence.

For native `return_inferencedata=False`, let the driver create a fresh NDArray for
each chain and convert with `pm.to_inference_data`. PyMC 6.3.1 shallow-copy behavior
makes a shared empty NDArray template risky for independent sample buffers; inspect
[backend source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/backends/ndarray.py)
and check accidental duplicate continuous streams. Resuming a nonempty NDArray is
not supported by that initializer. `trace_cov` flattens values for covariance, not
chain diagnostics. `find_hessian` gives value-space curvature; its inverse is not
a posterior covariance absent suitable Gaussian assumptions. `guess_scaling` is an
initialization heuristic, not a curvature certificate.

Disk-backed `ZarrTrace` can retain completed flushed chunks after interruption;
in-memory stores are not persistent, buffered writes may be lost and restart is
not guaranteed. Check package compatibility: PyMC 6.3.1 native Zarr backend uses
Zarr 2 APIs, while nutpie 0.16.11 requires Zarr >=3.1. Do not override conflicting
requirements. Choose a compatible native storage setup deliberately.

Preserve object-valued sampler warnings before removing nonserializable fields
from a conversion view with `drop_warning_stat`; keep raw storage intact. Zarr fill
metadata can mask legitimate zero draw IDs or false statistics when carried into
NetCDF. Check masking/scaling and exact reopened values before conversion. Diagnose
prefixed statistics by their real names rather than rewriting raw output.

Thinning reduces storage/downstream work, not already-computed Monte Carlo error.
`pm.stats.thin(posterior_dataset,factor=...)` should preserve selected draw identities
in a separate result. ESS per stored draw may rise while total information does
not; estimated ESS can fluctuate, especially for antithetic chains. Never use
thinning to hide nonstationarity, divergences or short chains. Keep all exported
groups aligned or explicitly export only the posterior Dataset.

Sources: [sampling driver](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/sampling/mcmc.py),
[SMC](https://github.com/pymc-devs/pymc/tree/v6.3.1/pymc/smc),
[Zarr storage](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/backends/zarr.py),
[rank diagnostics](https://doi.org/10.1214/20-BA1221),
[Bayesian Workflow](https://users.aalto.fi/~ave/Bayesian-Workflow.pdf)
(2026 corrected edition, §§11.3–11.5, 12.1 and Chapter 13).
