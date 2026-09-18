# Model, data and dimension semantics

## Construct and inspect the graph

Use `with pm.Model(coords=...) as model:` for registration; pass `model=model`
when operating outside the context. `pm.modelcontext` resolves a model, not a new
one. Named/nested models change registered prefixes. Reusing a name does not
create an independent parameter. Prefer the top-level `pm.Model` constructor to
aliases or internal context managers.

`pm.Deterministic` records an expression, not a density factor. Its value is fixed
conditional on its inputs, but can be uncertain marginally. Store interpretable
quantities needed downstream; large deterministics can dominate memory/storage.
After `model.copy()`, retrieve variables from the clone instead of mixing graphs.

| Interface | Purpose and boundary |
|---|---|
| `named_vars`, model indexing, `named_vars_to_dims` | Registered identity and labels, not posterior draws. |
| `free_RVs`, `observed_RVs`, `deterministics`, `data_vars` | Separate parameters, observations, expressions and predictors. `unobserved_RVs` can include deterministics; do not count it blindly as parameter dimension. |
| `value_vars`, `continuous_value_vars`, `discrete_value_vars` | Actual inference inputs. NUTS does not update discrete variables. |
| `rvs_to_values`, `values_to_rvs`, `rvs_to_transforms`, `rvs_to_initial_values` | Connect model RVs to sampling coordinates. Do not manually mutate coupled registries. |
| `initial_point`, `check_start_vals`, `point_logps`, `debug` | Locate invalid factors. `point_logps` rounds by default; compile density for tight numerical checks. |
| `eval_rv_shapes`, `shape_from_dims`, `dim_lengths`, `coords` | Inspect event/batch shape and symbolic lengths; evaluate lengths when needed. |
| `logp`, `dlogp`, `d2logp`, corresponding `compile_*` | Density/derivatives in value-variable coordinates. In PyMC 6.3.1, `d2logp` negates the Hessian by default; set `negate_output` explicitly in sign-sensitive calculations. |
| `datalogp`, `observedlogp`, `varlogp`, `varlogp_nojac` | Different factor totals. They are not interchangeable with pointwise observation likelihood. |
| `replace_rvs_by_values`, `compile_fn`, `Point` | Evaluate conditional on a numerical point rather than draw unresolved random inputs. `.eval()` mappings use symbolic-variable keys; compiled model point functions use value-variable names. |
| `table`, `str_repr`, `to_graphviz`, `profile` | Inspect structure and measure computation in the actual backend. Graphviz needs its Python package and system executable. |

For custom builders, `register_rv`, `add_named_variable`, `create_value_var`,
`make_obs_var`, `register_data_var` and `set_initval` maintain interdependent
registration, transforms, initial values and dimensions. Prefer constructors;
if extending registration, check density, derivatives and predictive generation.

## Data and observation identity

```python
with pm.Model(coords={"obs_id": row_ids, "feature": feature_names}) as model:
    x = pm.Data("x", X, dims=("obs_id", "feature"))
    beta = pm.Normal("beta", 0, 1, dims="feature")
    sigma = pm.HalfNormal("sigma", 1)
    mu = pm.Deterministic("mu", pm.math.dot(x, beta), dims="obs_id")
    pm.Normal("y", mu, sigma, observed=y, shape=x.shape[0], dims="obs_id")
```

Choose priors from the project's units; the example's scales are not universal.
Before stripping table labels, check row counts, unique ordered identities,
finite values, support and units. Equal shapes do not prove correct alignment.
Keep identities consistent across `observed_data`, `constant_data`, log likelihood
and predictive groups. Avoid reserved `chain`, `draw` and `__sample__` model dims.

Update shared containers with `pm.set_data({...}, model=model, coords={...})` or
`model.set_data(name, values, coords=...)`. Shape can change; rank cannot. Update
all predictors, exposures, offsets and group indices sharing a resized axis.
Prevalidate the complete update: setters are not transactional. Restore training
data in `finally` after prediction, and keep future observations out of the fit.
`add_coord`/`add_coords` register definitions; `set_dim` requires compatible values.
Raw `.set_value()` alone does not coordinate model-wide resizing.

Version caution: [PyMC 6.3.1 Data](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/data.py)
has no explicit legacy `mutable` argument and rejects generators and masked/NaN
containers. Missing values passed directly as likelihood observations can trigger
imputation; that requires a defensible missing-data model, not silent cleaning.
Example-data retrieval with `pm.get_data` does not establish license, privacy,
schema or suitability for a scientific question.

`pm.Minibatch(x, y, batch_size=...)` shares row indices across inputs. Separate
minibatches destroy pairing. Correct `total_size` likelihood scaling and an
appropriate stochastic optimizer are required; minibatching is not a generic
speed switch for full-data NUTS or a held-out split.

## Ordinary dims and experimental named computation

Ordinary `dims` labels shape; positional broadcasting governs arithmetic.
`alpha[group_idx]` is still needed to map group effects to observations. Coordinates
do not join tables, discover membership or reorder arrays.

`pymc.dims.Data` and `Deterministic` use xtensor dimension names for broadcasting
and transposition. Supply names for non-scalar raw arrays. A deterministic can
infer existing xtensor names; explicit dims can reorder them. Distribution
`core_dims` expresses support semantics, not just batch labels. Ordinary and named
constructors need not accept identical arguments. Keep the experimental warning.

Xtensors expose symbolic `dims`, `sizes`, `shape`, `ndim`, `dtype`, `broadcastable`,
`T`, `real` and `imag`; `.values` drops named-axis safeguards. In
[PyTensor 3.3.0](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/xtensor/type.py),
`as_numpy` returns the symbolic object, while `coords` and `.loc` are unimplemented:
these are named dimensions, not xarray coordinate joins. Graph owners describe
computation, not an identified causal structure.

## Frozen models

Construct immutable graphs with the supported `freeze_model` transformation, not
`FrozenModel(...)`. New graph nodes are rejected; some data remain settable only
when free variables do not depend on them. Inspect which data became constants
before promising reusable prediction.

Sources: [model API](https://www.pymc.io/projects/docs/en/stable/api/model/core.html),
[model implementation](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/model/core.py),
[dimensionality](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/dimensionality.html),
[experimental dims](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/dims_module.html).
