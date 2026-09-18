# Model transformations and causal queries

## Choose the operation from the question

| Intent | Operation | What it does not do |
|---|---|---|
| Condition on an RV's value | `pm.observe(model,{"X":value})`, then inference | Does not cut X's incoming edges or remove its likelihood. |
| Replace a structural assignment | `pm.do(model,{"X":value})` | Does not discover/validate a causal graph or identify effects from data. |
| Set a covariate-dependent policy | `pm.do(model,{"X":model["U"]+1})` | Is not a constant intervention; dependency on U remains. |
| Change value coordinates | `change_value_transforms`, `remove_value_transforms` | Does not intervene or change likelihood support. |
| Convert sample coordinates | `unconstrain_values`, `constrain_values` | Adds no inference or information. |
| Clone, label, freeze or remove minibatches | Graph transformations | Does not by itself establish numerical equivalence or speed. |

Model transformations return new models; retrieve their variables by name rather
than mix cloned/original symbols. `observe`/`do` accept name or variable keys with
compatible shape/dtype. Observe accepts eligible RVs/Deterministics, not Data;
observing a deterministic needs a derivable density. Re-observing replaces the
old value rather than appending another observation.

## Conditioning is not intervention

Consider known structural equations with independent standard-Normal errors:
`U=eU`, `X=U+eX`, `Y=2X+3U+eY`. U is a common cause. This mathematical example has
known coefficients, not causal parameters learned from one observation.

```python
with pm.Model() as scm:
    U = pm.Normal("U", 0, 1)
    X = pm.Normal("X", U, 1)
    mu_Y = pm.Deterministic("mu_Y", 2*X + 3*U)
    Y = pm.Normal("Y", mu_Y, 1)
conditioned = pm.observe(scm, {"X": 2.0})
intervened = pm.do(scm, {"X": 2.0})
```

The covariance of (U,X,Y) is `[[1,1,5],[1,2,7],[5,7,30]]`. In mean/variance
notation, `U|X=x ~ N(x/2,1/2)`, `Y|X=x ~ N(3.5*x,5.5)`, while
`Y|do(X=x) ~ N(2*x,10)`. Observing X selects information about U; intervening
preserves population U. A policy X=U+1 instead gives Y mean 2/variance 26.

Compiled observed-model logp is `log p(u)+log p(x|u)+log p(y|x,u)`, an
unnormalized conditional joint, not `log p(u,y|x)`: account for evidence p(x)
when checking normalized conditional densities. Intervention removes p(x|u).
Within a fixed U stratum, normalized Y queries can agree despite different
evidence constants.

Use forward sampling of an intervened generative model for its population query.
Use real inference for the observed model and save immediately. Prior predictive
sampling in an observed model does not infer U|X. If posterior predictive sampling
should retain U|X=x but generate fresh Y at fixed x, hold X fixed in the prediction
graph and deliberately regenerate Y:

```python
prediction_model = pm.do(conditioned, {"X": 2.0},
                         make_interventions_shared=False)
replication = pm.sample_posterior_predictive(
    conditional_posterior, model=prediction_model,
    var_names=["Y", "mu_Y"], sample_vars=["Y", "mu_Y"],
    freeze_vars=["U"], predictions=True, random_seed=42,
)
```

This retains the conditional posterior U|X=2 and therefore answers the conditional
Y query, **not** population do(X=2). In PyMC 6.3.1 `var_names` alone selects outputs,
while observed RVs absent from trace regenerate and trace RVs require explicit
resampling when intended. Check draw-wise deterministic identities and fresh-noise
moments, not just different graphs.

## Causal assumptions are separate

A well-defined intervention needs consistency, no interference (or an explicit
spillover model), invariant nonintervened mechanisms and an appropriate target
population. Backdoor adjustment additionally needs exchangeability conditional on
adequately measured U, positivity and selection assumptions:
`p(y|do(x))=integral p(y|x,u)p(u)du`. Hidden confounding or a latent U in a fitted
observational model does not identify structural effects without further design
or assumptions. MCMC cannot validate them. Population interventions, conditional
predictions and individual counterfactual abduction/action/prediction are distinct.

## Cloning, deterministic labels and numerical equivalence

`pymc.model.fgraph.clone_model`, `fgraph_from_model` and `model_from_fgraph` support
graph conversion. The conversion returns `(fgraph,memo)` mapping original variables
to cloned intermediate representations. Check density preservation and data
aliasing before relying on a clone. In PyMC 6.3.1 the implementation deep-copies
shared Data, RNGs and dimension lengths despite inconsistent clone docstrings.
It rejects nested submodel conversion and nondefault initial values; transform
the root and supply suitable inference starts afterward. Existing GP objects may
still reference the old graph and need rebuilding.

Metadata wrapper Ops such as ModelVar/ModelPotential are not executable scientific
transformations. Their presence in a graph does not authorize custom Op reuse.

```python
from pymc.model.transform import extract_deterministics, insert_deterministics
without_label, detached = extract_deterministics(model, "mu")
restored = insert_deterministics(without_label, detached)
```

Extraction inlines computation in descendants rather than erasing its effect.
Detached graphs are topologically ordered; insertion resolves matching names,
types and meaning. Check density and restored deterministic values. Do not pass
wrapper graphs directly to a sampler.

## Value-coordinate transformations

Removing a transform leaves scientific support unchanged. For q=logit(p),
`log_density_q=log_density_p+log(p)+log(1-p)`; MAP/Hessian comparisons must respect
coordinate measure. Transform-selection keys must identify free RVs.

Utilities in `pymc.model.transform_values` consume a Dataset, not the whole
DataTree. For example use `idata["posterior"].to_dataset()`. `unconstrain_values`
uses natural free-RV names; `constrain_values` uses value-variable names from
`model.rvs_to_values`. Supply every free RV and explicit sample dims (default
chain/draw). Sample coordinates are preserved, but transformed core dimensions,
such as simplex reduction, can change names/sizes. Do not promise arbitrary
coordinate identity or infer transform names from legacy suffix conventions.

## Freeze, prune and remove minibatches

`freeze_dims_and_data(model,dims=None,data=None)` fixes selected existing data/dims:
None means all, [] none. New additions are not automatically frozen; fixed data
cannot be updated through set_data. In 6.3.1 constant/strategy initvals are supported
but symbolic initvals are not. Check density before relying on static conversion.

`freeze_model` returns a FrozenModel with restricted mutation/caching. It freezes
data needed by free RVs; data used only by likelihoods/deterministics can remain
settable. RNG-detaching backends can have fresh-compilation exceptions; caching is
not a universal speed guarantee.

`prune_vars_detached_from_observed` retains observation ancestors, not arbitrary
marginalization of connected latent variables. With no observations it may remove
the desired outcome. Check `do(...,prune_vars=True)` carefully. Models with
Potentials have an ambiguous prior/likelihood distinction and are unsupported by
this pruning route in 6.3.1.

`remove_minibatched_nodes` restores full inputs for evaluation/prediction. It is not
a new fit or correction to a biased approximation. Recheck `total_size` scaling,
row order and dimensions. Ordinary NUTS still needs a deterministic full-data target.

Sources: [model transforms](https://github.com/pymc-devs/pymc/tree/v6.3.1/pymc/model/transform),
[graph conversion](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/model/fgraph.py),
[value conversion](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/model/transform_values.py),
[forward sampling](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/sampling/forward.py),
[Causal Inference: What If](https://miguelhernan.org/whatifbook).
