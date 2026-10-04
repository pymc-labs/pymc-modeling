# Troubleshooting PyMC models

Start with the failing phase and the original warning or traceback. Preserve
caches, warnings and raw results while diagnosing; deleting caches or changing
seeds can hide evidence without repairing the cause.

## Choose the first action

| Symptom | First action | Existing detail |
|---|---|---|
| Failed initial evaluation or nonfinite log density | Check observations against support; inspect `model.initial_point()`, `model.point_logps()` and `model.debug()` to locate invalid factors. | [Inspect the model](workflow.md#inspect-the-model) |
| Shape, dtype or observation mismatch | Check tensor rank, event/batch dimensions, ordered row identities and input contracts before casting or resizing. | [Construct and inspect the graph](model-data-dimensions.md#construct-and-inspect-the-graph); [data and observation identity](model-data-dimensions.md#data-and-observation-identity) |
| Sampling appears stuck or fails during adaptation | Distinguish compilation, initialization, adaptation and retained sampling using actual timing, warnings and tracebacks. | [Budgets and failure diagnosis](sampling.md#budgets-diagnostics-and-failure-diagnosis) |
| Divergences or weak exploration | Inspect geometry, scaling and identification; choose a parameterization and budget suited to the target rather than treating more draws as a universal fix. | [Sampling diagnostics](sampling.md#budgets-diagnostics-and-failure-diagnosis); [centered and non-centered forms](hierarchical.md#centered-and-non-centered-forms) |
| Prediction or persistence warnings | Check predictive conditioning, coordinated data/coordinate updates and separate raw/enriched storage. | [Preserve conditioning](predictive-checks.md#preserve-conditioning); [storage and conversion](sampling.md#storage-and-conversion) |

For deeper raw symbolic, compiler or Op problems, use `pytensor-workflows` when
available. For analysis of a completed fit, use `arviz-diagnostics`. The local
routes above remain usable when this skill is copied independently.

## Interpret failures instead of suppressing them

| Signal | Investigate |
|---|---|
| `ShapeError`, `ShapeWarning`, `DtypeError` | Alignment, dimensions and numeric representation. |
| `IncorrectArgumentsError`, `NotConstantValueError` | Argument contract or an unjustified graph-constant assumption. |
| `ImputationWarning` | Missing-data mechanism and its implied latent variables. |
| `ImplicitFreezeWarning` | Changed ancestors and the intended predictive conditioning; choose `sample_vars`/`freeze_vars` deliberately. |
| `UndefinedMomentException` | Missing moment/support-point implementation, not necessarily impossible inference. |
| `TruncationError` | Truncated random generation and its numerical limits. |
| `TraceDirectoryError` | Storage setup and recovery. |
| `BlockModelAccessError` | Prohibited context access during graph construction. |

Catch only failures the application can interpret, preserving causes.
`drop_warning_stat` can remove object-valued warning fields incompatible with
serialization. Save/report warning content first; deleting the field does not
repair the scientific cause. Configure supported sampling progress options rather
than manipulating renderer internals.

Sources: [exceptions](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/exceptions.py),
[warning serialization helper](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/util.py).
