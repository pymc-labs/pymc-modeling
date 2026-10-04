# Graph compilation, state, debugging, and profiling

## Inspect before changing the graph

State the intended calculation and its input/output shape, dtype, domain, and
state contracts. For example, half the squared residual norm is
`0.5 * sum((measurements - center)**2)`; graph transformations should preserve
that expression's values and derivatives unless a semantic change is intended.

API cautions below refer to PyTensor 3.3.0 and PyMC 6.3.1. Check the linked
official source against the installed version before relying on a signature,
mode default, or backend limitation.

```python
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.printing import debugprint

measurements = pt.vector("measurements", dtype="float64")
center = pt.scalar("center", dtype="float64")
energy = 0.5 * pt.sqr(measurements - center).sum()
debugprint(energy, print_type=True)
evaluate = pytensor.function(
    [measurements, center], energy, mode=Mode(linker="py", optimizer="fast_run")
)
debugprint(evaluate, print_type=True)
```

## Graph and type metadata

A Variable is a symbolic value, not its runtime NumPy array. Tensor, scalar,
sparse, typed-list, random and named-dimension variables share graph interfaces,
but their storage and shape semantics are not interchangeable.

| Interface family | Use and boundary |
| --- | --- |
| `Variable.type`, `name`, `auto_name`, `owner`, `index`, `tag`; `Node.name` | Inspect the type and producing expression. A leaf has no owner; an owned value is `v.owner.outputs[v.index]`. Names are labels, not identity. Printing IDs are not stable serialization keys. `tag` stores annotations, not current numerical values. |
| `Apply` / `AbstractApply`: `op`, `inputs`, `outputs`, `default_output`, `get_parents` | An Apply binds an Op to specific input/output Variables. Reading these edges answers which expression produced a value. An Op may have multiple outputs; do not assume index zero or a particular topological position. |
| `Constant.data`, `AtomicVariable`, `Constant` and scalar/tensor/sparse/typed-list constant subclasses | Constants are graph leaves with stored values and do not belong in the declared input list. Use the array/type contract of the subclass; do not mutate constant data to update a model. Use shared state instead. |
| `AtomicVariable.equals`, `signature`, `merge_signature`; `Constant.signature`; `NominalVariable.id` | These support structural matching, merging, and positional inputs in immutable inner graphs. They are not numerical equality tests, probability identities, or user parameter names. `equal_computations` compares graph structure under an input mapping; independent numerical references remain necessary. |
| `add_tag_trace`, `get_variable_trace_string`, `Scratchpad.info` | Creation traces can locate the user expression responsible for an error. Traces may contain paths or data-dependent source context; review them before sharing. `Scratchpad.clear` removes annotations, not graph dependencies. Trace extraction and scratchpad implementation helpers are not a reason to rewrite runtime inputs. |

The `owner_op` and `owner_op_and_inputs` properties are convenient views of the
same producer edges. Apply's `nin`/`nout` count inputs/outputs and `out` follows
its default-output convention; `params_type` describes Op parameter typing, not
a numerical model parameter. Shared tensor `dtype`, `ndim`, `shape`, `size`, and
`broadcastable` expose the tensor contract. Its `T`/`mT` and `real`/`imag` produce
symbolic tensor transformations, not independently copied shared storage.

Use `debugprint(expr, print_type=True)` before compilation, then
`debugprint(compiled_function, print_type=True)` afterwards. Rewrites can merge,
remove, fuse, or replace Ops, so inspect the graph actually executed.
`file="str", id_type="int"` captures graph text without process-memory addresses.
`v.dprint`, `Function.dprint`, and `FunctionGraph.dprint` offer the same workflow.
`pprint` provides mathematical notation, not complete type/storage information.

## Cloning and conversion

| Operation | Meaning and correct use |
| --- | --- |
| `v.clone(name=...)` | On a normal Variable, create a new unowned leaf of the same type; **not** a copy of the expression that produced `v`. Constant cloning may return the same immutable object. SharedVariable cloning shares its storage container. |
| `TensorType.clone`, `SliceType.clone`, `XTensorType.clone` | Clone type metadata. This neither copies data nor casts numerical arrays. Named dimensions and positional dimensions are different contracts. A changed type must remain valid for every downstream Op. |
| `clone(inputs, outputs)`, `clone_get_equiv` | Copy a bounded graph. Explicitly choose whether graph inputs/orphans are copied. The equivalence map is the safe way to find a corresponding intermediate in the clone. `Apply.clone` retains input Variables; `clone_with_new_inputs` rebuilds one Apply and must satisfy its Op's type requirements. |
| `graph_replace(outputs, mapping, strict=True)` | Replace subgraphs while preserving unaffected pieces where possible; strict mode rejects unused replacements. It does not assert that replacements have the same mathematical meaning. Use it for explicit model transformations only after declaring their meaning. |
| `clone_replace([output], mapping)` | Clone with substitutions. Import from `pytensor.graph.replace`; do not assume the root module exports it. Supplying a list makes the output contract explicit. |
| `rewrite_graph(expr, include=("canonicalize",), clone=True)` | Apply a selected rewrite collection to a clone. The default `clone=False` can mutate a supplied FunctionGraph. Algebraic rewrites still need domain-, dtype-, and tolerance-aware checks; floating-point arithmetic is not exact algebra. |
| `givens={old: new}` to `pytensor.function` | Compile-time substitution, not a runtime update. Default `rebuild_strict=True` protects type compatibility. Interdependent substitutions are unsafe; use an explicit ordered transformation when one replacement depends on another. |
| `vectorize_graph(outputs, replace={vector_input: matrix_input})` | Add batch dimensions without changing core dimensions. Compare a batched calculation with a row-wise reference. This differs from callable-oriented `pt.vectorize` and its core signature. |
| `Variable.eval({variable: value}, mode=...)` | Evaluate using current input values, compiling internally. Prefer an explicit reusable function for repeated work. Names may be ambiguous, so Variable keys are safest. No `tag.test_value` workflow is introduced. |

## Graph transformations and inner graphs

`graph_inputs` finds leaves including constants and shared state.
`explicit_graph_inputs` finds inputs that must be supplied by the caller.
`ancestors`, `vars_between`, `applys_between`, `truncated_graph_inputs`, and
`walk` support bounded traversal; blockers define the boundary, not a change in
mathematics. `io_toposort` orders Apply nodes by dependency. `get_var_by_name`
may return multiple matches; `apply_depends_on` asks about a dependency, not a
nonzero numerical derivative at one point. `general_toposort` is a generic
traversal interface, not a performance shortcut.

A `FunctionGraph` maintains `inputs`, `outputs`, `variables`, `apply_nodes`, and
`clients`. Use `get_clients`, `get_output_client`, `toposort`, `dprint`, and
`check_integrity` to inspect a separately cloned graph. Use `replace` or
`replace_all` for a deliberate transformation; direct assignments to an Apply's
input list bypass maintained client information. `import_missing=True` imports
new dependencies, but then the compiled input signature must supply them.
`add_input` / `add_output`, `remove_input` / `remove_output`, and
`import_var` / `import_node` alter the boundary; `remove_node` can remove more
than a single displayed line by pruning dependents. Keep these operations on an
owned mutable graph and re-establish a complete input signature afterwards.
Low-level client updates, feature callbacks, and destructive-rewrite orderings
are compiler-maintenance machinery, not an alternative graph-editing recipe.

`FunctionGraph.freeze()` produces a hashable `FrozenFunctionGraph`; its inputs
are nominal positional Variables, not the original named leaves.
`FrozenFunctionGraph.from_io` creates one from a graph boundary; `bind` substitutes
all root inputs to return fresh mutable output expressions, and `unfreeze`
returns a mutable FunctionGraph. `from_structural_inputs` lifts structurally
matching interior expressions into an input boundary, which differs from
identity-based substitution. It must still contain every required root.
`from_toposort` preserves an explicit execution order. Do not request
`dedup_nodes=True` on graphs containing inplace/destructive Ops: distinct buffers
can be semantically important.

`OpFromGraph` is useful for packaging a reusable symbolic component: declare
its input/output types, inspect `fgraph` and inner inputs/outputs, and decide
whether inlining is desired. `clone_with_inner_graph` creates a new component
with an independently supplied mutable/frozen graph; do not mutate an interned
`fgraph` in place. Compare the changed component against its intended reference
and check that the original graph retains its behavior.

Its `perform`, `infer_shape`, `pullback`, `pushforward`, and override hooks are
not arbitrary Python callbacks that automatically acquire backend support.
In PyTensor 3.3.0 the `OpFromGraph` constructor's `pullback` override takes
`(inputs, outputs, cotangents)`, while `pushforward` takes `(inputs, tangents)`,
unlike the three-argument `Op.pushforward` method. Check the installed builder
before authoring overrides. Implement the true chain rule, not a concealed
straight-through approximation.

`input_types`/`output_types` are established by the constructor's explicit graph
boundary. Inherited `itypes`/`otypes` are not a way to bypass that contract.
`destroy_map` and `view_map` declare permitted destruction/aliasing and must match
the actual operation; do not mark an Op as destructive to obtain an unmeasured
speedup. With immutable caller input and non-borrowed output, modifying a
returned array must not change the caller's input or a later call. Check this
observable guarantee for custom components.

### Shared values and compiled Function lifecycle

- `pytensor.shared(value)` selects a suitable typed shared constructor. Shared
  Variables are implicit inputs; passing one as a required input is an error.
  `get_value` and `set_value` read/update numerical storage visible to every
  compiled function using that Variable. Respect dtype, rank, and any fixed
  shape; an array of length one does not establish a universally fixed shape.
- `borrow=True` allows aliasing; it does not promise zero-copy or independent
  storage. A shared clone is **not** independent state.
  `TensorSharedVariable.zero` zeros tensor storage, not RNG or graph structure.
- `updates={state: next_state}` runs on each function call. Outputs describe
  the old state unless the output expression explicitly computes the next
  state. Multiple updates read the same old state: `{a: b, b: a}` swaps values
  rather than assigning sequentially.
- PyTensor 3.3.0 deprecates assigning `default_update` and using
  `no_default_updates`; prefer explicit mappings. RNG updates have a separate
  stochastic contract; deterministic accumulation does not establish it.
- `Function.copy(swap={old_shared: new_shared})` creates a callable using new
  state. A plain copy still shares shared Variables. `share_memory` concerns
  intermediate storage, not statistical independence, and copied callables
  must not be assumed thread-safe. Check independent state after an explicit swap.
- `Function.get_shared`, `maker.fgraph`, `profile`, `outputs`, and `name` support
  inspection. `free()` releases working storage, not a portable model artifact.
  Direct `input_storage`, `output_storage`, `vm`, `return_none`, `unpack_single`,
  and `trust_input` manipulation bypasses the public call contract; the last
  can skip essential checks and can lead to interpreter crashes.
- `In` / `SymbolicInput` define caller argument names, defaults, strictness,
  downcasting, mutability, borrowing, and updates; `Out` / `SymbolicOutput`
  define outputs and borrowing. Defaults and keyword names are calling
  conventions, not model parameter constraints. Strict tensor inputs require
  ndarrays: a strict float64 scalar input needs a zero-dimensional float64
  ndarray, not a Python or NumPy scalar object.

## Compilation modes and signature failures

A Mode combines a **rewrite selection** with a **linker/executor**. Mode names
alone do not establish hardware or numerical support. PyTensor 3.3.0 resolves
`FAST_RUN` through its configured linker, so inspect the selected linker rather
than assuming it always means C execution.

| Mode or family | Choice and boundary |
| --- | --- |
| `Mode(linker="py", optimizer=None)` | Pure-Python baseline without rewrites; abstract or backend-only Ops may lack Python implementations. |
| `FAST_COMPILE` | Python VM with limited rewrites; unstable algebra is not necessarily stabilized. |
| `Mode(linker="cvm", optimizer="fast_run")` | C VM with applicable C/Python thunks; this is not an all-C kernel guarantee. |
| `NUMBA` | Numba lowering; inspect unsupported nodes and object-mode fallback. The first call may include JIT work. |
| `Mode(linker="c", ...)` | Pure-C linker; every required Op needs compatible C support. |
| `JAX`, `PYTORCH`, `MLX` | Require corresponding optional packages and graph lowerings. Check dtype, AD, dynamic shape and device contracts separately. |
| `Mode.including`, `excluding`, `requiring`, `clone`; `RewriteDatabaseQuery` | Select rewrite tags locally. Optimization presets are not universal quality rankings; unsafe selections may remove domain assertions. |

An `UnusedInputError` means a declared input has no path to an output. Inspect
`explicit_graph_inputs` and the typed graph. Remove the input only when it is
truly irrelevant; if the scientific calculation needs it, restore its meaningful
contribution and update the independent reference. `on_unused_input="ignore"`,
disabled checks, or adding `0 * disconnected_input` conceal the mismatch.

Conversely, `MissingInputError` means a dependency is undeclared. Inspect graph
identity and cloning before adding a guessed variable name.

## Debugging and measured profiling

Choose the smallest graph and explicit numerical point that reproduces the
failure. Separate compilation errors, first-call/JIT costs, repeated execution,
and model adaptation or sampling. Capture the actual graph and numerical inputs
needed to reproduce the failure.

- `DebugMode` checks implementation/rewrite consistency and invalid values.
  `check_c_code=False` limits the check to Python implementations; C/Python
  parity, strides, inplace storage and lazy graphs require relevant cases.
- `NanGuardMode` locates NaN/Inf at runtime. Set `big_is_error` deliberately:
  a universal magnitude cutoff has no scientific meaning. A log density may
  legitimately be `-inf` outside support; finite values can still be ill-conditioned.
- `MonitorMode(pre_func=..., post_func=...)` calls
  `(fgraph, index, node, thunk)`. Thunk inputs/outputs are one-element storage
  containers; inspect post-call values to locate the first bad Op. MonitorMode
  owns its linker: do not assume callbacks transfer to JAX/Numba.
- `Print` is an identity-like Op with a printing side effect, useful for targeted
  runtime inspection. It can change optimization and timing. `pprint`,
  `debugprint`, `get_node_by_id`, `min_informative_str`, and `as_string` are
  representation tools; a display match is not a numerical equivalence proof.
- `ProfileStats(atexit_print=False, flag_time_thunks=True)` plus
  `pytensor.function(..., profile=profile)` captures a real profile. Its function,
  class/Op/node summaries distinguish compilation, rewriting, linking, wrapper,
  and thunk time. `reset` resets runtime counters, not historical compile cost.
  Class/Op aggregate helpers summarize the same measured node counters, not
  additional measurements. Memory summaries require memory profiling and
  representative shapes.
- For a performance question, use representative shapes, compiler/cache state,
  warmup, repeated trials, thread counts, process isolation, and uncertainty.
  Compare numerically before comparing runtime; report compile and steady-state
  costs separately. Never infer a speedup from fewer graph nodes alone.

`pydotprint`, `d3viz`, `d3write`, and `PyDotFormatter` provide optional
Graphviz/HTML visualization; install renderer dependencies only when needed.
For PyTensor 3.3.0, an explicit `FunctionGraph` with a materialized input list
avoids a raw-variable conversion path that can consume a single-pass input
iterator and raise `MissingInputError`; see the
[formatter source](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/d3viz/formatting.py).
Check the installed version before assuming that limitation persists.

Custom printers and `op_debug_information` can add helpful annotations.
Formatting must not change the graph or hide a non-identity operation.

`pytensor.compile.function_dump` serializes compilation arguments, including
shared values, using pickle and replaces its destination. Use a fresh owned
temporary path and minimized/synthetic inputs. Do not alter live shared state
to sanitize a dump; review sensitive contents before sharing, and never
unpickle an untrusted file.

## PyMC model-graph compiler families

Use `model.initial_point()` for the numerical mapping expected by a model's
compiled functions; transformed free variables may have transformed names and
values. An initial point is not a posterior draw. Model methods such as
`compile_logp` consume value-variable points; directly compiling an RV expression
can instead generate random draws, which is a different computation.

`pymc.pytensorf.compile` wraps `pytensor.function` with PyMC rewrites and collected
RNG updates. In PyMC 6.3.1 it can reseed by default;
`random_seed=False` separates compilation from reseeding.
`return_updates=True` returns the collected RNG mapping. Explicit `updates`
may override entries, so preserve each state transition when composing wrappers.
`replace_rng_nodes` may mutate its input graph: clone expression nodes before
detaching RNG state from a live graph. Check replay and independence, not just
the presence of an update mapping.

| Family | Practical use and boundary |
| --- | --- |
| `PointFunc`, `.f`, `.dprint`; `CallableTensor` | A PointFunc consumes a name-to-array mapping and delegates to its compiled function. CallableTensor substitutes a single explicit input and returns another expression, not a numeric evaluation. |
| `inputvars`, `cont_inputs`, `rvs_in_graph`, `expand_inner_graph` | Identify explicit, continuous, or random dependencies including inner graphs. Inspect before deciding whether a result is a log density or a stochastic draw. A lack of continuous inputs is not proof that a graph is constant. |
| `join_nonshared_inputs`, `make_shared_replacements` | Flatten selected inputs or freeze other model value variables into shared state. Preserve point shapes and order; changed shapes need a new mapping. Freezing is not marginalization. |
| `constant_fold`, `resolve_shapes`, `get_symbolic_rv_shapes` | Extract true constants or rewrite shape expressions without drawing RVs. `raise_not_constant=True` exposes nonconstant inputs; disabling it can return symbolic values with cloned inputs. |
| `gradient`, `jacobian`, `jacobian_diag`, `hessian`, `hessian_diag` | PyMC's wrappers flatten/stack inputs. In PyMC 6.3.1, `hessian(..., negate_output=True)` negates by default; choose the sign explicitly and compare analytic curvature. |
| `floatX`, `intX`, `smartfloatX`, `smarttypeX`, `largest_common_dtype` | Conversion helpers encode configured dtype choices; they do not guarantee a precision budget. `intX` follows PyMC's conversion map rather than meaning platform-default integer. Explicit dtype is preferable when a numerical tolerance depends on it. |
| `convert_data`, `convert_observed_data`, `extract_obs_data`, `dataframe_to_tensor_variable`, `ix_` | DataFrame/array/mask conversion and indexing utilities used at model graph boundaries. Missingness and support need a declared data model; extraction from arbitrary symbolic expressions can fail. The pandas tensor-conversion registration hook itself is not a distinct user compiler capability. |
| `replace_vars_in_graphs`, `toposort_replace`, `rewrite_pregrad` | Preserve substitution boundaries and order. `toposort_replace` mutates an owned FunctionGraph; pre-gradient stabilization needs domain-aware numerical checks. |
| `collect_default_updates`, `collect_default_updates_inner_fgraph`, `find_rng_nodes`, `replace_rng_nodes`, `reseed_rngs`, `normalize_rng_param` | Discover, detach and reseed RNG state, including inner graphs. Check actual state progression; repeated seeds do not establish independent simulation. |
| `resolve_backend_compile_kwargs` | Materializes the backend shortcut into a new kwargs dictionary. Supplying both backend and explicit mode is an error. PyMC's `"c"` shortcut maps to CVM, unlike PyTensor's pure-C linker. Never change a likelihood just to accommodate a compiler. |

## Configuration

Inspect current registered settings and the pinned source before changing them;
use `config.change_flags(...)` for allowed temporary settings and process-start
`PYTENSOR_FLAGS` for settings that cannot change after initialization. Record
`floatX`, mode/linker/rewriter selection, compiler availability, and thread-related
settings relevant to a comparison. Do not dump all configuration, compiler paths,
or user data into a shareable report.

- `exception_verbosity`, traceback limits, and stack-trace checking control the
  quality/cost of error evidence. `on_unused_input`, `trust_input`, unsafe rewrite
  selections, and removed assertions are not repairs for a defective graph.
- `profile`, `profile_memory`, `profile_optimizer`, `print_global_stats`, and
  `profiling__*` control measurement scope, report size, destination, and first-call
  accounting. `DebugMode__*` and `NanGuardMode__*` configure diagnostics, not
  universal scientific pass criteria.
- `allow_gc`, scan allocation/GC settings, `vm__lazy`, OpenMP, C compiler flags,
  Numba caching/fastmath, and compiledir/cache settings trade resources or
  numerical guarantees. They need workload-specific measurements. Do not delete
  shared caches or force-unlock an active compiler to conceal a stall.
- Invoke rewrite collections through Mode or `rewrite_graph`, rather than
  individual allocation, fusion, Scan or inplace compiler passes. Direct
  registry mutation and storage/alias bookkeeping are compiler development,
  not fixes for a model's graph.

## Authoritative sources

These version-pinned sources describe the API cautions above; inspect matching
installed code when documentation or behavior differs.

- [PyTensor release compilation entry and contract](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/maker.py),
  [release modes](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/mode.py).
- [Release Variable/Apply and cloning source](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/graph/basic.py),
  [graph replacement](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/graph/replace.py),
  [mutable/frozen FunctionGraph](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/graph/fg.py),
  [traversal](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/graph/traversal.py).
- [Release shared state](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/sharedvalue.py),
  [Function lifecycle/copy](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/executor.py),
  [OpFromGraph](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/builders.py).
- [Release profiler](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/debug/profiling.py),
  [MonitorMode](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/debug/monitormode.py),
  [DebugMode](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/debug/debugmode.py),
  [printing](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/printing.py).
- [PyMC 6.3.1 graph helpers](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/pytensorf.py),
  [model initial points and compilation](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/model/core.py).

