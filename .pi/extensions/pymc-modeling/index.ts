/**
 * PyMC Modeling Extension for pi
 *
 * Ports the pymc-modeling Claude Code plugin to pi:
 * - Custom tools: pymc_api_lookup, pymc_example_search, pymc_error_lookup
 * - Commands: /pymc-diagnose, /prior-check, /shape-check, /model-compare
 * - Auto-discovers skills from the project's skills/ directory
 * - Injects PyMC 6 / ArviZ 1.0 stack reminders
 * - Lint-checks .py files for deprecated PyMC patterns after write/edit
 */

import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Type } from "typebox";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

/* -------------------------------------------------------------------------- */
/*  Data Types                                                                */
/* -------------------------------------------------------------------------- */

interface ApiEntry {
	name: string;
	signature: string;
	description: string;
	gotchas?: string[];
	see_also?: string[];
}

interface CodePattern {
	id: string;
	keywords: string[];
	title: string;
	code: string;
	explanation: string;
}

interface ErrorPattern {
	pattern: string;
	keywords: string[];
	title: string;
	fix: string;
}

/* -------------------------------------------------------------------------- */
/*  Data Loading                                                              */
/* -------------------------------------------------------------------------- */

const EXT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(EXT_DIR, "data");

function loadJson<T>(filename: string): T {
	return JSON.parse(fs.readFileSync(path.join(DATA_DIR, filename), "utf-8")) as T;
}

let _apiData: ApiEntry[] | null = null;
let _patternsData: CodePattern[] | null = null;

function getApiData(): ApiEntry[] {
	if (!_apiData) {
		_apiData = [...loadJson<ApiEntry[]>("pymc_api.json"), ...loadJson<ApiEntry[]>("arviz_api.json")];
	}
	return _apiData!;
}

function getPatternsData(): CodePattern[] {
	if (!_patternsData) {
		_patternsData = loadJson<CodePattern[]>("patterns.json");
	}
	return _patternsData!;
}

const ERROR_PATTERNS: ErrorPattern[] = [
	{
		pattern: "divergence",
		keywords: ["divergence", "divergent", "diverging"],
		title: "Divergent transitions after tuning",
		fix:
			"1. Try non-centered parameterization for hierarchical models\n" +
			"2. Increase target_accept: pm.sample(target_accept=0.95)\n" +
			"3. Increase tune steps: pm.sample(tune=2000)\n" +
			"4. Reparameterize: use offset + scale instead of direct sampling\n" +
			"5. Check az.plot_pair(idata, divergences=True) to identify problematic parameters",
	},
	{
		pattern: "bad_energy",
		keywords: ["energy", "BFMI", "low BFMI", "bad energy"],
		title: "Low Bayesian Fraction of Missing Information (BFMI)",
		fix:
			"1. Check az.plot_energy(idata) — gap between distributions indicates problems\n" +
			"2. Reparameterize the model (non-centered parameterization)\n" +
			"3. Use more informative priors to constrain the posterior",
	},
	{
		pattern: "rhat",
		keywords: ["r_hat", "rhat", "convergence", "not converged", "1.05", "1.1"],
		title: "R-hat values above 1.01",
		fix:
			"1. Run more chains and draws: pm.sample(draws=2000, tune=2000, chains=4)\n" +
			"2. Check for multimodality in the posterior\n" +
			"3. Use more informative priors\n" +
			"4. Check az.plot_trace_dist(idata) for chain mixing issues\n" +
			"5. Try different initialization: pm.sample(initvals=...)",
	},
	{
		pattern: "shape_mismatch",
		keywords: ["shape", "dimension", "mismatch", "broadcast", "ValueError", "shape mismatch"],
		title: "Shape/dimension mismatch errors",
		fix:
			"1. Use coords and dims consistently\n" +
			"2. Check that observed data shape matches the model's expected shape\n" +
			"3. For matrix operations, verify dimensions with pm.math.dot\n" +
			"4. Use pm.model_to_graphviz(model) to visualize the model structure",
	},
	{
		pattern: "sampling_error",
		keywords: ["SamplingError", "sampling error", "bad initial energy", "initial point"],
		title: "SamplingError / bad initial energy",
		fix:
			"1. Check for invalid prior values (e.g., negative scale parameters)\n" +
			"2. Set explicit initvals: pm.sample(initvals={'param': value})\n" +
			"3. Verify observed data doesn't contain NaN or inf\n" +
			"4. Scale your data (standardize predictors)\n" +
			"5. Use pm.find_MAP() first to find a good starting point",
	},
	{
		pattern: "nutpie_log_likelihood",
		keywords: ["nutpie", "log_likelihood", "loo", "waic", "compute_log_likelihood"],
		title: "Missing log_likelihood with nutpie sampler",
		fix:
			"nutpie doesn't store log_likelihood automatically.\n" +
			"After sampling, call:\n" +
			"  pm.compute_log_likelihood(idata)\n" +
			"Then you can use az.loo(idata) or az.compare().",
	},
	{
		pattern: "label_switching",
		keywords: ["label switching", "label swap", "mixture", "identifiability", "multimodal"],
		title: "Label switching in mixture models",
		fix:
			"1. Apply ordered transform to component means:\n" +
			"   mu = pm.Normal('mu', 0, 10, dims='component',\n" +
			"                  transform=pm.distributions.transforms.ordered)\n" +
			"2. Use ordered(Dirichlet) for mixture weights\n" +
			"3. Set informative initvals for component locations\n" +
			"4. Consider using pm.Marginalized for discrete latent variables",
	},
	{
		pattern: "memory_error",
		keywords: ["memory", "MemoryError", "OOM", "out of memory", "killed"],
		title: "Out of memory during sampling",
		fix:
			"1. Reduce number of draws or chains\n" +
			"2. Use nutpie which is more memory-efficient: nuts_sampler='nutpie'\n" +
			"3. For GPs, use HSGP approximation instead of full GP\n" +
			"4. For large datasets, subsample or use minibatch approaches\n" +
			'5. Avoid storing unnecessary groups: idata["posterior"] only',
	},
	{
		pattern: "dims_conflict",
		keywords: ["dims", "coords", "dimension", "conflict", "cutpoints", "already exists"],
		title: "Dimension name conflicts",
		fix:
			"1. Don't use the same name for both a variable and a dimension\n" +
			"   BAD:  cutpoints = pm.Normal('cutpoints', dims='cutpoints')\n" +
			"   GOOD: cutpoints = pm.Normal('kappa', dims='cutpoint_dim')\n" +
			"2. Define all coords before the pm.Model context\n" +
			"3. Check that dim names don't clash with PyMC internals",
	},
	{
		pattern: "theano_pytensor",
		keywords: ["theano", "aesara", "pytensor", "import error", "module not found"],
		title: "Theano/Aesara/PyTensor import errors",
		fix:
			"PyMC 6+ uses PyTensor 3+ (not Theano or Aesara).\n" +
			"Replace:\n" +
			"  import theano.tensor as tt  ->  import pytensor.tensor as pt\n" +
			"  import aesara.tensor as at  ->  import pytensor.tensor as pt\n" +
			"Math operations: pt.dot, pt.exp, pt.log, pt.switch, etc.\n" +
			"PyTensor 3 removals to watch for: tag.test_value and " +
			"compute_test_value are gone (call the .eval method on a " +
			"symbolic variable with a point dict instead); Op.L_op and " +
			"Op.R_op were renamed to Op.pull_back and Op.push_forward.",
	},
	{
		pattern: "arviz_datatree",
		keywords: ["DataTree", "datatree", "ArviZ", "1.0", "groups", "xarray", "InferenceData"],
		title: "ArviZ 1.0 DataTree migration",
		fix:
			"ArviZ 1.0 removed InferenceData entirely; pm.sample() now returns xarray.DataTree.\n" +
			"Attribute access is gone — always use bracket access:\n" +
			'  dt["posterior"]       (was idata.posterior)\n' +
			'  dt["sample_stats"]    (not attribute access)\n' +
			'  dt.children.keys()     (was idata.groups())\n' +
			'  "posterior" in dt     (was hasattr(idata, "posterior"))\n' +
			'Default CI: ci_prob=0.89, ci_kind="eti" (was hdi_prob=0.94).\n' +
			"az.waic is removed — use az.loo. New LOO helpers: az.loo_expectations, " +
			"az.loo_metrics, az.loo_r2. Accessor after `import arviz_stats`: " +
			"ds.azstats.summary() / .rhat() / .ess() / .loo().",
	},
	{
		pattern: "slow_sampling",
		keywords: ["slow", "takes long", "performance", "speed", "hours"],
		title: "Slow MCMC sampling",
		fix:
			"1. Use nutpie: pm.sample(nuts_sampler='nutpie') — 2-5x faster\n" +
			"2. Reduce data size if possible (subsample for prototyping)\n" +
			"3. Use HSGP instead of full GP for n > 500\n" +
			"4. Non-centered parameterization reduces geometry complexity\n" +
			"5. Standardize predictors to improve sampler efficiency\n" +
			"6. Consider Laplace approximation for quick exploration",
	},
];

/* -------------------------------------------------------------------------- */
/*  Search Helpers                                                            */
/* -------------------------------------------------------------------------- */

function keywordScore(query: string, keywords: string[]): number {
	const q = query.toLowerCase();
	return keywords.reduce((acc, kw) => acc + (q.includes(kw.toLowerCase()) ? 1 : 0), 0);
}

function textMatch(query: string, text: string): boolean {
	const words = query.toLowerCase().split(/\s+/);
	const t = text.toLowerCase();
	return words.some((w) => t.includes(w));
}

function formatApiEntry(entry: ApiEntry): string {
	const parts = [
		`## ${entry.name}`,
		"```python",
		entry.signature,
		"```",
		entry.description,
	];
	if (entry.gotchas?.length) {
		parts.push("\n**Gotchas:**", ...entry.gotchas.map((g) => `- ${g}`));
	}
	if (entry.see_also?.length) {
		parts.push(`\n**See also:** ${entry.see_also.join(", ")}`);
	}
	return parts.join("\n");
}

function formatPattern(pattern: CodePattern): string {
	return [
		`## ${pattern.title}`,
		"```python",
		pattern.code,
		"```",
		pattern.explanation,
		`*Keywords: ${pattern.keywords.join(", ")}*`,
	].join("\n");
}

function formatError(err: ErrorPattern): string {
	return [`## ${err.title}`, "", err.fix].join("\n");
}

/* -------------------------------------------------------------------------- */
/*  PyMC Keyword Detection                                                    */
/* -------------------------------------------------------------------------- */

const PYMC_KEYWORDS = [
	"pymc", "pm.sample", "pm.Model", "mcmc", "bayesian", "posterior", "prior",
	"arviz", "az.plot", "az.summary", "inference", "nuts", "nutpie", "pytensor",
	"gaussian process", "hierarchical", "multilevel", "mixture", "bart",
	"divergence", "r_hat", "ess", "loo", "waic", "elpd", "predictive",
	"log_likelihood", "coords", "dims", "non-centered", "parameterization",
	"convergence", "diagnostics", "effective sample size",
];

function isPymcRelated(text: string): boolean {
	const lower = text.toLowerCase();
	return PYMC_KEYWORDS.some((kw) => lower.includes(kw.toLowerCase()));
}

/* -------------------------------------------------------------------------- */
/*  Lint Helpers                                                              */
/* -------------------------------------------------------------------------- */

interface LintIssue {
	pattern: string;
	message: string;
	severity: "warning" | "error";
}

function lintPymcCode(content: string): LintIssue[] {
	const issues: LintIssue[] = [];
	const lines = content.split("\n");

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const lineNum = i + 1;

		// Attribute access on DataTree (ArviZ 1.0 breaking change)
		if (/idata\.\w+\s*[^[=]/.test(line) && !line.includes("#")) {
			issues.push({
				pattern: "idata.attr",
				message: `Line ${lineNum}: Use bracket access idata["group"] instead of attribute access idata.group (ArviZ 1.0 / DataTree)`,
				severity: "error",
			});
		}

		// az.waic (removed in ArviZ 1.0)
		if (/az\.waic\b/.test(line)) {
			issues.push({
				pattern: "az.waic",
				message: `Line ${lineNum}: az.waic is removed in ArviZ 1.0 — use az.loo instead`,
				severity: "error",
			});
		}

		// compute_log_likelihood=True in pm.sample (FutureWarning in PyMC 6)
		if (/pm\.sample\(.*compute_log_likelihood\s*=\s*True/.test(line)) {
			issues.push({
				pattern: "compute_log_likelihood=True",
				message: `Line ${lineNum}: Passing compute_log_likelihood=True to pm.sample emits a FutureWarning in PyMC 6. Call pm.compute_log_likelihood(idata, model=model) explicitly after sampling.`,
				severity: "warning",
			});
		}

		// tag.test_value (removed in PyTensor 3)
		if (/\.tag\.test_value/.test(line) || /compute_test_value/.test(line)) {
			issues.push({
				pattern: "tag.test_value",
				message: `Line ${lineNum}: tag.test_value and compute_test_value are removed in PyTensor 3. Use .eval() on a symbolic variable with a point dict instead.`,
				severity: "error",
			});
		}

		// samples= in sample_prior_predictive (renamed to draws= in PyMC 6)
		if (/sample_prior_predictive\(.*samples\s*=/.test(line)) {
			issues.push({
				pattern: "samples=",
				message: `Line ${lineNum}: sample_prior_predictive argument is draws= (not samples=) in PyMC 6.`,
				severity: "error",
			});
		}

		// pm.compile_pymc (renamed to pm.compile in PyMC 6)
		if (/pm\.compile_pymc\b/.test(line)) {
			issues.push({
				pattern: "pm.compile_pymc",
				message: `Line ${lineNum}: pm.compile_pymc is renamed to pm.compile in PyMC 6.`,
				severity: "warning",
			});
		}

		// Theano/Aesara imports
		if (/import\s+(theano|aesara)(\.tensor)?/.test(line)) {
			issues.push({
				pattern: "theano/aesara import",
				message: `Line ${lineNum}: PyMC 6+ uses PyTensor 3+. Replace 'import theano.tensor as tt' or 'import aesara.tensor as at' with 'import pytensor.tensor as pt'.`,
				severity: "error",
			});
		}

		// Op.L_op / Op.R_op (renamed in PyTensor 3)
		if (/Op\.(L_op|R_op)\b/.test(line)) {
			issues.push({
				pattern: "Op.L_op/R_op",
				message: `Line ${lineNum}: Op.L_op and Op.R_op are renamed to Op.pull_back and Op.push_forward in PyTensor 3.`,
				severity: "error",
			});
		}

		// convert_to_inference_data (renamed in ArviZ 1.0)
		if (/convert_to_inference_data/.test(line)) {
			issues.push({
				pattern: "convert_to_inference_data",
				message: `Line ${lineNum}: convert_to_inference_data is renamed to az.convert_to_datatree in ArviZ 1.0.`,
				severity: "error",
			});
		}
	}

	return issues;
}

/* -------------------------------------------------------------------------- */
/*  Command Content                                                           */
/* -------------------------------------------------------------------------- */

const CMD_PRIOR_CHECK = `Generate and analyze prior predictive checks for a PyMC model.

1. Find the PyMC model definition. Look in the current file or ask the user to specify which file contains the model:

Search for files containing "pm.Model" or "pymc.Model" in the working directory.

2. Read the model file and identify the model context block (with pm.Model(...) as model:).

3. Generate code to run prior predictive sampling:

\`\`\`python
import pymc as pm
import arviz as az
import numpy as np

# [Insert the model definition code here]

with model:
    prior_pred = pm.sample_prior_predictive(draws=500, random_seed=42)
\`\`\`

4. Create prior predictive plots:

\`\`\`python
# Prior predictive check plot
az.plot_ppc_dist(prior_pred, group="prior_predictive", kind="ecdf")

# Plot prior distributions for key parameters
az.plot_dist(prior_pred["prior"]["PARAM_NAME"].values.flatten())
\`\`\`

5. Analyze the prior predictions:

\`\`\`python
# Get the observed variable name from the model
obs_var = [v.name for v in model.observed_RVs][0]

prior_y = prior_pred["prior_predictive"][obs_var].values
print(f"Prior predictive range: [{np.min(prior_y):.4f}, {np.max(prior_y):.4f}]")
print(f"Prior predictive mean: {np.mean(prior_y):.4f}")
print(f"Prior predictive std: {np.std(prior_y):.4f}")
print(f"Prior predictive median: {np.median(prior_y):.4f}")

# Check for extreme values
q01, q99 = np.quantile(prior_y, [0.01, 0.99])
print(f"Prior predictive 1%-99% range: [{q01:.4f}, {q99:.4f}]")
\`\`\`

6. Report findings:
   - Are prior predictions in a plausible range for the observed data?
   - Are there extreme or impossible values (negative counts, probabilities > 1, etc.)?
   - Do the prior predictions span the range of observed data?
   - Suggest specific prior adjustments if issues are found (tighten wide priors, change distribution family, adjust hyperparameters).`;

const CMD_SHAPE_CHECK = `Validate PyMC model shapes and dimensions before committing to a full sampling run.

1. Find the PyMC model definition. Look in the current file or ask the user to specify:

Search for files containing "pm.Model" or "pymc.Model" in the working directory.

2. Read the model file and identify the model context block.

3. Run model.debug() to check for shape and value issues:

\`\`\`python
import pymc as pm
import numpy as np

# [Insert the model definition code here]

# Debug check: reports shape mismatches, invalid parameter values, etc.
model.debug()
\`\`\`

4. Run a fast prior predictive check with minimal draws to catch broadcasting errors:

\`\`\`python
with model:
    # Single draw is enough to catch shape/broadcasting errors
    try:
        prior_test = pm.sample_prior_predictive(draws=1, random_seed=42)
        print("Prior predictive sampling: OK")
    except Exception as e:
        print(f"Shape/broadcasting error detected: {e}")
\`\`\`

5. Check coords and dims consistency:

\`\`\`python
# Verify coords match data dimensions
print("Model coords:")
for dim_name, coord_vals in model.coords.items():
    print(f"  {dim_name}: {len(coord_vals)} values")

# Check each variable's dims
print("\\nVariable dimensions:")
for rv in model.free_RVs + model.observed_RVs:
    dims = model.named_vars_to_dims.get(rv.name, "none")
    print(f"  {rv.name}: dims={dims}")
\`\`\`

6. Check for common shape issues:
- Mismatched coord lengths: Do coordinate arrays match the corresponding data dimensions?
- Index vector bounds: Are index vectors within the valid range for their target parameter?
- Design matrix shape: Is X shape (N, K) and not (K, N)?
- Broadcasting traps: Will (J,) and (N,) shapes broadcast to (N, J) unexpectedly?
- MutableData compatibility: If using pm.set_data(), are shapes compatible?

7. Report results:
   - List any shape mismatches with the specific variable, expected shape, and actual shape.
   - For each issue, provide the fix (corrected dims, proper indexing, transpose, etc.).
   - If no issues found, confirm that shapes and dimensions are consistent.`;

const CMD_MODEL_COMPARE = `Compare two or more Bayesian models using PSIS-LOO-CV (Pareto-Smoothed Importance Sampling Leave-One-Out Cross-Validation).

1. Find model result files. Look for .nc (NetCDF) files in the current directory, or ask the user to specify 2 or more result files:

Glob for **/*.nc files in the working directory.

2. Load each result file using ArviZ 1.0 DataTree API:

\`\`\`python
import arviz as az
import pymc as pm

# Load results
model_results = {}
for name, path in MODEL_FILES.items():
    dt = az.from_netcdf(path)
    model_results[name] = dt
    print(f"Loaded {name}: groups = {list(dt.children.keys())}")
\`\`\`

3. Ensure log-likelihood is available for each model. If missing, compute it:

\`\`\`python
for name, dt in model_results.items():
    if "log_likelihood" not in dt.children:
        print(f"WARNING: {name} is missing log_likelihood group.")
        print("Rerun with pm.compute_log_likelihood(dt, model=model) after pm.sample()")
\`\`\`

4. Run model comparison using LOO-CV:

\`\`\`python
# ArviZ 1.0: LOO is the only information criterion (WAIC is removed)
comparison = az.compare(model_results)
print(comparison)
\`\`\`

5. Generate comparison plot:

\`\`\`python
az.plot_compare(comparison)
\`\`\`

6. Interpret results:

\`\`\`python
print("\\n--- Model Comparison Interpretation ---")
print(f"\\nBest model: {comparison.index[0]}")
print("ELPD difference from best:")
for model_name in comparison.index[1:]:
    row = comparison.loc[model_name]
    elpd_diff = row["elpd_loo"] - comparison.iloc[0]["elpd_loo"]
    se_diff = row["dse"]
    print(f"  {model_name}: {elpd_diff:.1f} +/- {se_diff:.1f}")
    if abs(elpd_diff) < 2 * se_diff:
        print("    -> Not meaningfully different from best model")
    else:
        print("    -> Meaningfully worse than best model")

# Stacking weights
print("\\nStacking weights (for model averaging):")
for model_name in comparison.index:
    w = comparison.loc[model_name, "weight"]
    print(f"  {model_name}: {w:.3f}")

# Pareto k warnings
for name, dt in model_results.items():
    loo_result = az.loo(dt)
    n_bad = (loo_result.pareto_k > 0.7).sum().item()
    if n_bad > 0:
        print(f"\\nWARNING: {name} has {n_bad} observations with Pareto k > 0.7")
        print("  LOO estimates may be unreliable for this model")
\`\`\`

7. Provide a summary:
   - Rank models by ELPD (expected log pointwise predictive density)
   - Note whether differences are meaningful (ELPD difference > 2*SE)
   - Report stacking weights for model averaging
   - Flag any Pareto k warnings that make LOO estimates unreliable
   - Note: WAIC is not available in ArviZ 1.0; use PSIS-LOO-CV exclusively`;

const CMD_PYM_DIAGNOSE = `Run a comprehensive MCMC diagnostic analysis on sampling results.

1. First, find a results file. Look for .nc (NetCDF) files in the current directory or ask the user to specify one:

Glob for **/*.nc or **/results*.nc files in the working directory.

2. Load the results using ArviZ 1.0 DataTree API:

\`\`\`python
import arviz as az
import numpy as np

dt = az.from_netcdf("RESULTS_FILE_PATH")
print(f"Groups: {list(dt.children.keys())}")
\`\`\`

3. Run the full diagnostic workflow:

Divergences:
\`\`\`python
if "sample_stats" in dt.children:
    divergences = dt["sample_stats"]["diverging"].values
    n_div = divergences.sum()
    total = divergences.size
    print(f"Divergences: {n_div} / {total} ({100*n_div/total:.1f}%)")
\`\`\`

Summary statistics (R-hat, ESS):
\`\`\`python
# ArviZ 1.0 uses 0.89 ETI by default (not 0.94 HDI)
summary = az.summary(dt, var_names=["~log_likelihood"])
print(summary)

# Flag problematic parameters
rhat_bad = summary[summary["r_hat"] > 1.01]
ess_bulk_bad = summary[summary["ess_bulk"] < 400]
ess_tail_bad = summary[summary["ess_tail"] < 400]

if len(rhat_bad) > 0:
    print(f"\\nR-hat > 1.01 for: {list(rhat_bad.index)}")
if len(ess_bulk_bad) > 0:
    print(f"\\nLow ESS bulk (<400) for: {list(ess_bulk_bad.index)}")
if len(ess_tail_bad) > 0:
    print(f"\\nLow ESS tail (<400) for: {list(ess_tail_bad.index)}")
\`\`\`

Rank plot:
\`\`\`python
az.plot_rank(dt)
\`\`\`

Energy diagnostic:
\`\`\`python
if "sample_stats" in dt.children:
    energy = dt["sample_stats"]["energy"].values.flatten()
    e_bfmi = np.var(np.diff(energy)) / np.var(energy)
    print(f"E-BFMI: {e_bfmi:.3f} ({'OK' if e_bfmi > 0.3 else 'LOW - poor exploration'})")
\`\`\`

LOO-CV / Pareto k diagnostics (if log_likelihood exists):
\`\`\`python
if "log_likelihood" in dt.children:
    loo_result = az.loo(dt)
    print(loo_result)
    pareto_k = loo_result.pareto_k
    n_bad = (pareto_k > 0.7).sum().item()
    if n_bad > 0:
        print(f"\\n{n_bad} observations with Pareto k > 0.7 (unreliable LOO estimate)")
\`\`\`

Note: WAIC is not available in ArviZ 1.0. Use PSIS-LOO-CV exclusively.

4. Interpret all results together and provide:
   - A summary of overall sampling quality (good / acceptable / problematic)
   - Specific issues found, ordered by severity
   - Concrete remediation steps for each issue:
     - Divergences: suggest non-centered parameterization, increased target_accept, or prior adjustments
     - High R-hat: suggest more samples, check for multimodality
     - Low ESS: suggest reparameterization, more samples, or thinning
     - Bad Pareto k: suggest model revision for influential observations`;

/* -------------------------------------------------------------------------- */
/*  Extension Factory                                                         */
/* -------------------------------------------------------------------------- */

export default function pymcModelingExtension(pi: ExtensionAPI) {
	/* ------------------------------------------------------------------------ */
	/*  Resource Discovery                                                      */
	/* ------------------------------------------------------------------------ */

	pi.on("resources_discover", async (event, _ctx) => {
		const skillPath = path.join(event.cwd, "skills");
		if (!fs.existsSync(skillPath)) return;

		// Collect names of globally installed skills to avoid collisions
		const globalSkillDirs = [
			path.join(os.homedir(), ".agents", "skills"),
			path.join(os.homedir(), ".pi", "agent", "skills"),
		];
		const globalNames = new Set<string>();
		for (const dir of globalSkillDirs) {
			if (!fs.existsSync(dir)) continue;
			for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
				if (entry.isDirectory()) globalNames.add(entry.name);
				if (entry.isFile() && entry.name.endsWith(".md")) {
					globalNames.add(entry.name.replace(/\.md$/, ""));
				}
			}
		}

		// Only contribute local skill subdirectories that don't collide with globals
		const skillPaths: string[] = [];
		for (const entry of fs.readdirSync(skillPath, { withFileTypes: true })) {
			if (entry.isDirectory() && !globalNames.has(entry.name)) {
				skillPaths.push(path.join(skillPath, entry.name));
			}
		}

		if (skillPaths.length > 0) {
			return { skillPaths };
		}
	});

	/* ------------------------------------------------------------------------ */
	/*  Context Injection                                                       */
	/* ------------------------------------------------------------------------ */

	pi.on("before_agent_start", async (event, _ctx) => {
		if (!isPymcRelated(event.prompt)) return;

		const extra = `
## PyMC Modeling Assistant Active

You are working with PyMC 6+, PyTensor 3+, and ArviZ 1.0+. Critical API reminders:
- pm.sample() returns xarray.DataTree — access groups with brackets: idata["posterior"], NOT idata.posterior
- Call pm.compute_log_likelihood(idata, model=model) explicitly after sampling (do NOT pass compute_log_likelihood=True to pm.sample)
- az.waic is removed — use az.loo exclusively
- Default CI is 0.89 ETI (eti_5.5% / eti_94.5%), not 0.94 HDI
- sample_prior_predictive(draws=N) not samples=N
- Prefer nutpie sampler: pm.sample(nuts_sampler="nutpie")
- Use non-centered parameterization for hierarchical models
- tag.test_value is gone in PyTensor 3 — use .eval() with a point dict
`;
		return { systemPrompt: event.systemPrompt + extra };
	});

	/* ------------------------------------------------------------------------ */
	/*  Post-Read PyMC Import Detection                                         */
	/* ------------------------------------------------------------------------ */

	pi.on("tool_result", async (event, ctx) => {
		if (event.toolName !== "read") return;
		if (event.isError) return;

		const filePath = (event.input as { path?: string }).path;
		if (!filePath || (!filePath.endsWith(".py") && !filePath.endsWith(".ipynb"))) return;

		// Read the file from disk to check for PyMC imports
		const absPath = path.isAbsolute(filePath) ? filePath : path.join(ctx.cwd, filePath);
		try {
			const content = fs.readFileSync(absPath, "utf-8");
			
			// Check for PyMC/PyTensor/ArviZ imports
			const hasPymcImports = /import\s+(pymc|pytensor|arviz)|from\s+(pymc|pytensor|arviz)/.test(content);
			
			if (hasPymcImports) {
				pi.sendMessage(
					{
						customType: "pymc-detection",
						content: `Detected PyMC/PyTensor/ArviZ imports in ${filePath}. The pymc-modeling skill is available for PyMC 6+ / ArviZ 1.0+ guidance.`,
						display: false,
					},
					{ deliverAs: "steer" }
				);
			}
		} catch {
			// File may not exist or be unreadable; skip silently
		}
	});

	/* ------------------------------------------------------------------------ */
	/*  Post-Write/Edit Lint Checks                                             */
	/* ------------------------------------------------------------------------ */

	pi.on("tool_result", async (event, ctx) => {
		if (event.toolName !== "write" && event.toolName !== "edit") return;
		if (event.isError) return;

		const filePath = (event.input as { path?: string }).path;
		if (!filePath || !filePath.endsWith(".py")) return;

		// Read the file from disk to lint the final state
		const absPath = path.isAbsolute(filePath) ? filePath : path.join(ctx.cwd, filePath);
		try {
			const content = fs.readFileSync(absPath, "utf-8");
			const issues = lintPymcCode(content);
			if (issues.length > 0) {
				const warnings = issues
					.filter((i) => i.severity === "warning")
					.map((i) => `- ${i.message}`);
				const errors = issues
					.filter((i) => i.severity === "error")
					.map((i) => `- ${i.message}`);

				const parts: string[] = [];
				if (errors.length) {
					parts.push(`**PyMC code issues detected in ${filePath}:**\n${errors.join("\n")}`);
				}
				if (warnings.length) {
					parts.push(`**PyMC code warnings in ${filePath}:**\n${warnings.join("\n")}`);
				}

				if (ctx.hasUI) {
					ctx.ui.notify(`PyMC lint: ${errors.length} errors, ${warnings.length} warnings in ${path.basename(filePath)}`,
						errors.length > 0 ? "error" : "warning"
					);
				}

				pi.sendMessage(
					{
						customType: "pymc-lint",
						content: parts.join("\n\n"),
						display: true,
					},
					{ deliverAs: "steer" }
				);
			}
		} catch {
			// File may not exist or be unreadable; skip silently
		}
	});

	/* ------------------------------------------------------------------------ */
	/*  Custom Tools                                                            */
	/* ------------------------------------------------------------------------ */

	pi.registerTool({
		name: "pymc_api_lookup",
		label: "PyMC API Lookup",
		description: "Look up PyMC/ArviZ function signatures, descriptions, and gotchas",
		promptSnippet: "Look up PyMC or ArviZ API documentation by function name",
		promptGuidelines: [
			"Use pymc_api_lookup when the user asks about a specific PyMC or ArviZ function, its parameters, or known gotchas.",
		],
		parameters: Type.Object({
			function_name: Type.String({
				description: "Function name to look up (e.g., 'pm.sample', 'az.loo', 'Normal')",
			}),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
			const query = params.function_name.toLowerCase().trim();
			const api = getApiData();
			const matches: ApiEntry[] = [];
			const partials: ApiEntry[] = [];

			for (const entry of api) {
				const nameLower = entry.name.toLowerCase();
				const shortName = nameLower.split(".").pop() ?? nameLower;
				if (query === nameLower || query === shortName) {
					matches.unshift(entry);
				} else if (nameLower.includes(query) || query.includes(shortName)) {
					partials.push(entry);
				}
			}

			const combined = [...matches, ...partials];

			if (!combined.length) {
				// Fuzzy: search in descriptions
				for (const entry of api) {
					if (textMatch(params.function_name, entry.description)) {
						combined.push(entry);
					}
				}
			}

			if (!combined.length) {
				return {
					content: [
						{
							type: "text",
							text: `No API entry found for '${params.function_name}'. Try a different name or use pymc_example_search for patterns.`,
						},
					],
				};
			}

			const text = combined
				.slice(0, 5)
				.map(formatApiEntry)
				.join("\n\n---\n\n");

			return { content: [{ type: "text", text }], details: { matches: combined.slice(0, 5).map((e) => e.name) } };
		},
	});

	pi.registerTool({
		name: "pymc_example_search",
		label: "PyMC Example Search",
		description: "Search for PyMC code examples and patterns matching a query",
		promptSnippet: "Search PyMC code patterns and examples by keyword",
		promptGuidelines: [
			"Use pymc_example_search when the user asks for code examples, patterns, or 'how do I' questions about PyMC modeling.",
		],
		parameters: Type.Object({
			query: Type.String({
				description: "Search query (e.g., 'hierarchical non-centered', 'mixture model', 'LOO comparison')",
			}),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
			const query = params.query;
			const patterns = getPatternsData();
			const scored: Array<{ score: number; pattern: CodePattern }> = [];

			for (const pattern of patterns) {
				let score = keywordScore(query, pattern.keywords);
				if (score === 0) {
					if (textMatch(query, pattern.title) || textMatch(query, pattern.explanation)) {
						score = 1;
					}
				}
				if (score > 0) {
					scored.push({ score, pattern });
				}
			}

			scored.sort((a, b) => b.score - a.score);

			if (!scored.length) {
				return {
					content: [
						{
							type: "text",
							text: `No patterns found for '${query}'. Try broader keywords like 'hierarchical', 'mixture', 'GP', 'time series'.`,
						},
					],
				};
			}

			const text = scored
				.slice(0, 5)
				.map((s) => formatPattern(s.pattern))
				.join("\n\n---\n\n");

			return { content: [{ type: "text", text }], details: { matches: scored.slice(0, 5).map((s) => s.pattern.id) } };
		},
	});

	pi.registerTool({
		name: "pymc_error_lookup",
		label: "PyMC Error Lookup",
		description: "Look up common PyMC errors and their fixes",
		promptSnippet: "Look up common PyMC errors, diagnostics, and troubleshooting steps",
		promptGuidelines: [
			"Use pymc_error_lookup when the user reports an error, warning, or symptom (e.g., divergences, shape mismatch, slow sampling).",
		],
		parameters: Type.Object({
			error_message: Type.String({
				description:
					"Error message or description (e.g., 'divergences', 'shape mismatch', 'slow sampling')",
			}),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
			const query = params.error_message;
			const scored: Array<{ score: number; err: ErrorPattern }> = [];

			for (const err of ERROR_PATTERNS) {
				let score = keywordScore(query, err.keywords);
				if (score === 0 && textMatch(query, err.title)) {
					score = 1;
				}
				if (score > 0) {
					scored.push({ score, err });
				}
			}

			scored.sort((a, b) => b.score - a.score);

			if (!scored.length) {
				return {
					content: [
						{
							type: "text",
							text:
								`No specific error pattern found for '${query}'.\n\n` +
								"General debugging tips:\n" +
								"1. Check az.summary(idata) for r_hat and ESS\n" +
								"2. Look at az.plot_trace_dist(idata) for mixing\n" +
								"3. Check for divergences in idata[\"sample_stats\"]\n" +
								"4. Try simplifying the model first, then add complexity\n" +
								"5. Use pm.model_to_graphviz(model) to verify structure",
						},
					],
				};
			}

			const text = scored
				.slice(0, 3)
				.map((s) => formatError(s.err))
				.join("\n\n---\n\n");

			return { content: [{ type: "text", text }], details: { matches: scored.slice(0, 3).map((s) => s.err.pattern) } };
		},
	});

	/* ------------------------------------------------------------------------ */
	/*  Commands                                                                */
	/* ------------------------------------------------------------------------ */

	pi.registerCommand("pymc-diagnose", {
		description: "Run full MCMC diagnostics on a DataTree file",
		handler: async (_args, _ctx) => {
			pi.sendUserMessage(CMD_PYM_DIAGNOSE);
		},
	});

	pi.registerCommand("prior-check", {
		description: "Generate and analyze prior predictive checks for the current model",
		handler: async (_args, _ctx) => {
			pi.sendUserMessage(CMD_PRIOR_CHECK);
		},
	});

	pi.registerCommand("shape-check", {
		description: "Validate model shapes and dimensions before sampling",
		handler: async (_args, _ctx) => {
			pi.sendUserMessage(CMD_SHAPE_CHECK);
		},
	});

	pi.registerCommand("model-compare", {
		description: "Compare multiple Bayesian models using LOO-CV",
		handler: async (_args, _ctx) => {
			pi.sendUserMessage(CMD_MODEL_COMPARE);
		},
	});
}
