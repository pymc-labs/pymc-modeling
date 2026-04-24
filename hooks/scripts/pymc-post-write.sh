#!/usr/bin/env bash
# PostToolUse hook on Write|Edit that checks for common PyMC mistakes in .py files
# and reminds about diagnostics after pm.sample().
# Reads tool_use input from stdin JSON.

set -euo pipefail

input=$(cat)

# Extract file path from the tool input
file_path=$(echo "$input" | jq -r '.tool_input.file_path // .tool_input.path // empty' 2>/dev/null || true)

if [ -z "$file_path" ]; then
  exit 0
fi

# Only process Python files
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

# Get file content: use content from input if available, otherwise read from disk
content=$(echo "$input" | jq -r '.tool_input.content // empty' 2>/dev/null || true)
if [ -z "$content" ] && [ -f "$file_path" ]; then
  content=$(cat "$file_path" 2>/dev/null || true)
fi

if [ -z "$content" ]; then
  exit 0
fi

warnings=()

# Check for pm.Flat or pm.HalfFlat usage
if echo "$content" | grep -qE 'pm\.(Flat|HalfFlat)\b'; then
  warnings+=("Flat/HalfFlat priors detected. These are improper priors that can cause sampling issues. Consider using weakly informative priors instead (e.g., pm.Normal with a wide sd, or pm.HalfNormal).")
fi

# Check for missing observed= in likelihood distributions
# Look for common likelihood distributions without observed= on the same line or next few characters
if echo "$content" | grep -qE 'pm\.(Normal|Bernoulli|Poisson|Binomial|NegativeBinomial|StudentT|Cauchy|Exponential|Gamma|Beta|Categorical|Multinomial|Dirichlet|Uniform|LogNormal|HalfNormal|Wald|Weibull)\(' ; then
  # Check if any of these calls lack observed=
  if echo "$content" | grep -E 'pm\.(Normal|Bernoulli|Poisson|Binomial|NegativeBinomial|StudentT|Exponential|Gamma|Beta|Categorical|Multinomial|LogNormal|Wald|Weibull)\([^)]*\)' | grep -vqE 'observed\s*=' 2>/dev/null; then
    # Only warn if there's a pm.sample() call too (suggests this is a full model, not just priors)
    if echo "$content" | grep -qE 'pm\.sample\('; then
      warnings+=("Some distribution calls may be missing observed= argument. Verify that your likelihood term includes observed=data. Prior distributions do not need observed=.")
    fi
  fi
fi

# Check for deprecated ArviZ 0.x idata.posterior access pattern
if echo "$content" | grep -qE 'idata\.(posterior|prior|posterior_predictive|prior_predictive|observed_data|sample_stats|log_likelihood)\b'; then
  warnings+=("Detected idata.posterior-style access pattern. For ArviZ 1.0+, use dt[\"posterior\"] (bracket notation on DataTree) instead of attribute access on InferenceData.")
fi

# Check for pm.compile_pymc (renamed in PyMC 6)
if echo "$content" | grep -qE 'pm\.compile_pymc\b'; then
  warnings+=("pm.compile_pymc was renamed to pm.compile in PyMC 6. Update the call to pm.compile(...).")
fi

# Check for sample_prior_predictive(samples=...) which was renamed to draws= in PyMC 6.
# Two-stage check because call arguments may be split across lines.
if echo "$content" | grep -q 'sample_prior_predictive'; then
  if echo "$content" | grep -qE '\bsamples\s*='; then
    warnings+=("sample_prior_predictive now uses draws= instead of samples= in PyMC 6. Rename the samples= keyword argument to draws=.")
  fi
fi

# Check for az.waic (deprecated in ArviZ 1.0)
if echo "$content" | grep -qE 'az\.waic\('; then
  warnings+=("az.waic is deprecated in ArviZ 1.0. Prefer az.loo for model comparison; use az.loo_metrics / az.loo_expectations for predictive metrics.")
fi

# Check for PyTensor test_value / compute_test_value (removed in PyTensor 3)
if echo "$content" | grep -qE 'tag\.test_value|compute_test_value'; then
  warnings+=("tag.test_value and compute_test_value were removed in PyTensor 3. Call the .eval() method on a symbolic variable with a point-dict (e.g. var.eval({x: x_val})) instead.")
fi

# Check for L_op / R_op method definitions (renamed in PyTensor 3)
if echo "$content" | grep -qE '\bdef\s+(L_op|R_op)\b'; then
  warnings+=("L_op and R_op were renamed in PyTensor 3. Use pull_back (replaces L_op) and push_forward (replaces R_op).")
fi

# Check for deprecated .add_groups( on InferenceData / DataTree
if echo "$content" | grep -qE '\.add_groups\('; then
  warnings+=(".add_groups() is deprecated. Use .update(...) or direct assignment (dt[\"group_name\"] = ds) instead.")
fi

# Remind about diagnostics after pm.sample()
if echo "$content" | grep -qE 'pm\.sample\('; then
  warnings+=("Remember to add convergence diagnostics after pm.sample(): check divergences, r_hat, ESS, and run posterior predictive checks. Save results immediately with .to_netcdf().")
fi

if [ ${#warnings[@]} -gt 0 ]; then
  combined=$(printf '%s\n' "${warnings[@]}")
  jq -n --arg msg "$combined" '{
    "systemMessage": $msg
  }'
fi

exit 0
