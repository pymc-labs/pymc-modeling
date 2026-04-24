#!/usr/bin/env bash
# Suggest PyMC skills based on keywords in the user's prompt.
# Runs as a UserPromptSubmit hook -- receives JSON on stdin with "user_prompt" field.
# Must exit 0 regardless of match (hooks must not fail).

set -euo pipefail

input=$(cat)
prompt=$(echo "$input" | jq -r '.user_prompt // empty' 2>/dev/null || true)

if [ -z "$prompt" ]; then
  exit 0
fi

# Convert to lowercase for matching
prompt_lower=$(echo "$prompt" | tr '[:upper:]' '[:lower:]')

suggest_pymc=false
suggest_pymc_testing=false
suggest_prior_elicitation=false
suggest_model_evaluation=false
suggest_pymc_extras=false

# PyMC modeling keywords
pymc_keywords=(
  "bayesian" "pymc" "mcmc" "posterior" "inference" "arviz"
  "prior" "sampling" "divergence" "hierarchical model"
  "gaussian process" "bart" "nuts" "hmc" "nutpie" "probabilistic"
  "credible interval" "posterior predictive" "prior predictive"
  "trace" "r_hat" "rhat" "ess_bulk" "convergence" "hsgp"
  "zero.inflated" "mixture model" "multilevel" "brms"
  "logistic regression.*bayes" "poisson regression.*bayes"
  "censored" "truncated" "ordinal" "causal inference"
  "do.calculus" "pm\\.model" "pm\\.sample" "pm\\.normal"
  "pull_back" "push_forward" "arviz_base" "arviz-stats"
)

for kw in "${pymc_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_pymc=true
    break
  fi
done

# PyMC testing keywords
pymc_testing_keywords=(
  "testing pymc" "test.*pymc" "pymc.*test" "mock.sample"
  "mock_sample" "pytest.*pymc" "pymc.*pytest" "unit test.*model"
  "test fixture.*pymc" "ci.*pymc" "pymc.*ci"
)

for kw in "${pymc_testing_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_pymc_testing=true
    break
  fi
done

# Prior elicitation keywords
prior_elicitation_keywords=(
  "find_constrained_prior" "preliz" "elicit" "prior selection"
  "prior predictive" "constrained prior" "prior elicitation"
  "expert knowledge.*prior" "prior.*expert" "informative prior"
  "weakly informative" "domain knowledge.*prior"
)

for kw in "${prior_elicitation_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_prior_elicitation=true
    break
  fi
done

# Model evaluation keywords
model_evaluation_keywords=(
  "model comparison" "loo" "elpd" "stacking" "bayes factor"
  "cross-validation" "waic" "model averaging" "model weight"
  "az\\.compare" "az\\.loo" "pointwise.*loo" "loo.pit"
  "k.pareto" "pareto.k" "information criterion"
  "loo_expectations" "loo_metrics" "azstats" "loo_r2"
)

for kw in "${model_evaluation_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_model_evaluation=true
    break
  fi
done

# PyMC-extras keywords
pymc_extras_keywords=(
  "pymc_extras" "pmx" "splines" "distributional regression"
  "r2d2" "marginalize" "fit_laplace" "laplace approximation"
  "horseshoe" "finnish horseshoe" "regularized horseshoe"
  "pymc.extras" "pymc-extras"
)

for kw in "${pymc_extras_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_pymc_extras=true
    break
  fi
done

# Build suggestion message
messages=()
if [ "$suggest_pymc" = true ]; then
  messages+=("Consider using the **pymc-modeling** skill for Bayesian modeling guidance.")
fi
if [ "$suggest_pymc_testing" = true ]; then
  messages+=("Consider using the **pymc-testing** skill for PyMC model testing guidance.")
fi
if [ "$suggest_prior_elicitation" = true ]; then
  messages+=("Consider using the **prior-elicitation** skill for prior selection and elicitation guidance.")
fi
if [ "$suggest_model_evaluation" = true ]; then
  messages+=("Consider using the **model-evaluation** skill for model comparison and evaluation guidance.")
fi
if [ "$suggest_pymc_extras" = true ]; then
  messages+=("Consider using the **pymc-extras** skill for pymc-extras features (splines, R2D2, marginalization, Laplace).")
fi

if [ ${#messages[@]} -gt 0 ]; then
  combined=$(printf '%s ' "${messages[@]}")
  # Output as JSON systemMessage for Claude
  jq -n --arg msg "$combined" '{
    "systemMessage": $msg
  }'
fi

exit 0
