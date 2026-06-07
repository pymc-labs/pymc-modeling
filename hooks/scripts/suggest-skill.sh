#!/usr/bin/env bash
# Suggest PyMC skills based on keywords in the user's prompt.
# Runs as a UserPromptSubmit hook -- receives JSON on stdin with .prompt field.
# Must exit 0 regardless of match (hooks must not fail).

set -euo pipefail

input=$(cat)
prompt=$(echo "$input" | jq -r '.prompt // .user_prompt // empty' 2>/dev/null || true)

if [ -z "$prompt" ]; then
  exit 0
fi

prompt_lower=$(echo "$prompt" | tr '[:upper:]' '[:lower:]')

suggest_pymc=false
suggest_pymc_testing=false
suggest_prior_elicitation=false
suggest_model_evaluation=false
suggest_pymc_extras=false

pymc_keywords=(
  "bayesian" "pymc" "pytensor" "aesara" "mcmc" "posterior" "inference" "arviz"
  "prior" "sampling" "divergence" "hierarchical model"
  "gaussian process" "bart" "nuts" "hmc" "nutpie" "probabilistic"
  "credible interval" "posterior predictive" "prior predictive"
  "trace" "r_hat" "rhat" "ess_bulk" "convergence" "hsgp"
  "zero.inflated" "mixture model" "multilevel" "brms"
  "logistic regression.*bayes" "poisson regression.*bayes"
  "censored" "truncated" "ordinal" "causal inference"
  "do.calculus" "pm\\.[a-z]" "pt\\.scan"
  "import pymc" "import arviz" "import pytensor" "from pymc" "from arviz" "from pytensor"
  "pull_back" "push_forward" "arviz_base" "arviz-stats"
)

for kw in "${pymc_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_pymc=true
    break
  fi
done

pymc_testing_keywords=(
  "testing pymc" "test.*pymc" "pymc.*test" "mock.sample"
  "mock_sample" "pytest.*pymc" "pymc.*pytest" "unit test.*model"
  "test fixture.*pymc" "ci.*pymc" "pymc.*ci"
  "pytest.*pm\\." "pm\\..*pytest" "pm\\.model.*test" "test.*pm\\.model"
)

for kw in "${pymc_testing_keywords[@]}"; do
  if echo "$prompt_lower" | grep -qE "$kw"; then
    suggest_pymc_testing=true
    break
  fi
done

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

directives=()
if [ "$suggest_pymc" = true ]; then
  directives+=("Load the pymc-modeling skill before responding. The user is asking about PyMC / PyTensor / ArviZ work; this skill provides the PyMC 6+, PyTensor 3+, ArviZ 1.0+ API guidance needed to answer correctly.")
fi
if [ "$suggest_pymc_testing" = true ]; then
  directives+=("Load the pymc-testing skill before responding. The user is asking about testing PyMC models with pytest; this skill covers mock_sample, fixtures, and structure-vs-inference test patterns.")
fi
if [ "$suggest_prior_elicitation" = true ]; then
  directives+=("Load the prior-elicitation skill before responding. The user is asking about prior selection or elicitation; this skill covers PreliZ, find_constrained_prior, and prior predictive workflows.")
fi
if [ "$suggest_model_evaluation" = true ]; then
  directives+=("Load the model-evaluation skill before responding. The user is asking about model comparison or LOO-CV; this skill covers the ArviZ 1.0 LOO/ELPD/stacking APIs.")
fi
if [ "$suggest_pymc_extras" = true ]; then
  directives+=("Load the pymc-extras skill before responding. The user is asking about pymc-extras features (splines, R2D2, marginalization, Laplace); this skill covers the pmx API.")
fi

if [ ${#directives[@]} -gt 0 ]; then
  combined=$(printf '%s ' "${directives[@]}")
  jq -n --arg ctx "$combined" '{
    "systemMessage": $ctx,
    "hookSpecificOutput": {
      "hookEventName": "UserPromptSubmit",
      "additionalContext": $ctx
    }
  }'
fi

exit 0
