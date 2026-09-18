---
name: prior-elicitation
description: >-
  Elicit and check Bayesian prior assumptions in meaningful units. Use for expert
  quantiles, PreliZ, constrained priors, prior-predictive plausibility and prior
  sensitivity, including power scaling, regularized horseshoe and R2D2 assumptions.
---

# Prior elicitation

A prior is a scientific assumption, not a scale-free default. Work from the
quantity's meaning to a distribution, then check its observable implications.

## Workflow

1. **Define the quantity.** Establish the estimand, population, support, units,
   transformation/link, reference predictor values and information available
   before outcomes. Separate physical bounds from high-probability intervals,
   parameter uncertainty from future-observation variability, and independent
   prior information from outcome-based scaling.
2. **Elicit, do not invent.** Ask for quantiles, probabilities, meaningful predictor
   contrasts and tail judgments; record their source and disagreements. Label
   teaching numbers as illustrative assumptions. Do not tune priors to held-out
   outcomes or fabricate expert answers.
3. **Translate and check.** Use analytic formulas or PreliZ `maxent`/`quartile`.
   Inspect optimizer status and achieved CDFs independently. Maximum entropy is
   conditional on a chosen family and parameterization, not universally
   noninformative. Prefer PreliZ over deprecated `pm.find_constrained_prior`.
4. **Simulate prior predictions.** Use `pm.sample_prior_predictive(draws=...)`;
   access the resulting DataTree with `prior["prior_predictive"]`. Check support,
   conditional scales, dataset extremes and scientific contrasts against stated
   plausibility judgments, beyond observed-range coverage. Revisit assumptions
   when predictions are implausible, preserving support and simulation uncertainty.
   These are plausibility checks, not SBC of an inference implementation.
5. **Assess reasonable alternatives.** Vary plausible scales and tails while
   holding data, likelihood, transformations and estimand fixed for a prior
   sensitivity comparison. Rough exploratory fits can flag which assumptions to
   investigate; diagnose every posterior supporting reported conclusions with
   divergences, rank R-hat, bulk/tail ESS, estimand-specific MCSE and trace/rank plots.
6. **Report the actual sensitivity.** Compare physical-scale effects, meaningful
   event probabilities, uncertainty intervals and new-observation predictions.
   Stable answers over a few priors do not prove data dominance, global robustness
   or model adequacy. Power sensitivity is local and also needs reliable
   importance weights; its heuristic labels are not scientific decisions.

## References

- [Elicitation, PreliZ, prior predictions and sensitivity](references/elicitation.md)
  — a complete prior-predictive example, unit conversions, expert workflow,
  analytic checks and power reweighting.
- [Shrinkage and regularization](references/shrinkage.md) — regularized horseshoe
  scales, R2D2 variance allocation, geometry and interpretation.

Use the installed package's version-matched API documentation when adapting the
examples. PreliZ widgets need an interactive notebook; pymc-extras is optional
and only needed for its R2D2 helper. This skill does not require another skill.
