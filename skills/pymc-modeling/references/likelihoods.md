# Support-aware likelihood and transform selection

## Start with the observation process

Declare the observational unit, conditional independence, exposure, units,
measurement resolution, missingness and attainable endpoints. Distinguish an
exact zero from rounding, censoring and structural absence. Recording at a limit,
excluding units from a sample, and transforming measurement units require different
likelihoods. Broadcasting does not establish independence. Do not clip invalid
data or add arbitrary pseudocounts to make a model run: fix units/coding or model
the actual measurement/selection process.

`Name.dist(...)` creates an unregistered RV for wrappers or density calculations;
`Name("y", ..., observed=y)` registers a likelihood. Prefer public `pm.logp`,
`pm.logcdf`, `pm.logccdf` and `pm.icdf` dispatch rather than implementation methods.
Availability varies by family. A support point is an initialization convention,
not necessarily a mean, median or mode; an undefined analytic mean is not repaired
by substituting a support point.

## Scalar continuous families

PyMC parameter meanings need not match SciPy's.

| Family | Support and parameters | Important boundary |
|---|---|---|
| Normal | Real; mu, positive sigma or precision tau | sigma is SD; do not supply both scale forms. |
| StudentT, Cauchy | Real; positive scale; T nu>0 | T mean requires nu>1, variance nu>2 and equals sigma² nu/(nu-2). Cauchy has neither. Heavy tails do not delete outliers or fix leverage/missing groups. |
| Laplace, AsymmetricLaplace | Real; positive b; asymmetric kappa>0 or q in (0,1) | Laplace b is scale; asymmetric b is inverse scale. Quantile-regression working likelihood needs justification. |
| Logistic | Real; location mu, scale s>0 | Response density, not Bernoulli with a logistic link. |
| SkewNormal, SkewStudentT | Real; location/scale plus skew/tail parameters | Location is generally not mean. Jones–Faddy skew T uses a,b>0; check moment existence. |
| ExGaussian | Real; Normal mu,sigma and Exponential mean nu | Still allows negative responses. SciPy exponnorm K=nu/sigma. |
| Gumbel, Moyal | Real; location and positive scale | Not positive-only; choose the correct tail direction. |
| Exponential, Gamma, ChiSquared | Nonnegative; Exponential lam and Gamma beta are rates | Gamma alternatively accepts positive mu,sigma. ChiSquared(nu)=Gamma(nu/2, rate=1/2). Infinite density at zero is not an atom. |
| InverseGamma | Positive; alpha,beta>0 | beta is inverse-gamma scale; mean requires alpha>1, variance alpha>2. |
| Weibull | Positive; shape alpha, scale beta | Hazard shape differs from an exponential assumption. |
| Wald | Positive; mu,lam>0, optional shift alpha | SciPy invgauss(mu/lam, scale=lam), with shift applied separately. |
| Rice | Positive; nu>=0, sigma>0, or b=nu/sigma | Radial amplitude, not a generic duration law. |
| LogNormal | Positive; mu,sigma on log scale | Mean exp(mu+sigma²/2); original-scale log-density needs the measurement Jacobian if fitting log(y). |
| HalfNormal, HalfStudentT, HalfCauchy | Nonnegative; positive scale, T also nu>0 | Normalized half-laws; scale/tail mass needs elicitation, not just positive initialization. |
| Pareto | y>=m>0; alpha>0 | Hard threshold; mean/variance require alpha>1/>2. |
| Beta, LogitNormal, Kumaraswamy | Continuous unit interval | No endpoint atoms. Beta concentration positive; alternative mu,sigma needs sigma²<mu(1-mu). Binomial denominators should not be discarded. |
| Uniform, Triangular, TruncatedNormal | Ordered bounds; triangular interior mode c | Equal density is scale-dependent; truncated Normal includes normalization, unlike an interval transform. |
| VonMises | Circular angle, radians; concentration kappa | -pi and pi are the same direction. Use circular summaries. |
| Interpolated | Ordered finite lattice, nonnegative ordinates, positive integrated density | Normalized interpolated PDF with zero density outside its grid; check tails/resolution and density-estimation uncertainty. |
| Flat, HalfFlat | Improper constant kernels | No genuine prior RNG; establish posterior propriety. Arbitrary constants invalidate Bayes-factor uses. |
| DiracDelta | Point mass at c | Not a narrow Normal or a continuously varying NUTS parameter. |
| PolyaGamma | Positive; h>0, real tilt z | Augmentation law; numerical density/CDF/RNG needs optional polyagamma. Mean h/4 at z=0, h*tanh(z/2)/(2z) otherwise. |

## Counts, categories and ordered outcomes

| Family | Contract and scientific distinction |
|---|---|
| Bernoulli, Binomial | Binary or integer 0..n; p in [0,1], n nonnegative integer. Bernoulli also accepts logit_p. Preserve trial denominators. |
| BetaBinomial | 0..n with positive alpha,beta; heterogeneity in Binomial probability, not independent Beta measurements. |
| Poisson | Nonnegative integers; E=Var=mu. Variable exposure enters log(mu)=log(exposure)+predictor; handle zero exposure substantively. |
| NegativeBinomial | E=mu, Var=mu+mu²/alpha; equivalent n=alpha, p=alpha/(alpha+mu). n is dispersion, not an observed trial upper bound. |
| Geometric | Trials through first success: 1,2,...; shift explicitly if observations count failures. |
| HyperGeometric | Sampling without replacement; compatible population N, successes k, draws n; support max(0,n-N+k)..min(k,n). |
| DiscreteUniform | Both integer bounds inclusive; normalizer upper-lower+1. |
| DiscreteWeibull | Nonnegative integers, q in (0,1), beta>0; mass q^(y^beta)-q^((y+1)^beta). Not an arbitrarily rounded Weibull. |
| Categorical | Integer 0..K-1; p simplex or logits. Final probability axis is consumed, labels have no metric. |
| OrderedLogistic, OrderedProbit | Integer 0..K-1, K-1 increasing cutpoints; logistic versus Normal latent error. |
| OrderedMultinomial | Category-count vector summing to n from ordered trials sharing eta/cutpoints. |

Validate finite integer-valued counts, range, rank, totals and integer overflow
**before casting**. Casting fractional counts is not support validation.

Cumulative-link probabilities are
`G(c[k+1]-eta)-G(c[k]-eta)` with infinite exterior cutpoints. Use stable log-domain
CDF differences in extreme tails. Free intercept and cutpoint locations are
aliased: fix an anchor or otherwise identify location. An ordered transform does
not sort prior-predictive draws; an anchored first cutpoint plus positive spacings
is a generatively ordered prior. Ordered-transformed variables need strictly
ordered finite initvals, but initialization does not create identification.

## Censoring, truncation and interval recording

For a continuous base density f and CDF F:

- Censoring retains all units but records `max(L,min(X,U))`. At L use mass F(L),
  at U use survival 1-F(U), and in the interior use f(y). Limits must align by row.
- Truncation samples only retained units: density is f(y)/(F(U)-F(L)). For integer
  inclusive bounds, the normalizer is F(U)-F(L-1), not F(U)-F(L).
- Interval censoring `L<X<=U` contributes F(U)-F(L), not a truncated density or
  exact observation. Use stable log-CDF or survival differences and correct
  discrete inclusivity.

```python
pm.Censored("recorded", pm.Normal.dist(mu, sigma),
            lower=lower, upper=upper, observed=recorded)
# Alternative sampling process, not another factor for the same observations:
pm.Truncated("selected", pm.Normal.dist(mu, sigma),
             lower=lower, upper=upper, observed=selected)
```

Wrappers clone unnamed univariate RVs; reuse of a Python object does not share a
latent realization. Check base CDF/quantile support. Truncated generation may use
specialized, inverse-CDF or finite-step rejection sampling; increasing
`max_n_steps` cannot fix an impossible interval. Continuous censored/hurdle laws
have atoms and should not be free variables explored by ordinary continuous
NUTS. For forward clipping, generate an uncensored latent and a clipped deterministic.
Noninformative censoring conditional on covariates needs justification.

## Excess zeros and robust tails

PyMC `psi` is probability of entering the **body**, not structural-zero probability.
For count-body PMF g:

- ZeroInflatedPoisson/Binomial/NegativeBinomial: P(0)=1-psi+psi*g(0),
  P(y>0)=psi*g(y), mean psi*E_body. ZIP variance is
  psi*mu+psi*(1-psi)*mu².
- HurdlePoisson/NegativeBinomial: P(0)=1-psi; positive mass
  psi*g(y)/(1-g(0)). Its body parameter mu is the untruncated mean.
- HurdleGamma/LogNormal: atom 1-psi at zero and positive density psi*f(y).
  In PyMC 6.3.1 the continuous body is not machine-epsilon truncated despite
  inconsistent older docstrings. Gamma beta is rate; mu,sigma refer to the
  underlying Gamma mean/SD. LogNormal parameters remain on log scale.

Overdispersion alone does not identify a structural-zero population. Rounding,
exposure or detection limits can cause zeros differently. A StudentT response can
reduce tail influence without solving leverage, heteroscedasticity or wrong links.
Use prior/PPC sensitivity to assess these mechanisms, not a family name as proof.

## Multivariate, compositional and spatial families

| Family | Event and interpretation |
|---|---|
| MvNormal, MvStudentT | Last axis is one joint vector, preceding axes are batches. Covariance/Cholesky must be valid. MvStudentT scale becomes covariance only after nu/(nu-2), for nu>2. |
| MatrixNormal | Two event axes; separable row/column covariance. Unrestricted factor scales are nonidentified. |
| KroneckerNormal | Vector ordering must match covariance-factor product; independent sigma² I optional. Separability is a substantive restriction. |
| Dirichlet | Positive simplex with positive concentration vector, not necessarily totaling one; not independent Betas. |
| Multinomial, DirichletMultinomial | Integer event vector summing to n; one scalar event logp. The latter adds heterogeneity. |
| StickBreakingWeights | K Beta(1,alpha) breaks produce K+1 weights; final weight is remainder. Finite truncation changes the model. Validate malformed weights yourself. |
| ZeroSumNormal | Proper on a lower-dimensional subspace, singular in ambient space. For one K-axis, marginal variance is sigma²(1-1/K), not sigma². |
| Wishart | SPD event; integer nu>p-1, one of V/scale_chol. PyMC 6.3.1 supplies a CholeskyCovTransform; assess prior and geometry rather than declare it categorically unusable. WishartBartlett is deprecated. |
| LKJCholeskyCov | Positive unnamed SD distribution, eta>0; returns chol,corr,stds with compute_corr. Covariance is chol@chol.T. |
| LKJCorr | PyMC 6.3.1 returns a lower correlation Cholesky factor despite conflicting prose; construct R=L@L.T. |
| CAR | Precision tau*(D-alpha*W). Check symmetric adjacency, positive degrees, graph components, node alignment and positive definiteness. |
| ICAR | Singular intrinsic-Laplacian difference penalty plus soft sum constraint; disconnected components require explicit identification/propriety treatment. |

Version-specific cautions from [PyMC 6.3.1 multivariate source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/multivariate.py):
CAR omits a fixed-graph normalizing constant relative to a full Gaussian; restore
it when absolute scores or graph comparisons require normalization. ICAR's soft
sum penalty does not supply a general scale-dependent normalizer, and prior RNG
is unimplemented. Formulate a normalized constrained Gaussian when inferring
scale or computing marginal likelihoods requires one. LKJ normalizing-constant
sign/representation deserves an independent check before absolute-density uses
or inference on eta: for n=2, `(rho+1)/2 ~ Beta(eta,eta)` is a simple oracle.
Factor support and transform roundtrips alone cannot check density normalization.
Check the installed release before treating any of these as a current limitation.

## Transforms and Jacobians

Forward maps constrained x to sampling z; backward maps z to x:
`log p_Z(z) = log p_X(backward(z)) + log|d backward(z)/dz|`.
`pm.logp(rv,x)` uses original support; compiled model logp normally includes the
transform Jacobian. Do not add it twice. A Normal z with deterministic exp(z)
already induces LogNormal x. A transformed observed response needs its measurement
Jacobian for original-scale normalized density; a fixed data-only term can cancel
in parameter inference, but not inconsistently scaled predictive comparisons.

| Transform | Backward map and boundary |
|---|---|
| log, log_exp_m1 | exp or softplus to positive values; exact zero requires infinite coordinates. Softplus Jacobian is log sigmoid(z). |
| logodds | sigmoid to (0,1); no endpoint atoms. |
| Interval | Affine sigmoid for finite bounds, exponential distance for one-sided bounds. Bound callbacks must use distribution inputs, not unrelated RV closures. No truncation normalizer is added. |
| ordered | First value plus exp spacings; positive/descending options alter chart. Tied starts cause log(0). Prior forward draws are not automatically sorted. |
| simplex | K-1 free coordinates to K weights; use the K-1 measure, not a singular ambient determinant. |
| ZeroSumTransform | Orthonormal subspace embedding; zero log Jacobian relative to that subspace measure. |
| CholeskyCovPacked | Exponentiate packed diagonal; log Jacobian is sum of transformed diagonals. Output remains a factor. |
| CholeskyCovTransform | Free vector to L to SPD L L.T; Jacobian n log 2 + sum_k(n-k+2) log L_kk. |
| CholeskyCorrTransform | Free partial coordinates to row-normalized lower factor, not correlation matrix itself. Verify nondefault orientation separately. |
| circular | Local angular chart; wrapping is not a global real-line bijection. |
| Chain | Ordered composition with compatible event dimensions and summed Jacobians. |
| SumTo1 | Historical chart restores last component; remaining coordinates are still constrained. Use simplex instead of deprecated sum_to_1. |

Check interior roundtrips and finite-difference log determinants on the appropriate
independent coordinates. Those checks do not establish the base density's normalizer.

## Dtype and named dimensions

Float64 observations do not prevent float32 intermediates from constants; a final
cast cannot recover lost precision. Choose construction-time constant policy and
dtype deliberately for tight checks, preserving failures rather than loosening
oracles. Scalar-event `(N,K)` Normals have `(N,K)` logp; `(N,K)` MvNormal has `(N,)`.
Shape labels cannot convert one event law into the other.

`pymc.dims` uses named xtensors, unlike ordinary positional `dims`. Supply labelled
non-scalar inputs and support `core_dims`. Dirichlet retains its category axis;
Categorical consumes it. In PyMC 6.3.1 named `.dist()` uses `dim_lengths`, not
positional `shape`/`size`/`dims`; registered variables need model coordinates.
Named MvNormal needs two distinct covariance core dims, exactly one shared with
mu; use cov or chol, never both (tau is not supported there). Named distributions
need matching `DimTransform` implementations rather than regular tensor transforms.

Sources: [continuous families](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/continuous.py),
[discrete families](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/discrete.py),
[mixtures](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/mixture.py),
[transforms](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/transforms.py),
[named distributions](https://github.com/pymc-devs/pymc/tree/v6.3.1/pymc/dims/distributions),
[Censored](https://www.pymc.io/projects/docs/en/stable/api/distributions/censored.html),
[Truncated](https://www.pymc.io/projects/docs/en/stable/api/distributions/truncated.html).
