# FOCE / FOCEI

First-Order Conditional Estimation (FOCE) and its interaction variant (FOCEI) are the primary estimation methods in ferx-core. They use a two-level nested optimization to find maximum likelihood estimates of population parameters.

## Algorithm Overview

### Outer Loop (Population Parameters)

The outer loop optimizes the population parameters \\( \theta \\), \\( \Omega \\), and \\( \sigma \\) by minimizing the population objective function value (OFV):

\\[ \text{OFV} = -2 \log L = \sum_{i=1}^{N} \text{OFV}_i \\]

Parameters are internally transformed for unconstrained optimization:
- **Theta**: Log-transformed (ensures positivity)
- **Omega**: Cholesky factorization (ensures positive-definiteness)
- **Sigma**: Log-transformed (ensures positivity)

### Inner Loop (Empirical Bayes Estimates)

For each subject, the inner loop finds the empirical Bayes estimate (EBE) of the random effects \\( \hat{\eta}_i \\) by minimizing the individual negative log-likelihood:

\\[ -\log p(\eta_i | y_i, \theta, \Omega, \sigma) = \frac{1}{2} \left[ \eta_i^T \Omega^{-1} \eta_i + \log|\Omega| + \sum_j \left( \frac{(y_{ij} - f_{ij})^2}{V_{ij}} + \log V_{ij} \right) \right] \\]

The inner loop uses BFGS, falling back to Nelder-Mead simplex if BFGS fails.

### Gradient route (AD vs FD)

The BFGS gradient is computed by reverse-mode **automatic differentiation (AD)** when available, and by central **finite differences (FD)** otherwise. AD requires the crate built with the `autodiff` feature (the Enzyme toolchain) *and* an analytical PK model; it is resolved **per subject**, so individual subjects fall back to FD for ODE models, steady-state (SS) doses, system resets (EVID 3/4), or time-varying covariates on a model the event-driven AD path doesn't yet support. AD is the default (`gradient_method = auto`) and is typically on par with FD on small models and faster as the parameter count grows.

The startup banner reports the route **actually resolved** across the population — not just the requested setting — so a silent AD→FD fallback is visible:

```
  gradient: AD (single-snapshot)  [requested: auto]
```

When the population splits across routes, the banner shows per-route subject counts, e.g. `AD (event-driven) ×118, FD ×3`. A build without the `autodiff` feature always reads `FD  [requested: auto; autodiff not compiled in]`.

This `gradient:` line appears for gradient-driven estimators (FOCE/FOCEI/GN) and for `imp`, which reuses the EBE Hessian built via the same route. SAEM is sampling-based and reports its E-step kernel on a `sampler:` line instead — see [SAEM](saem.md).

If the EBE search wanders into a region where the individual NLL evaluates to a non-finite value (for example, an ODE model whose integration blows up at extreme \\( \eta \\)), that point is treated as the worst possible objective rather than aborting the fit. The subject is reported as non-converged and estimation continues for the remaining subjects.

## FOCE vs FOCEI

### Standard FOCE

Uses linearized predictions around the EBEs:

\\[ f_0 = f(\hat{\eta}) - H \hat{\eta} \\]

where \\( H \\) is the Jacobian matrix \\( \partial f / \partial \eta \\). The per-subject objective is:

\\[ \text{OFV}_i = (y - f_0)^T \tilde{R}^{-1} (y - f_0) + \log|\tilde{R}| \\]

where \\( \tilde{R} = H \Omega H^T + R(f_0) \\).

### FOCEI (Interaction)

Uses individual predictions directly without linearization:

\\[ \text{OFV}_i = (y - \hat{f})^T V^{-1} (y - \hat{f}) + \hat{\eta}^T \Omega^{-1} \hat{\eta} + \log|\tilde{R}| \\]

FOCEI is more accurate when the residual variance depends on the predicted value (proportional or combined error models), because it captures the interaction between random effects and residual error.

## Optimizer Options

### NLopt Algorithms (Recommended)

| Algorithm | Key | Description |
|-----------|-----|-------------|
| SLSQP | `slsqp` | Sequential Least Squares Programming. Handles bounds well. **Default and recommended.** |
| L-BFGS | `nlopt_lbfgs` | Limited-memory BFGS. Good for large parameter spaces. |
| MMA | `mma` | Method of Moving Asymptotes. Alternative constrained optimizer. |
| BOBYQA | `bobyqa` | Derivative-free trust-region via quadratic interpolation. Useful when FD gradients are unreliable (e.g. noisy FOCE surface). |

### Built-in Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| BFGS | `bfgs` | Quasi-Newton with backtracking line search. |
| L-BFGS | `lbfgs` | Memory-efficient BFGS variant. |

### Newton Trust-Region (argmin)

| Algorithm | Key | Description |
|-----------|-----|-------------|
| Trust region | `trust_region` | Newton trust-region with Steihaug conjugate-gradient subproblem. Uses the AD-based outer gradient (`subject_nll_pop_grad`) and a BHHH approximate Hessian `H ≈ 4 Σ gᵢgᵢᵀ` — always positive semi-definite, so the Steihaug subproblem stays well-conditioned. The CG budget defaults to `ceil(sqrt(n_params)).clamp(5, n_params)` (typically 5 for standard NLME models); pin it explicitly with `steihaug_max_iters`. |

## Global Search

Enable gradient-free pre-search to help escape local minima:

```
[fit_options]
  method        = foce
  global_search = true
  global_maxeval = 2000
```

The pre-search uses NLopt's CRS2-LM algorithm (Controlled Random Search) to explore the parameter space before handing off to the local optimizer. The number of evaluations auto-scales with model complexity if `global_maxeval` is not set.

## Covariance Step

When `covariance = true`, ferx-core computes the variance-covariance matrix of the parameter estimates using a finite-difference Hessian at the converged solution. This provides:

- **Standard errors (SE)** for all parameters
- **Relative standard errors (%RSE)** for assessing estimation precision
- **Omega SEs** via delta method from the Cholesky parameterization

## Convergence

The outer loop terminates when any of:
- The gradient norm falls below `outer_gtol` (default 1e-6)
- The maximum number of iterations (`maxiter`) is reached
- The optimizer reports convergence (NLopt `XtolReached` or `FtolReached`)

The inner loop terminates when the gradient norm falls below `inner_tol` (default `1e-4`, matching NONMEM's ~3 SIGDIGITS inner-loop precision — see [`fit-options.md`](../model-file/fit-options.md#general-options) for tuning guidance) or `inner_maxiter` (default 200) iterations are reached.

## Warm Starting

The inner loop warm-starts EBE estimation from the previous outer iteration's EBEs. This significantly reduces computation time, especially in later iterations when parameters change slowly.

## Configuration Example

```
[fit_options]
  method     = focei
  maxiter    = 500
  covariance = true
  optimizer  = slsqp
```
