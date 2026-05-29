# Outer Optimizers

The **outer optimizer** is the algorithm that minimises the population objective function (OFV) over the transformed parameter vector `[log θ, chol(Ω), log σ]`.

It only applies to methods that *have* a single outer-loop minimisation over that vector:

| method | does `optimizer` apply? |
|---|---|
| `foce` / `focei` | yes — picks the algorithm for the population OFV minimisation |
| `gn` (pure Gauss-Newton) | no — GN runs its own Levenberg-Marquardt loop |
| `gn_hybrid` | yes for the FOCEI polish phase only; the preceding GN phase ignores it |
| `saem` | **no** — SAEM has no single outer minimisation; the M-step uses a hardcoded NLopt algorithm and is not user-selectable. Setting `optimizer` under `method = saem` triggers a warning and is otherwise ignored. |
| `[saem, focei]` (chain) | applies to the FOCEI polish stage only |
| `imp` | no — IS is a sampling pass, not an optimisation |

Set via `[fit_options]`:

```
[fit_options]
  method    = focei
  optimizer = bobyqa
```

---

## Available optimizers

### NLopt algorithms

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `slsqp` | Sequential Least Squares Programming | **Default.** Gradient-based, handles bounds well. Fast on well-conditioned models. |
| `bobyqa` | Bounded Optimization BY Quadratic Approximation | Derivative-free. Recommended when the fixed-EBE gradient is noisy (ODE models, sparse data). Often reaches a lower OFV than `slsqp` without reconverging. |
| `nlopt_lbfgs` | L-BFGS via NLopt | Limited-memory BFGS. Useful for high-parameter-count models. |
| `mma` | Method of Moving Asymptotes | Alternative constrained gradient optimizer. Rarely needed. |

### Built-in algorithms

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `bfgs` | BFGS with backtracking line search | Quasi-Newton, no NLopt dependency. |
| `lbfgs` | L-BFGS | Memory-efficient BFGS variant. |

### Newton trust-region

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `trust_region` | Newton trust-region with Steihaug CG | Uses the AD-based outer gradient and a BHHH approximate Hessian `H ≈ 4 Σ gᵢgᵢᵀ` (always positive semi-definite). The CG budget defaults to `ceil(sqrt(n_params)).clamp(5, n_params)`; pin it with `steihaug_max_iters`. Best combined with `inits_from_nca` since it benefits from good starting values. |

---

## When to use which

**`slsqp` (default)** — start here. Works well for most 1–2 compartment models with 2–4 random effects.

**`bobyqa`** — best single alternative. Use when `slsqp` converges to a suspiciously high OFV, or for ODE models and sparse data where the finite-difference gradient is noisy. Derivative-free means it re-evaluates EBEs at every trial point, so it naturally avoids the fixed-EBE gradient bias that can stall `slsqp`.

**`trust_region`** — for models with many parameters (large θ + Ω) or when combined with `inits_from_nca`. The second-order curvature information helps when starting values are already in the basin.

**`gn` / `gn_hybrid`** — these are [Gauss-Newton estimation methods](gauss-newton.md), not outer optimizers in the same sense. They replace the FOCE outer loop entirely rather than selecting an algorithm within it.

**`lbfgs` / `nlopt_lbfgs` / `mma`** — rarely needed. Prefer `slsqp` or `bobyqa`.

---

## Fixed-EBE gradient bias and `reconverge_gradient_interval`

By default, gradient-based optimizers (`slsqp`, `lbfgs`, etc.) hold each subject's EBEs fixed while computing the population gradient. This is cheap but omits the response of the inner solution to θ and Ω. On ill-conditioned models the omitted term causes `slsqp` to stall well above the `bobyqa` optimum.

Set `reconverge_gradient_interval = 1` to re-solve the inner loop at every gradient evaluation, recovering the full gradient surface at roughly **5–6× cost**. A value of `N` reconverges every Nth evaluation and uses the cheap gradient in between — often enough to close most of the gap:

| Configuration | OFV | Wall time |
|---|---:|---:|
| `slsqp` (fixed-EBE, default) | 68,252 | 390 s |
| `slsqp` + `interval = 10` | 66,118 | 633 s |
| `slsqp` + `interval = 1` | **65,485** | 1,871 s |
| `bobyqa` (derivative-free) | 65,598 | 315 s |

On this cefepime 2-compartment dataset `bobyqa` reaches a near-optimal solution faster than any reconverged `slsqp` setting. The reconverged gradient is most useful when derivative-free search is too slow (high parameter count) or when a gradient optimizer is required for other reasons.

IOV models (`kappa`/`block_kappa`) always reconverge regardless of this setting.

---

## Global pre-search

Enable `global_search = true` to run a derivative-free CRS2-LM pre-search before the local optimizer. This helps escape local minima on complex models:

```
[fit_options]
  method        = focei
  global_search = true
  global_maxeval = 2000   # auto-scaled if omitted
```

The pre-search uses NLopt's Controlled Random Search and requires lower/upper bounds on all theta parameters. It hands off to the local optimizer (`optimizer` key) once the budget is exhausted.

See also [`fit-options.md`](../model-file/fit-options.md) for the full list of optimizer-related options.
