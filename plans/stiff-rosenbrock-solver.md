# Plan: Stiff ODE solver (Rosenbrock23) behind the `solve_ode_g` seam

**Tracking issue:** [sn248/ferx-core#1](https://github.com/sn248/ferx-core/issues/1) (stiff-system support: TMDD / QSP / PBPK / tight indirect-response)
**Scope:** ferx-core (primary) + ferx-r (follow-up once any `pub` API lands)
**Status:** design drafted; awaiting core-team sign-off (touches the numerics + `.ferx` surface,
so this is a design gate per `docs/development/sdlc.qmd` §4). Multi-PR / phased.
**Builds on:** the generic `solve_ode_g::<T: PkNum>` stepper (`src/ode/solver.rs`) and the
`Dual2` analytic-sensitivity path (`src/sens/`). Reuses `scale_tol`, `OdeSolverStats`,
`MAX_CONSECUTIVE_MIN_STEP_CLAMPS`, `NONFINITE_ERR_SHRINK_FACTOR` already in `solver.rs`.

---

## 1. Objective & scope

Add an opt-in, L-stable, linearly-implicit **Rosenbrock23** (Shampine–Reichelt `ode23s`)
integrator behind the existing `solve_ode` / `solve_ode_g` seam, generic over `PkNum` so it
serves both the `f64` prediction path and the `Dual2<N>` analytic-sensitivity path. Default
behaviour is unchanged (RK45); the stiff method is selected via
`[fit_options] ode_solver = rosenbrock`. This unlocks stiff model classes (TMDD, QSP, PBPK,
tight indirect-response loops) that the explicit Dormand–Prince RK45 resolves only by collapsing
to `min_dt` and exhausting the step budget (documented pain point at `types.rs:4432`).

**Why an algorithm, not an external crate.** Every mature Rust/C stiff solver (`diffsol` BDF/SDIRK,
SUNDIALS via `sundials-rs`) integrates concrete `f64` state through its own matrix types; none can
carry a `Dual2<N>` scalar. Wrapping one forces either finite-difference gradients (abandoning the
analytic-sensitivity design) or a separate CVODES sensitivity RHS with its *own* error control
(breaking the invariant that the sensitivity is the derivative of the scheme actually run). An
algorithm implemented generically over `PkNum` is the only way value and sensitivity come from the
same steps. `PkNum` already exposes full field arithmetic (`+ - * / neg`) plus `.val()`, so the
implicit linear solves run entirely in `T` and the `Dual2` derivatives propagate through the linear
algebra for free.

**Why Rosenbrock23 first (vs RODAS).** Rosenbrock methods are *linearly implicit* — each step is a
fixed sequence of linear solves with **no Newton iteration**, so there is no convergence loop to
differentiate through in `Dual2`. Rosenbrock23 is robust to an inexact (finite-difference) Jacobian
by design, matches the `reltol ≈ 1e-4` regime pharmacometrics uses, and has the smallest coefficient
set to validate to the project's "stable" bar. RODAS (order 4) needs an exact Jacobian to keep its
order — with an FD Jacobian it degrades toward order 3 while paying 6-stage cost — so it is deferred
to a later `OdeMethod` variant (paired with an exact Jacobian or a W-variant like ROS34PW2). The
`OdeMethod` enum, generic LU, and `W`-assembly are shared, so RODAS becomes mostly a new tableau on
top of this work, not a second project.

## 2. Invariants to preserve (non-negotiable)

1. **Seam signature unchanged.** `solve_ode_g(rhs, u0, t_span, params, saveat, opts)` and
   `solve_ode(...)` keep their exact signatures; the ~10 call sites in `sens/ode_provider.rs` and
   `ode/predictions.rs` are untouched. The method choice rides inside `opts`.
2. **Value-only step control.** Accept/reject and `dt` adaptation read `.val()` only, so `Dual2`
   derivatives flow through a *fixed* step sequence (the existing `solve_ode_g` invariant).
3. **Memoryless I-controller only.** Mirror the deliberate RK45 decision at `solver.rs:248–257`: no
   PI/Gustafsson history term. A history-dependent accept/reject makes the trajectory a noisier
   function of θ and stalls the FD FOCEI line search. This is the most important
   correctness-adjacent constraint.
4. **Sensitivity = derivative of the scheme actually run.** All Rosenbrock arithmetic — Jacobian
   assembly, `W = I − hdJ`, the LU, the stage solves — is done in `T`, so the `Dual2` components are
   the exact derivative of the numerical solution. No separate sensitivity RHS.
5. **Default = RK45, bit-for-bit.** `OdeMethod::default()` is `Rk45`; the existing body is entered
   unchanged unless the option is set.

## 3. Files touched

| File | Change |
|---|---|
| `src/ode/rosenbrock.rs` | **New.** Generic `rosenbrock23_g::<T: PkNum>(...)` core + a small dense LU-with-partial-pivoting over `T`. |
| `src/ode/solver.rs` | Add `pub enum OdeMethod { Rk45, Rosenbrock23 }`; add `method: OdeMethod` to `OdeSolverOptions` (stays `Copy`); dispatch at the top of `solve_ode_dense` and `solve_ode_g_with_stats`. |
| `src/ode/mod.rs` | `mod rosenbrock;` + re-export `OdeMethod`. |
| `src/types.rs` | Parse/store `ode_solver` in `FitOptions`; map it in `CompiledModel::sync_ode_solver_opts` (alongside the existing `ode_reltol/abstol/max_steps`). |
| `src/parser/model_parser.rs` | Accept `ode_solver = rk45 \| rosenbrock` key in `[fit_options]`; error on unknown values. |
| `docs/model-file/ode-models.qmd`, `docs/model-file/fit-options.qmd` | Document the option + when to reach for stiff. |
| `CHANGELOG.md` | `[Unreleased] → Added` entry with issue/PR ref. |

## 4. The algorithm (Rosenbrock23 / `ode23s`)

Constants: `d = 1/(2+√2) ≈ 0.2928932188`, `e32 = 6 + √2 ≈ 7.4142135624`.

Per step, given state `u`, time `t`, step `h`:

```
F0 = f(t, u)                          // reuse across a rejected step; FSAL-reuse F2→F0 across accepts
J  = fd_jacobian(f, t, u, F0)         // ∂f/∂u, column j = (f(u+εⱼeⱼ) − F0)/εⱼ, all in T
T0 = fd_time_deriv(f, t, u, F0)       // ∂f/∂t; = 0 within an autonomous segment (§6), else FD in t
W  = I − h·d·J                        // assembled in T
LU = lu_partial_pivot(W)              // one factorization, reused for all 3 solves; pivot on .val()

k1 = LU.solve(F0 + h·d·T0)
F1 = f(t + h/2, u + (h/2)·k1)
k2 = LU.solve(F1 − k1) + k1
u_new = u + h·k2                      // 2nd-order solution (the advancing solution)
F2 = f(t + h, u_new)
k3 = LU.solve(F2 − e32·(k2 − F1) − 2·(k1 − F0) + h·d·T0)

err_vec  = (h/6)·(k1 − 2·k2 + k3)     // 3rd-order-accurate error estimate
err_norm = rms_i( err_vec[i].val() / scale_tol(abstol, reltol, u_new[i].val(), u[i].val()) )
```

**Cost:** 3 RHS evals + `n` for the FD Jacobian (+`1` for `T0` when non-autonomous) + one `n×n` LU
per step. `n ≤ ~8`, so LU is negligible next to the `Dual2` arithmetic; the win is taking O(1) steps
where RK45 takes thousands.

**Step-size control** (memoryless I-controller, mirroring RK45 `solver.rs:376–386`):

```
p_hat  = 2                            // lower order of the embedded pair
factor = if !err_norm.is_finite() { NONFINITE_ERR_SHRINK_FACTOR }
         else if err_norm > 1e-15 { 0.9 * err_norm.powf(-1.0/(p_hat+1)) }   // 0.9 = safety
         else { 5.0 };
dt       = (h * factor.clamp(0.2, 5.0)).max(min_dt);
accepted = err_norm <= 1.0 || h <= min_dt;
```

**Generic dense LU over `T`** (new, small): Doolittle LU with partial pivoting, pivot row chosen by
`max |a_ij.val()|`; forward/back substitution in `T`. Singular pivot (`|pivot.val()| < tiny`) →
signal failure so the step is shrunk/rejected (§6). ~40 lines; the only place `.val()` is consulted
in the linear algebra, consistent with invariant #2.

## 5. Dispatch wiring

`solve_ode_dense` / `solve_ode_g_with_stats` gain a one-line `match opts.method`:

```rust
match opts.method {
    OdeMethod::Rk45         => { /* existing body, unchanged */ }
    OdeMethod::Rosenbrock23 => return rosenbrock23_g(rhs, u0, t_span, params, saveat, opts, stats),
}
```

`OdeSolverOptions { abstol, reltol, max_steps, initial_dt, min_dt, method }` — add `method`, default
`Rk45`, keep `#[derive(Copy)]`. `sync_ode_solver_opts` sets it from `FitOptions.ode_solver`. Parser
accepts the key with an unknown-value error (fail loudly, matching repo parser conventions).

## 6. Edge cases & safeguards (reuse RK45 policy)

- **Autonomous-segment shortcut:** the dose/infusion segment loop makes `f` autonomous *within* a
  segment (infusion rate constant, no dose events mid-segment), so `T0 = 0` and the FD-in-t eval is
  skipped. Keep the FD-`T0` path guarded by a flag for models with time-varying covariates /
  `MODEL_TIME` inside a segment.
- **Singular `W`:** near-zero pivot → treat like a failed step: shrink toward `min_dt` via
  `NONFINITE_ERR_SHRINK_FACTOR`, counting toward the same `MAX_CONSECUTIVE_MIN_STEP_CLAMPS` escape
  hatch so a pathological system terminates instead of spinning.
- **Non-finite error / diverging RHS:** identical handling to `solver.rs:322–371`
  (`nonfinite_min_step`, consecutive-clamp break).
- **`saveat` landing:** clamp `dt_eff` to not overshoot the next save time and push the state when
  `|t − saveat| < 1e-12`, exactly as RK45 (lines 267–270, 356–362). Fill trailing saveat with the
  last state on early break. Dense/Hermite interpolation is **out of scope** for v1 (Rosenbrock23's
  interpolant can be added later if soft-sampling is needed on the stiff path).
- **FD Jacobian scaling:** `εⱼ = √(f64::EPSILON) · max(|u_j.val()|, typ_j)` with a floor; perturbation
  and divisor are real scalars lifted via `T::from_f64`, so `∂J/∂p` propagates correctly.

## 7. Testing (per `CLAUDE.md` tiers + NONMEM rule)

**Tier 1 (unit, in `rosenbrock.rs`):**
1. **Value parity, non-stiff:** exp-decay and a 2-cpt system — Rosenbrock23 vs RK45 agree to `reltol`.
2. **Sensitivity correctness:** reuse the `solve_ode_g_sensitivity_matches_closed_form` template —
   `Dual2` output matches the closed-form `∂u/∂k`; and matches a finite-difference of the *value*
   path (the scheme's own derivative).
3. **Stiffness win:** a linear 2-state with eigenvalues `{−1, −10⁴}` (or Robertson-style) — assert
   Rosenbrock converges to tolerance with orders-of-magnitude fewer steps than RK45 via
   `OdeSolverStats`, and that RK45 hits `min_step_clamped_steps` while Rosenbrock does not.
4. **Generic LU:** round-trip `W·x = b` solve accuracy over `f64` and `Dual2`, including a pivoting
   case.

**Tier 2/3 (NONMEM-anchored):** a stiff PK/PD fit — TMDD quasi-steady-state or a rapid-equilibrium
2-cpt — run to convergence under `ode_solver = rosenbrock`, compared against **NONMEM `ADVAN13/14`**
output committed under `nonmem_anchor/` (θ/Ω/OFV within tolerance). Gate behind `slow-tests`.

## 8. Docs, changelog, cross-repo

- `ode-models.qmd`: new "Stiff systems / `ode_solver`" section with RK45-vs-Rosenbrock guidance and
  the NONMEM comparison table.
- `fit-options.qmd`: document the `ode_solver` key and default.
- `CHANGELOG.md`: `Added — stiff Rosenbrock23 ODE solver, opt-in via [fit_options] ode_solver = rosenbrock (#NN)`.
- `ferx-r`: if `OdeMethod` / the option is surfaced through a `pub` API, follow up with the matching
  wrapper PR and `tools/update-ferx-core-lock.sh` bump (never a bare `cargo update`).

## 9. Sequencing (each step compiles & tests green)

1. `OdeMethod` enum + `OdeSolverOptions.method` (default `Rk45`) + dispatch stubs → nothing behaves
   differently. _(small)_
2. Generic `T` LU + its unit test. _(small–medium)_
3. `rosenbrock23_g` core with `T0 = 0` autonomous path + value-parity and sensitivity unit tests.
   _(medium — the bulk)_
4. Non-finite/singular/min-step safeguards + stiff step-count test. _(small)_
5. `ode_solver` parse + `sync_ode_solver_opts` plumbing + parser test. _(small)_
6. NONMEM-anchored stiff fit test + docs + changelog. _(medium)_

Steps 1–4 change nothing observable until `ode_solver` is set (step 5), so the work can land
incrementally behind the default.

## 10. Risks

- **Order reduction if `T0` is wrongly zeroed** on a truly non-autonomous segment → keep the FD-`T0`
  guard and test a time-varying-covariate ODE.
- **`Dual2` LU cost at large `N`** (wide θ+η): O(n³·N) per step; fine for `n ≤ 8`, note it and revisit
  only if profiling flags it.
- **FD Jacobian conditioning** on very stiff systems — `√eps` scaling with a floor mitigates; if it
  bites, the exact-Jacobian route (via the existing `DualMixed` type) is the escape hatch, and it is
  also the door to a RODAS-4 variant later.
