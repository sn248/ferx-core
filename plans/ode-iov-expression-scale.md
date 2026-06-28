# ODE IOV + `ExpressionScale` `obs_scale` analytic gradient (#575)

## Problem
ODE models with IOV (`n_kappa > 0`) **and** an `ExpressionScale` `obs_scale`
divisor (e.g. `obs_scale = V1`) route the analytic sensitivity gradient to FD on
both loops. IOV alone is analytic (#466); `ExpressionScale obs_scale` alone is
analytic (#534, non-IOV static walk). The combination is gated out.

Gate: `ode_provider.rs:483` (`ode_iov_supported`):
```rust
if !matches!(model.scaling, ScalingSpec::None) || model.log_transform { return false; }
```

## Approach
Mirror #534. The `ExpressionScale` divisor is applied as a **post-walk quotient**
on the `SubjectSens` (outer) / `Vec<ObsGrad>` (inner) — NOT baked into the walk.
For IOV the scale jet is **per occasion group** (the divisor depends on the
group's κ through the PK params), unlike the subject-static non-IOV case.

Axis layout (already used by the IOV walk): stacked dual
`(θ[0..n_theta], η_bsv[0..n_eta], κ_g·n_kappa…)`, total `M = n_theta + n_stacked`,
`n_stacked = n_eta + K·n_kappa`. The `SubjectSens` obs rows carry `df_deta`,
`d2f_deta2`, `d2f_deta_dtheta` over `n_stacked`. The quotient math is the same as
`apply_expression_scale` with `n_eta → n_stacked`.

The scale's own jet over stacked axes comes from `prog.eval_scale_dual::<M>(theta,
eta_bsv, cov, var_duals)` where `eta_bsv = stacked_eta[0..n_eta]` and `var_duals`
are the scale-referenced PK slots seeded over the stacked axes via the existing
`seed_pk_dual2_iov` (outer) / `seed_pk_dual1_iov` (inner). `eval_scale_dual` seeds
θ on `0..n_theta` and BSV η on `n_theta+k`; κ derivatives flow purely through
`var_duals`. Consistent with the stacked layout.

## Changes (all in `src/sens/`)

1. **Gate** `ode_iov_supported` (`ode_provider.rs:483`): allow `None` and
   `ExpressionScale { deriv: Some(p), .. }` with `!log_transform`,
   `p.n_theta_axis()==n_theta`, `p.n_eta_axis()==n_eta`, `n_axes()∈1..=MAX_ODE_AXES`.
   Still decline `ScalarScale`/LTBS (out of scope; separate gap).

2. **Per-subject gate** `ode_iov_subject_supported` (`ode_provider.rs:~1753`):
   decline `ExpressionScale` + `subject.has_tv_covariates()` (per-event scale jet
   not handled — matches `ode_tvcov_supported` declining ExpressionScale).

3. **Outer** `run_subject_iov`: after building `SubjectSens`, if `ExpressionScale`,
   apply per-group quotient. New helper `apply_expression_scale_iov::<M>` (in
   ode_provider.rs): for each occasion group, build var_duals from
   `seed_pk_dual2_iov`, eval scale jet, apply quotient (n_stacked η axes) to that
   group's obs rows.

4. **Inner** `run_subject_iov_eta`: same, η-only first-order quotient via
   `seed_pk_dual1_iov` + `eval_scale_dual1::<N>` (N = n_stacked), per group.

   Both helpers reuse `iov_combined_derivs_dyn` / `combined_for` exactly as the
   walk seeding does, so the scale's PK derivatives match the walk's.

## Tests (tier 1 unit, in ode_provider.rs `#[cfg(test)]`)
- `ode_iov_expr_scale_outer_matches_fd`: outer packed gradient vs Richardson FD on
  a 1-/2-cpt ODE IOV model with `obs_scale = V` — ≤1e-6 rel (mirror
  `population_packed_gradient_ode_*_matches_fd`).
- `ode_iov_expr_scale_inner_matches_fd`: inner η-gradient vs FD of `individual_nll`.
- `ode_iov_expr_scale_equals_formc`: estimates/gradient with `obs_scale = central/V`
  Form-C readout (already analytic) == `obs_scale = V` divisor (numerical twin).
- Gate unit: `ode_iov_supported` true for ExpressionScale model, false under LTBS.

## NONMEM / parity
Cross-check against the Form-C-readout variant (already analytic, NONMEM-validated
path) — identical estimates confirms the divisor quotient. Note in PR / docs.

## Out of scope (follow-ups)
- Analytical-PK (closed-form) IOV + ExpressionScale (`iov_analytical_supported`,
  provider.rs:639) has the same gap — separate.
- IOV + ScalarScale / LTBS.
- IOV + ExpressionScale + TV-cov (per-event jet).
