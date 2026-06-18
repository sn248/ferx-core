# Plan: Analytic FOCE/FOCEI sensitivities for time-varying covariates (#367)

Last AD-parity gap for the analytical PK models. TV-covariates make the PK
parameters **switch mid-decay** (the covariate enters the individual-parameter
program, e.g. `CL = TVCL·(WT/70)^0.75·exp(ETA_CL)`, and `WT` changes over the
record), exactly like IOV — so the closed-form *superposition* provider cannot
represent them, but the **Dual2 event-driven walk already can**.

## Key insight: TV-cov ≈ IOV walk minus κ

The IOV provider (`subject_sensitivities_iov`) already does the hard part: it
seeds **per-event** `PkDual`s and walks the event schedule over `Dual2<M>`,
reading `∂conc/∂unknowns` straight off. TV-cov is the same machinery with two
simplifications and one addition:

- **No stacked κ.** The unknowns are just `(θ, η)`, so the dual width is
  `M = n_theta + n_eta` (no per-occasion κ blocks). The output `SubjectSens` has
  the **standard `(n_eta, n_theta)` shape**, identical to the non-TV provider.
- **Per-event covariates instead of per-occasion κ.** Each event's PK-param
  derivatives are evaluated at *that event's* covariate snapshot via the existing
  `pd_from_program::<M>(prog, model, cov, θ, η)` — which already accepts an
  arbitrary `cov: &HashMap` (the same call IOV uses per group).
- **Must handle `pk_only` (EVID=2) events.** Covariate changes that fall between
  observations are carried by EVID=2 records (`pk_only_times` /
  `pk_only_covariates`). The IOV provider bailed on these; the TV-cov walk must
  seed and walk them (the walk already accepts `pk_at_pk_only`).

Because the output shape is the standard one, the **entire outer + inner
assembly is reused unchanged** — no new gradient code. This is a much smaller
change than IOV: it is essentially one new provider function plus gate edits.

## Production reference (FD validation target)

`crate::pk::compute_predictions_with_tv` (→ `compute_event_pk_params_into`, which
builds per-event params at `dose_cov(k)` / `obs_cov(j)` / `pk_only_cov(m)`, then
runs `event_driven::event_driven_predictions`). The `f64` instance of
`event_driven_sens_g` already matches this bit-for-bit, so the Dual2 walk seeded
at per-event covariates must match its central differences.

## Work items

### 1. Provider: `subject_sensitivities_tvcov` (`src/sens/provider.rs`)
Mirror `subject_sensitivities_iov`, simplified:
- Gate: `analytical_supported(model) && subject.has_tv_covariates()`, no ODE, no
  SS doses (first cut — `d.ss` → fall back), no lagtime (first cut). Resets are
  **allowed** (the walk handles them; validate).
- `M = n_theta + n_eta`; dispatch over `M ∈ 1..=24` (reuse the IOV macro range).
- Per **event** (not per occasion-group): evaluate `pd_from_program::<M>` at the
  event's covariate map, build a `PkDual<Dual2<M>>` seeded on the `(θ, η)` axes
  (`grad[m]=∂p/∂θ_m`, `grad[n_theta+k]=∂p/∂η_k`, Hessian likewise). Cache by the
  covariate map's identity where cheap (consecutive equal-covariate events share
  a dual — common when covariates change only a few times).
- Build `pk_at_dose` / `pk_at_obs` / `pk_at_pk_only` from the per-event duals;
  run `event_driven_sens_g`; read `∂conc/∂(θ,η)` into `ObsSens` (standard shape,
  the `c.grad[m]`/`c.grad[n_theta+k]` mapping, same clamp parity as IOV).

### 2. Dispatch into the existing provider (`subject_sensitivities`)
In `subject_sensitivities` (and the light `subject_eta_grad`), replace the
`has_tv_covariates() → None` gate with: if `has_tv_covariates()`, delegate to the
TV-cov walk; else keep the current superposition / explicit-kernel path. The
returned `SubjectSens` shape is identical, so `sens_supported`, the outer
`population_gradient_sens[_foce]`, and the inner Jacobian all work untouched.
- **Inner loop (first cut):** keep FD inner for TV-cov subjects (analytic
  *outer*, FD *inner*) — mirrors the LTBS choice; add a first-order per-event
  light walk later if profiling wants it.

### 3. Validation (CLAUDE.md: every feature needs a test + a NONMEM comparison)
- **Provider unit tests** (`sens::provider::tests`): value/∂η/∂²η/∂θ/∂²η∂θ vs
  central FD of `compute_predictions_with_tv`, for 1-/2-/3-cpt, with (a)
  covariate change at an observation and (b) a covariate change carried by an
  EVID=2 breakpoint between observations, and (c) TV-cov **+ reset** together.
- **Outer packed-gradient tests** (`sens_outer_gradient::tests`): FOCEI + FOCE
  packed gradient vs Richardson reconverged-FD on a TV-cov subject (reuse the
  `precise_ebe` + `marginal_nll[_foce]` harness — standard shape, no IOV helper
  needed).
- **Tier-3 convergence test** (`tests/tvcov_convergence.rs`): deterministic
  dataset (time-varying weight on CL), fit with `lbfgs` to convergence,
  `FERX_SENS_CHECK` confirms the analytic gradient fires; recover truth.
- **NONMEM cross-check**: a `(WT/70)` on CL model with WT changing across
  occasions, FOCEI + FOCE on klebsiella; compare OFV + θ/Ω/σ. Commit the `.lst`
  under `tests/nonmem/`.

### 4. Docs + changelog
- `docs/src/model-file/individual-parameters.md` (or `faq.md`): note TV-covariate
  models now use the analytic gradient on `lbfgs`/`bfgs`/`slsqp`.
- CHANGELOG `Added`: extend the analytic-gradient scope entry; drop "time-varying
  covariates" from the fallback list (it is the last remaining item there).

## Scope boundaries (first cut → follow-ups)
- **In:** 1-/2-/3-cpt analytical, log-normal & program parameterizations, TV-cov
  on any PK parameter, EVID=2 covariate breakpoints, TV-cov + resets, FOCE+FOCEI.
- **Defer:** TV-cov **+ steady-state** doses (needs dual SS at per-event cov);
  TV-cov **+ lagtime**; analytic **inner** gradient for TV-cov subjects;
  TV-cov **+ IOV** simultaneously (would extend the IOV stacked walk to also vary
  covariates per event — natural but separate).

## Risks
- **Dual width `M = n_theta + n_eta`** can be larger than the non-TV path's
  `N ≤ 8` (PK-param count) for covariate-rich models — dispatch caps at 24 like
  IOV; beyond that, fall back to FD. Acceptable.
- **End-of-interval parameter convention**: the walk already mirrors production,
  so the param that governs each decay interval is the upcoming event's — no
  special handling, but the FD reference (`compute_predictions_with_tv`) pins it.
- **Per-event dual rebuild cost**: many distinct covariate values → many
  `pd_from_program` evaluations per subject. Cache consecutive-equal covariates;
  profile on a realistic dataset before optimizing further.

See [[iov-tvcov-fold-into-ode-367]] for the shared event-driven-walk rationale
and [[sens-dual2-clean-slate-367]] for the provider/outer architecture.
