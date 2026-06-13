# Changelog

All notable changes to **ferx-core** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(see the [Releases](https://ferx-nlme.github.io/ferx-core/development/sdlc.html#9-releases)
section of the SDLC for the versioning policy).

<!--
  HOW TO MAINTAIN THIS FILE
  - Add an entry under [Unreleased] in the same PR as your change, in the right
    category (Added / Changed / Deprecated / Removed / Fixed / Security /
    Performance). One bullet, user-facing language, reference the issue/PR (#NN).
  - At release time: rename [Unreleased] to the new version + date, then start a
    fresh empty [Unreleased] and update the compare links at the bottom.
  - The R wrapper (ferx-r) tracks its own user-facing changes in NEWS.md.
-->

## [Unreleased]

### Added
- Propensity-score-matched simulation: `simulate_with_options()` with a new
  `SimulateOptions { seed, propensity_match }`. When `propensity_match` is set,
  each replicate's drawn etas are reassigned to subjects by optimal Mahalanobis
  matching (under the model Ω) against the subjects' fitted (posthoc) etas, so a
  subject's observed dosing/sampling design is paired with a similar drawn eta.
  This corrects VPC bias from treatment adaptation in real-world data (longer
  intervals for high-clearance patients, etc.). Operates on observed data;
  returns the usual simulation rows for the caller to build the VPC (#288).
- New `importance_sampling_map` (alias `impmap`) estimation method: a Monte-Carlo
  EM estimator equivalent to NONMEM `METHOD=IMPMAP`. Each iteration re-centers a
  per-subject importance-sampling proposal on the conditional mode (MAP) and
  updates θ/Ω/σ from the importance-weighted posterior moments. Runs standalone
  or chained (`methods = [focei, impmap]`); multivariate-normal proposal by
  default (`impmap_proposal_df = normal`), Student-t optional. Validated against
  FOCEI on warfarin. IOV and SDE models are not yet supported (#270).
- Importance sampling can now run **standalone** (`method = imp`), evaluating the
  IS log-likelihood at the initial parameters — IMP derives the EBEs/Jacobian it
  needs via a FOCE inner loop at those parameters instead of requiring a
  preceding estimator. Useful for scoring imported/fixed parameter sets. IMP
  still may appear at most once and must be the terminal stage of a chain.
- Feature maturity labels (`stable` / `beta` / `experimental`) documented for
  every major feature: a new *Feature Maturity* docs page with definitions and a
  per-feature table, plus a maturity banner on each feature reference page.
  Experimental features (`[diffusion]` / SDE, `[covariate_nn]` / neural networks)
  now emit a runtime warning at fit time (`W_EXPERIMENTAL_SDE`,
  `W_EXPERIMENTAL_NN`), also surfaced by `ferx check` (#175).
- `covariance_method` fit option: choose the covariance estimator, mirroring
  NONMEM `$COV MATRIX=` — `r` (inverse Hessian `R⁻¹`, default), `s` (inverse
  score cross-product `S⁻¹`), or `rsr` (the Huber–White sandwich `R⁻¹SR⁻¹`,
  robust to model mis-specification). Supported for FOCEI/IOV fits (#223).
- `covariance_fallback = sir` fit option: when the FD Hessian is non-positive-definite,
  run SIR with an `|eigenvalue|`-rectified proposal (4× inflated) instead of leaving
  the covariance step as failed; `covariance_status` reports `sir_fallback` (#223).
- `covariance_matrix:` block in `*-fit.yaml`: the full optimizer-space parameter
  covariance matrix (log-theta, Cholesky-omega, log-sigma; kappa appended for IOV
  models), parameter-labelled, emitted when the covariance step succeeds or is
  regularised. Omega/kappa diagonal entries are keyed `log_chol_<eta>` (packed
  value is `log(L_ii)`); off-diagonal entries are keyed `chol_<row>_<col>`
  (`L_ij`, not log-transformed) (#236).
- Time-to-event / survival modelling (Phase 1): `[event_model]` block, TTE
  datareader, likelihood, and API wiring, behind the `survival` feature
  (#191, #192).
- `[data_selection]` block with NONMEM-style `IGNORE`/`ACCEPT` record filtering,
  plus an `ExclusionSummary` on `FitResult` surfaced in the CLI and YAML output.
- Combined ferx-core + ferx-r development documentation: a Development Lifecycle
  (SDLC) page and a Contributing page in the book.

### Changed
- IMP (importance sampling) now jointly samples (η, κ) for IOV models,
  integrating over inter-occasion variability so the reported `−2 log L` is
  directly comparable to FOCE/FOCEI and NONMEM `METHOD=IMP`. Previously κ was
  held fixed at its EBE mode, giving a partial marginal; `kappa_treatment` in
  the fit YAML is now `marginalized` rather than `fixed_at_mode` (#186).

### Fixed
- A `[structural_model]` PK parameter that references a name not defined in
  `[individual_parameters]` (e.g. `pk one_cpt_oral(cl=CL, ...)` with no `CL`)
  is now a parse error instead of being silently dropped and defaulting the
  slot to 0.0 — which previously produced a "converged" but structurally broken
  fit (all predictions floored, 100% shrinkage). An unrecognized PK-parameter
  key (e.g. the typo `clx=`) is likewise rejected, and a numeric-literal value
  (e.g. `ka=1.0`) is now honored as a constant rather than dropped to 0.0 (#261).
- FOCEI covariance standard errors (non-IOV) now include the `log|H̃|` EBE-response
  curvature for mu-referenced structural parameters, bringing the non-IOV stencil
  in line with the IOV stencil and matching NONMEM `$COV MATRIX=R` more closely on
  models with η-dependent (proportional/combined) residual error. The fixed-η̂
  analytic gradient previously dropped this term — the envelope theorem zeros the
  inner objective but not `log|H̃|` — and the resulting SE gap grew with the
  proportional error magnitude. Additive-error SEs are unchanged (the correction is
  identically zero when `∂R/∂f = 0`) (#274).
- IOV models: `[derived]` columns, `[output]` individual parameters, and the
  TAD column in `sdtab` now use each observation's **occasion** kappa instead of
  silently treating every kappa as zero. Post-fit diagnostic columns that depend
  on a κ-varying parameter (e.g. `CL`, `V`, `KA`) were wrong for IOV subjects;
  the fitted estimates, OFV, and IPRED/IWRES were unaffected (#238).
- The `sdtab` TAD column now shifts each dose by **its own** absorption lag —
  evaluated with that dose's occasion kappa and that dose's covariate snapshot —
  rather than applying the observation's lag to every dose. This changes TAD only
  when the absorption lag varies across doses, i.e. when it carries IOV (kappa) or
  depends on a time-varying covariate, *and* dosing spans the differing values
  (e.g. BID across two occasions); models with a constant lag are unaffected
  (follow-up to #238).
- FOCE (non-interaction) omega standard errors now match NONMEM `$EST METHOD=1`
  `$COVARIANCE MATRIX=R` (to ~3–6% on warfarin, previously ~31% low). The
  covariance step had added the Ω prior (`η̂ᵀΩ⁻¹η̂ + log|Ω|`) on top of the
  Sheiner–Beal marginal, which already carries Ω through `R̃ = HΩHᵀ + R` —
  double-counting Ω and flattening the omega-block curvature. FOCE estimates were
  already correct; only the SEs were affected (#243).
- The covariance step now succeeds on models with a mixed block + diagonal Ω: the
  structural-zero cross-block off-diagonals (`free_mask == false`) are excluded
  from the parameter set like FIX parameters, so their flat Hessian diagonal no
  longer aborts the step. This affected both FOCE and FOCEI (#243).
- Covariance standard errors now match NONMEM `$COVARIANCE MATRIX=R` (within ~2%
  on warfarin). The covariance step reconverges the inner EBE loop at every
  finite-difference point — holding the EBEs fixed gave an indefinite Hessian
  that was clipped and inflated theta/sigma SEs 30–94× — and applies the correct
  factor of two for the `−2·logL` objective (every SE was previously `1/√2` too
  small) (#209, #196, #129).
- Covariance step: `fd_hessian_step` is now an *initial* step; ferx automatically
  halves it up to 8× if any diagonal FD stencil is non-finite (#223).
- IOV FOCEI marginal likelihood now matches NONMEM after the Almquist Laplace
  correction (#109, #203).
- SAEM no longer collapses a block Ω to a rank-1 (near-unit-correlation)
  solution (#191).
- Stacked `EVID=4` reset occasions are segmented onto a monotonic timeline
  (#195, #197).
- `sdtab` no longer emits stray ETA columns (regression from #185).
- `warfarin --simulate` works again, and the docs `verify-build` step is fixed
  (#199, #200).

### Performance
- The covariance step is now built as a single parallel work-list over the
  finite-difference points (subjects iterated serially within each point) instead
  of firing a per-subject parallel reduction at every perturbed point. This removes
  the fork/join overhead of up to `4·n_free` rayon barriers in series — the
  bottleneck was scheduling, not core utilisation — making the covariance step
  ~9–11× faster across error models and structures, with bit-identical results.
  Both stencils are flattened: the non-IOV analytic-gradient difference and the
  IOV `OFV`-second-difference (the latter has `~2·n_free²` points, so it benefits
  even more) (#256).
- The covariance Hessian is built from a central difference of the analytical
  population gradient — reusing H-matrix columns for mu-referenced parameters
  instead of finite-differencing predictions — making the covariance step ~9×
  faster than scalar finite differencing on warfarin, scaling with the number of
  free parameters (#209, #196).
- Autodiff inner gradients now flow through `EVID=3/4` resets and lag time,
  removing a large finite-difference fallback slowdown (#198).

## [0.1.5] - 2026-06-01

Released before this changelog was started. See the
[GitHub release](https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.5)
and `git log v0.1.0..v0.1.5` for details.

## [0.1.0] - 2026-05-29

Initial tagged release. See the
[GitHub release](https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.0).

[Unreleased]: https://github.com/FeRx-NLME/ferx-core/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/FeRx-NLME/ferx-core/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.0
