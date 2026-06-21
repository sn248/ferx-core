//! Tier-3 convergence + simulation-estimation (SSE) tests for Phase 1 TTE.
//!
//! Gated behind BOTH `survival` (the TTE feature) and `slow-tests` (these run a
//! full fit to convergence, so they are skipped on the per-PR `Test` job and run
//! nightly via `slow-tests.yml` — see the test-tier rules in CLAUDE.md).
//!
//! Two kinds of guard:
//!
//!   * **SSE** (`tte_sse_*`): generate a dataset from known `(theta, Omega)` with
//!     ferx's *own* `simulate()`, refit with ferx, and assert parameter recovery.
//!     This is the primary guard on the generative path (plan §14.11): a wrong
//!     event-time sampler or a wrong likelihood constant shows up here as
//!     non-recovery, which a fit-only test cannot detect.
//!
//!   * **Cross-tool** (`tte_convergence_*`): fit the committed reference dataset
//!     `tests/reference/tte_exponential/tte_exp.csv` — the *same* file NONMEM and
//!     nlmixr2 fit — through the real datareader (`read_population_for`, so the
//!     DV→`obs_records` routing is exercised too) and assert the estimates land in
//!     the documented bands. The fixed-effects (`n_eta = 0`) fit is checked
//!     *exactly* against the base-R `survival::survreg` closed-form MLE
//!     (`lambda = events / sum(time)`); the mixed-effects fit must instead recover
//!     the data-generating `lambda_pop = 0.1`, `omega^2 = 0.25` (the pooled
//!     fixed-effects rate is biased low by the ignored between-subject variance —
//!     which is the whole point of fitting the mixed model).

#![cfg(all(feature = "survival", feature = "slow-tests"))]

mod common;

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{EventType, ObsRecord, Population};
use ferx_core::{fit, simulate_with_seed, FitOptions, SimOutcome};

// ── Model strings ────────────────────────────────────────────────────────────

/// Data-generating ("truth") model used by the SSE test: lambda_pop = 0.1,
/// omega^2 = 0.25 on log(lambda). `scale` is the Exponential rate.
const EXP_TRUTH: &str = r"
[parameters]
  theta TVLAMBDA(0.1, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.25

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// Mixed-effects fit model, initialised *away* from the truth so recovery is a
/// real test (rate 2x low, variance ~3x low).
const EXP_FIT: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

/// Fixed-effects (n_eta = 0) fit model — ordinary parametric Exponential PH, the
/// plain-likelihood special case (plan D7). Anchored against `survreg`.
const EXP_FIT_FIXED: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
";

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fit_opts() -> FitOptions {
    // Default is already FOCEI / outer_maxiter = 500 / interaction = true — i.e.
    // run-to-convergence. We only quiet it and request the covariance step so the
    // comparison table can report SEs.
    FitOptions {
        verbose: false,
        run_covariance_step: true,
        ..FitOptions::default()
    }
}

/// Build a TTE-only population from `(time, dv)` pairs. `dv`: 1 = exact event,
/// 0 = right-censored. All rows route to CMT 2.
fn tte_pop_from_pairs(data: &[(f64, u8)]) -> Population {
    let subjects = data
        .iter()
        .enumerate()
        .map(|(i, &(t, dv))| {
            let event_type = if dv == 1 {
                EventType::Exact
            } else {
                EventType::RightCensored
            };
            let mut s = common::subject(&format!("{}", i + 1), vec![], vec![], vec![], vec![]);
            s.obs_records = vec![ObsRecord::Event {
                time: t,
                event_type,
                entry_time: 0.0,
                cmt: 2,
            }];
            s
        })
        .collect();

    Population {
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

/// One bare TTE subject (a single Event record on CMT 2) used only as a template
/// for `simulate()` — the drawn event time replaces the placeholder.
fn tte_sim_template(n: usize) -> Population {
    let subjects = (0..n)
        .map(|i| {
            let mut s = common::subject(&format!("{}", i + 1), vec![], vec![], vec![], vec![]);
            s.obs_records = vec![ObsRecord::Event {
                time: 0.0,
                event_type: EventType::Exact,
                entry_time: 0.0,
                cmt: 2,
            }];
            s
        })
        .collect();
    Population {
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

const REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_exponential/tte_exp.csv"
);

/// `survival::survreg(Surv(TIME,DV) ~ 1, dist="exponential")` on `tte_exp.csv`.
/// Closed form `lambda = events / sum(time) = 82 / 1100.6`. Regenerate via
/// `tests/reference/tte_exponential/survreg.R`.
const SURVREG_LAMBDA: f64 = 0.074506;

// ── Tests ────────────────────────────────────────────────────────────────────

/// SSE: simulate from (lambda_pop=0.1, omega^2=0.25) with ferx's own `simulate()`,
/// apply administrative censoring at t=24 (matching the reference design), refit,
/// and assert both parameters are recovered. Guards the event-time sampler and the
/// censored-likelihood together.
#[test]
fn tte_sse_exponential_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 24.0;
    const SEED: u64 = 20260621;

    let truth = parse_model_string(EXP_TRUTH).expect("truth model must parse");
    let template = tte_sim_template(N);

    let sims = simulate_with_seed(&truth, &template, &truth.default_params, 1, SEED);
    assert_eq!(sims.len(), N, "one simulated event per template subject");

    // simulate_tte draws *uncensored* event times (observed=true for every draw);
    // apply the administrative censoring ourselves, exactly as simulate.R does.
    let pairs: Vec<(f64, u8)> = sims
        .iter()
        .map(|r| match r.outcome {
            SimOutcome::Event { time, .. } => {
                if time <= T_CENSOR {
                    (time, 1)
                } else {
                    (T_CENSOR, 0)
                }
            }
            _ => panic!("expected an Event outcome for a TTE simulation"),
        })
        .collect();

    let n_events = pairs.iter().filter(|(_, dv)| *dv == 1).count();
    assert!(
        (0.5..0.98).contains(&(n_events as f64 / N as f64)),
        "sanity: simulated event fraction {}/{} should be neither ~0 nor ~1",
        n_events,
        N
    );

    let model = parse_model_string(EXP_FIT).expect("fit model must parse");
    let pop = tte_pop_from_pairs(&pairs);
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("SSE fit must succeed");

    let lambda = r.theta[0];
    let omega2 = r.omega[(0, 0)];
    eprintln!("[SSE] lambda_pop = {lambda:.5} (truth 0.1), omega^2 = {omega2:.5} (truth 0.25), OFV = {:.4}", r.ofv);

    // N is large enough that single-replicate Monte-Carlo noise is small, so these
    // bands test the *estimator* (near-unbiased recovery), not luck. Observed
    // (seed-fixed, deterministic): lambda ~= 0.099, omega^2 ~= 0.232 — the ~7%
    // omega^2 shortfall is the expected mild FOCEI-Laplace bias for TTE at this
    // event rate (plan §3.3), and grows at lower event rates. Bands are wide
    // enough to absorb that bias yet catch a gross sampler/likelihood error
    // (wrong constant, sign flip, factor-of-2).
    assert!(
        (0.090..0.110).contains(&lambda),
        "lambda_pop not recovered: got {lambda:.5}, expected ~0.1"
    );
    assert!(
        (0.200..0.275).contains(&omega2),
        "omega^2 not recovered: got {omega2:.5}, expected ~0.25 (mild Laplace bias allowed)"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool: mixed-effects FOCEI fit of the committed reference dataset. Must
/// recover the data-generating parameters (this is the row NONMEM/nlmixr2 fill).
#[test]
fn tte_convergence_exponential_mixed() {
    let model = parse_model_string(EXP_FIT).expect("fit model must parse");
    let (pop, _cov) =
        read_population_for(&model, &None, REF_CSV, None, None, None).expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 100, "reference dataset is 100 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");

    let lambda = r.theta[0];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[mixed] lambda_pop = {lambda:.5} (truth 0.1), omega^2 = {omega2:.5} (truth 0.25), OFV = {:.4}, converged = {}",
        r.ofv, r.converged
    );

    // Single 100-subject realisation (seed 42) is noisy, so the exact estimates
    // (deterministic: lambda ~= 0.0768, omega^2 ~= 0.290, OFV ~= 588.93) are what
    // gets tabulated next to NONMEM/nlmixr2 in
    // `tests/reference/tte_exponential/expected.md` — every tool fits *this* file,
    // so they should agree with each other regardless of the 0.1/0.25 truth. The
    // bands here are only a gross-failure guard.
    assert!(
        (0.06..0.14).contains(&lambda),
        "lambda_pop off: got {lambda:.5}, expected ~0.1"
    );
    assert!(
        (0.12..0.50).contains(&omega2),
        "omega^2 off: got {omega2:.5}, expected ~0.25"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool, exact: fixed-effects (n_eta=0) Exponential MLE must match the
/// base-R `survreg` closed-form rate on the same dataset to a tight tolerance.
#[test]
fn tte_convergence_exponential_fixed_matches_survreg() {
    let model = parse_model_string(EXP_FIT_FIXED).expect("fixed-effects model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");

    let (pop, _cov) =
        read_population_for(&model, &None, REF_CSV, None, None, None).expect("reference CSV reads");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let lambda = r.theta[0];
    eprintln!(
        "[fixed] lambda = {lambda:.6} (survreg {SURVREG_LAMBDA:.6}), OFV = {:.4}",
        r.ofv
    );

    let rel_err = (lambda - SURVREG_LAMBDA).abs() / SURVREG_LAMBDA;
    assert!(
        rel_err < 0.01,
        "fixed-effects rate {lambda:.6} must match survreg {SURVREG_LAMBDA:.6} within 1% (rel_err {rel_err:.4})"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

// ═══════════════════════════ Weibull ═══════════════════════════════════════════
//
// Validation dataset puts between-subject variability on the *shape* (scale fixed):
// scale_pop = 20, shape_pop = 2, omega^2(log shape) = 0.20, censored at t = 30.

const WEIBULL_TRUTH: &str = r"
[parameters]
  theta TVSCALE(20.0, 0.1, 500.0)
  theta TVSHAPE(2.0,  0.1, 10.0)
  omega ETA_SHAPE ~ 0.20

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE * exp(ETA_SHAPE)
";

const WEIBULL_FIT: &str = r"
[parameters]
  theta TVSCALE(15.0, 0.1, 500.0)
  theta TVSHAPE(1.2,  0.1, 10.0)
  omega ETA_SHAPE ~ 0.05

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE * exp(ETA_SHAPE)
";

const WEIBULL_FIT_FIXED: &str = r"
[parameters]
  theta TVSCALE(15.0, 0.1, 500.0)
  theta TVSHAPE(1.2,  0.1, 10.0)

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE
";

const WEIBULL_REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_weibull/tte_weibull.csv"
);

// survreg(Surv(TIME,DV)~1, dist="weibull") on tte_weibull.csv, mapped to ferx's
// (scale, shape) via shape = 1/scale_sr, scale = exp(intercept). See survreg.R.
const SURVREG_WEIBULL_SHAPE: f64 = 2.119250;
const SURVREG_WEIBULL_SCALE: f64 = 22.176599;

/// SSE: recover (scale=20, shape=2, omega^2=0.20) from a large simulated dataset.
#[test]
fn tte_sse_weibull_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 30.0;
    const SEED: u64 = 20260622;

    let truth = parse_model_string(WEIBULL_TRUTH).expect("truth model must parse");
    let sims = simulate_with_seed(&truth, &tte_sim_template(N), &truth.default_params, 1, SEED);

    let pairs: Vec<(f64, u8)> = sims
        .iter()
        .map(|r| match r.outcome {
            SimOutcome::Event { time, .. } => {
                if time <= T_CENSOR {
                    (time, 1)
                } else {
                    (T_CENSOR, 0)
                }
            }
            _ => panic!("expected an Event outcome"),
        })
        .collect();

    let model = parse_model_string(WEIBULL_FIT).expect("fit model must parse");
    let r = fit(
        &model,
        &tte_pop_from_pairs(&pairs),
        &model.default_params,
        &fit_opts(),
    )
    .expect("SSE fit must succeed");

    let scale = r.theta[0];
    let shape = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[SSE weibull] scale = {scale:.4} (truth 20), shape = {shape:.4} (truth 2), omega^2 = {omega2:.4} (truth 0.20)"
    );

    // Structural parameters recover tightly under FOCEI.
    assert!(
        (18.5..21.5).contains(&scale),
        "scale not recovered: got {scale:.4}, expected ~20"
    );
    assert!(
        (1.85..2.15).contains(&shape),
        "shape not recovered: got {shape:.4}, expected ~2"
    );
    // omega^2 (frailty on the *shape* — a nonlinear hazard parameter) is materially
    // OVER-estimated by FOCEI-Laplace: observed ~0.35 vs truth 0.20 (+~75%), and it
    // does NOT vanish as the true omega^2 shrinks (truth 0.05 → ~0.16). A SAEM fit of
    // the same data gives ~0.13 — i.e. the estimators straddle the truth, confirming
    // a FOCEI approximation limitation for nonlinear-parameter frailty (plan §3.3/§13:
    // SAEM/IMP preferred for TTE), not a likelihood bug (fixed-effects matches survreg
    // exactly). Tracked in #440. The assertion is therefore only a sane-range guard
    // (omega^2 neither collapsed to ~0 nor exploded), wide enough to also pass once the
    // estimator is improved — see expected.md for the documented numbers.
    assert!(
        (0.12..0.55).contains(&omega2),
        "omega^2 out of sane range: got {omega2:.4} (FOCEI over-estimates nonlinear-frailty omega^2; truth 0.20, see #440)"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool, exact: fixed-effects (n_eta=0) Weibull MLE must match `survreg`
/// (both scale and shape) on the same dataset.
#[test]
fn tte_convergence_weibull_fixed_matches_survreg() {
    let model = parse_model_string(WEIBULL_FIT_FIXED).expect("fixed-effects model must parse");
    assert_eq!(model.n_eta, 0, "fixed-effects model must have no etas");

    let (pop, _cov) = read_population_for(&model, &None, WEIBULL_REF_CSV, None, None, None)
        .expect("reference CSV reads");
    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");

    let scale = r.theta[0];
    let shape = r.theta[1];
    eprintln!(
        "[fixed weibull] scale = {scale:.4} (survreg {SURVREG_WEIBULL_SCALE:.4}), shape = {shape:.4} (survreg {SURVREG_WEIBULL_SHAPE:.4}), OFV = {:.4}",
        r.ofv
    );

    let scale_err = (scale - SURVREG_WEIBULL_SCALE).abs() / SURVREG_WEIBULL_SCALE;
    let shape_err = (shape - SURVREG_WEIBULL_SHAPE).abs() / SURVREG_WEIBULL_SHAPE;
    assert!(
        scale_err < 0.01,
        "scale {scale:.4} must match survreg {SURVREG_WEIBULL_SCALE:.4} within 1% (rel_err {scale_err:.4})"
    );
    assert!(
        shape_err < 0.01,
        "shape {shape:.4} must match survreg {SURVREG_WEIBULL_SHAPE:.4} within 1% (rel_err {shape_err:.4})"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// Cross-tool: mixed-effects (frailty-on-shape) FOCEI fit of the committed Weibull
/// dataset — the row NONMEM/nlmixr2 fill. Records ferx's estimates; the FOCEI
/// over-estimation of the shape-frailty omega^2 (#440) shows up here on real data too.
#[test]
fn tte_convergence_weibull_mixed() {
    let model = parse_model_string(WEIBULL_FIT).expect("fit model must parse");
    let (pop, _cov) = read_population_for(&model, &None, WEIBULL_REF_CSV, None, None, None)
        .expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 100, "reference dataset is 100 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let scale = r.theta[0];
    let shape = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[weibull mixed] scale = {scale:.4} (truth 20), shape = {shape:.4} (truth 2), omega^2 = {omega2:.4} (truth 0.20), OFV = {:.4}",
        r.ofv
    );

    // Gross-failure guard; exact estimates are tabulated in expected.md.
    assert!(
        (15.0..26.0).contains(&scale),
        "scale off: got {scale:.4}, expected ~20"
    );
    assert!(
        (1.5..2.8).contains(&shape),
        "shape off: got {shape:.4}, expected ~2"
    );
    assert!(
        (0.05..0.70).contains(&omega2),
        "omega^2 off: got {omega2:.4}, expected ~0.20 (FOCEI over-estimates; see #440)"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

// ═══════════════════════════ Gompertz ══════════════════════════════════════════
//
// Validation dataset is a fixed-effects 2-arm RCT: h = alpha*exp(gamma*t)*exp(loghr*TRT),
// alpha = exp(-6) ≈ 0.00248, gamma = exp(-5.4) ≈ 0.00450, loghr = -0.8, censored at 365.
// No survreg/base-R anchor exists for Gompertz, so recovery is the guard (NONMEM/nlmixr2
// are the cross-tool hand-off). The fit exercises the [event_model] covariate + loghr path.

const GOMPERTZ_FIT: &str = r"
[parameters]
  theta TVALPHA(0.001, 1e-6, 1.0)
  theta TVGAMMA(0.003, 1e-5, 5.0)
  theta LHR(-0.3, -5.0, 5.0)

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA
  loghr  = LHR * TRT
";

const GOMPERTZ_REF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/reference/tte_gompertz/tte_gompertz.csv"
);

/// Frailty-Gompertz SSE truth (BSV on gamma) — recover alpha, gamma, omega^2.
const GOMPERTZ_TRUTH_FRAILTY: &str = r"
[parameters]
  theta TVALPHA(0.002, 1e-6, 1.0)
  theta TVGAMMA(0.05,  1e-5, 5.0)
  omega ETA_GAMMA ~ 0.05

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA * exp(ETA_GAMMA)
";

const GOMPERTZ_FIT_FRAILTY: &str = r"
[parameters]
  theta TVALPHA(0.001, 1e-6, 1.0)
  theta TVGAMMA(0.03,  1e-5, 5.0)
  omega ETA_GAMMA ~ 0.02

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA * exp(ETA_GAMMA)
";

/// Cross-tool: fixed-effects Gompertz RCT fit on the committed dataset. Recovers
/// the data-generating alpha/gamma/loghr and exercises the TRT covariate path.
/// (NONMEM/nlmixr2 columns are the hand-off.)
#[test]
fn tte_convergence_gompertz_rct_recovers() {
    let model = parse_model_string(GOMPERTZ_FIT).expect("gompertz model must parse");
    assert_eq!(model.n_eta, 0, "RCT model is fixed-effects");
    assert!(
        model.referenced_covariates.contains(&"TRT".to_string()),
        "TRT must be picked up from the loghr expression"
    );

    let (pop, _cov) = read_population_for(&model, &None, GOMPERTZ_REF_CSV, None, None, None)
        .expect("reference CSV reads");
    assert_eq!(pop.subjects.len(), 300, "reference dataset is 300 subjects");

    let r = fit(&model, &pop, &model.default_params, &fit_opts()).expect("fit must succeed");
    let alpha = r.theta[0];
    let gamma = r.theta[1];
    let loghr = r.theta[2];
    eprintln!(
        "[gompertz RCT] alpha = {alpha:.6} (truth 0.00248), gamma = {gamma:.6} (truth 0.00450), loghr = {loghr:.4} (truth -0.8), OFV = {:.4}",
        r.ofv
    );

    assert!(
        (0.0012..0.0045).contains(&alpha),
        "alpha off: got {alpha:.6}, expected ~0.00248"
    );
    assert!(
        (0.0025..0.0070).contains(&gamma),
        "gamma off: got {gamma:.6}, expected ~0.00450"
    );
    assert!(
        (-1.3..-0.3).contains(&loghr),
        "loghr off: got {loghr:.4}, expected ~-0.8"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}

/// SSE: frailty-Gompertz (BSV on gamma) recovery of alpha/gamma/omega^2.
#[test]
fn tte_sse_gompertz_recovers_truth() {
    const N: usize = 2000;
    const T_CENSOR: f64 = 80.0;
    const SEED: u64 = 20260623;

    let truth = parse_model_string(GOMPERTZ_TRUTH_FRAILTY).expect("truth model must parse");
    let sims = simulate_with_seed(&truth, &tte_sim_template(N), &truth.default_params, 1, SEED);

    let pairs: Vec<(f64, u8)> = sims
        .iter()
        .map(|r| match r.outcome {
            SimOutcome::Event { time, .. } => {
                if time <= T_CENSOR {
                    (time, 1)
                } else {
                    (T_CENSOR, 0)
                }
            }
            _ => panic!("expected an Event outcome"),
        })
        .collect();

    let model = parse_model_string(GOMPERTZ_FIT_FRAILTY).expect("fit model must parse");
    let r = fit(
        &model,
        &tte_pop_from_pairs(&pairs),
        &model.default_params,
        &fit_opts(),
    )
    .expect("SSE fit must succeed");

    let alpha = r.theta[0];
    let gamma = r.theta[1];
    let omega2 = r.omega[(0, 0)];
    eprintln!(
        "[SSE gompertz] alpha = {alpha:.5} (truth 0.002), gamma = {gamma:.5} (truth 0.05), omega^2 = {omega2:.4} (truth 0.05)"
    );

    // alpha and gamma trade off (the Gompertz baseline/growth collinearity), so
    // allow a wider alpha band; gamma is the better-determined of the two.
    assert!(
        (0.0012..0.0028).contains(&alpha),
        "alpha not recovered: got {alpha:.5}, expected ~0.002"
    );
    assert!(
        (0.043..0.060).contains(&gamma),
        "gamma not recovered: got {gamma:.5}, expected ~0.05"
    );
    // omega^2 (frailty on *gamma*, a nonlinear hazard parameter) is over-estimated by
    // FOCEI-Laplace (~0.08 vs truth 0.05), the same nonlinear-frailty limitation seen
    // for the Weibull shape (#440). Sane-range guard only; documented in expected.md.
    assert!(
        (0.03..0.13).contains(&omega2),
        "omega^2 out of sane range: got {omega2:.4} (FOCEI over-estimates nonlinear-frailty omega^2; truth 0.05, see #440)"
    );
    assert!(r.ofv.is_finite(), "OFV must be finite");
}
