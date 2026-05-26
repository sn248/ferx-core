//! Convergence / objective-consistency tests for the IOV fitting path.
//!
//! Gate the slow ones so they are skipped in the default PR job (run nightly /
//! on demand):
//!
//!   cargo test --features slow-tests --test iov_convergence
//!
//! ## What these check (issue #101)
//!
//! 1. `foce_subject_nll_iov` is a *proper* augmented FOCE marginal — the
//!    per-occasion κ are integrated out through R̃ like the BSV η (kappa-augmented
//!    H + block-diagonal Σ), not via an additive κ MAP penalty. All estimation
//!    paths share one consistent objective.
//!
//! 2. For IOV models the outer optimizer computes its gradient by **re-converging
//!    the EBEs at each FD point** (`reconverged_fd_gradient`). The IOV variance
//!    components — especially Ω_iov — are weakly identified and their gradient is
//!    dominated by the EBE response (raising Ω_iov un-shrinks the kappas); the
//!    old fixed-EBE gradient missed this and left Ω_iov pinned at its initial
//!    value. With the fix, gradient FOCEI moves the variance components and a
//!    `saem → focei` chain polishes SAEM's result down to a *better* optimum
//!    (OFV ≈ 288.8 vs SAEM's ≈ 303) — see issue #101 rec #2.
//!
//! The FOCEI optimum (CL≈0.17, V≈8.5, KA≈1.15, Ω_iov≈0.047) matches the NONMEM
//! 7.5.1 reference basin (tests/nonmem/warfarin_iov.ctl). With the continuous
//! per-occasion-aware prediction (issue #104) its OFV is within ≈17 units of
//! NONMEM's 308.83 (down from ≈40); the residual is the simultaneous
//! cross-occasion dose/obs event ordering — see `tests/warfarin_iov_nonmem.rs`.
//!
//! 3. Pure FOCEI/SLSQP now reaches the minimum from the model's cold default
//!    start: for IOV models the SLSQP path auto-enables per-coordinate scaling
//!    so its uniform gradient cap no longer starves the omega/omega_iov step
//!    (issue #101 rec #2). Very far-off starts (e.g. a residual-error init off
//!    by >2×) can still stall; SAEM / `saem → focei` remain the most robust.
//!
//! The trust-region outer optimizer does not support IOV models (n_kappa > 0).
//! Use slsqp, bfgs, bobyqa, lbfgs, or mma for IOV fits.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult, Optimizer};
use std::path::Path;

/// OFV the FOCEI marginal minimum reaches on warfarin_iov with the continuous
/// per-occasion-aware prediction (issue #104). Pure SLSQP, pure BFGS, and the
/// `saem → focei` chain all agree here; the parameters match the NONMEM
/// reference basin (CL≈0.17, V≈8.5, Ω_iov≈0.047).
const IOV_FOCEI_OFV: f64 = 288.8;

fn load() -> (ferx_core::CompiledModel, ferx_core::Population) {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");
    (model, population)
}

/// Run a single method with the given optimizer.
fn run_single(method: EstimationMethod, optimizer: Optimizer) -> FitResult {
    let (model, population) = load();
    let mut opts = FitOptions::default();
    opts.method = method;
    opts.optimizer = optimizer;
    opts.outer_maxiter = 800;
    opts.run_covariance_step = false;
    opts.verbose = false;
    fit(&model, &population, &model.default_params, &opts).expect("IOV fit must succeed")
}

/// Run a `saem → focei` method chain with the given polishing optimizer.
fn run_chain(optimizer: Optimizer) -> FitResult {
    let (model, population) = load();
    let mut opts = FitOptions::default();
    opts.methods = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
    opts.optimizer = optimizer;
    opts.outer_maxiter = 800;
    opts.run_covariance_step = false;
    opts.verbose = false;
    fit(&model, &population, &model.default_params, &opts).expect("IOV chain fit must succeed")
}

/// SLSQP FOCEI on warfarin_iov returns a finite OFV (smoke).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_slsqp_converges() {
    let result = run_single(EstimationMethod::FoceI, Optimizer::Slsqp);
    assert!(
        result.ofv.is_finite(),
        "SLSQP IOV OFV must be finite, got {}",
        result.ofv
    );
}

/// The `saem → focei` chain reaches the FOCEI marginal minimum and, crucially,
/// improves on SAEM alone — the regression guard for the reconverged-EBE
/// gradient (issue #101 rec #2). Before that fix the fixed-EBE gradient left
/// FOCEI pinned at SAEM's point; now it descends to a strictly better OFV.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_chain_improves_on_saem_and_reaches_reference() {
    let saem = run_single(EstimationMethod::Saem, Optimizer::Slsqp);
    assert!(
        saem.ofv.is_finite() && saem.omega_iov.is_some(),
        "SAEM must return a finite OFV and omega_iov, got {}",
        saem.ofv
    );

    let chain = run_chain(Optimizer::Slsqp);
    assert!(
        chain.ofv.is_finite(),
        "chain OFV must be finite, got {}",
        chain.ofv
    );
    // FOCEI polish reaches the reference minimum...
    assert!(
        (chain.ofv - IOV_FOCEI_OFV).abs() < 2.0,
        "saem→focei chain OFV {:.4} should reach the FOCEI reference {:.2}",
        chain.ofv,
        IOV_FOCEI_OFV
    );
    // ...and meaningfully improves on SAEM alone (would NOT, with a fixed-EBE
    // gradient that can't move the variance components).
    assert!(
        chain.ofv < saem.ofv - 3.0,
        "FOCEI polish must improve on SAEM by >3 OFV units: chain {:.4} vs SAEM {:.4}",
        chain.ofv,
        saem.ofv
    );
}

/// The two outer optimizers (SLSQP, BFGS) driving the augmented marginal in a
/// `saem → focei` chain reach the same minimum — the objective and the
/// reconverged gradient are optimizer-independent. BFGS may report
/// `converged = false` while sitting on the minimum, so this asserts on OFV.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_chain_slsqp_and_bfgs_agree() {
    let slsqp = run_chain(Optimizer::Slsqp);
    let bfgs = run_chain(Optimizer::Bfgs);
    assert!(slsqp.ofv.is_finite() && bfgs.ofv.is_finite());
    assert!(
        (slsqp.ofv - bfgs.ofv).abs() < 1.0,
        "saem→focei OFV must match across optimizers: SLSQP {:.4} vs BFGS {:.4}",
        slsqp.ofv,
        bfgs.ofv
    );
    assert!(
        (slsqp.ofv - IOV_FOCEI_OFV).abs() < 2.0,
        "saem→focei/SLSQP OFV {:.4} should reach the FOCEI reference {:.2}",
        slsqp.ofv,
        IOV_FOCEI_OFV
    );
}

/// SAEM + IOV (smoke): fit() must return Ok with a finite OFV after a handful of
/// SAEM iterations. Tier-2 — calls the public API but exits after 5 iterations.
#[test]
fn iov_saem_smoke_returns_finite_ofv() {
    let (model, population) = load();
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.outer_maxiter = 5; // just enough to exercise the full E+M loop
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("SAEM IOV smoke must not Err");

    assert!(
        result.ofv.is_finite(),
        "SAEM IOV smoke OFV must be finite after 5 iter, got {}",
        result.ofv
    );
    assert!(
        result.omega_iov.is_some(),
        "omega_iov must be present in SAEM IOV result"
    );
}

/// Pure FOCEI/SLSQP from the model's cold default start reaches the FOCEI
/// minimum on warfarin_iov — no SAEM seeding needed. This is the payoff of the
/// IOV+SLSQP auto-scaling fix (issue #101 rec #2): before it, SLSQP's uniform
/// gradient cap starved the omega step and the fit stalled at OFV ≈ 292 with the
/// variance components pinned near their initial values.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_pure_slsqp_from_cold_start_reaches_minimum() {
    let focei = run_single(EstimationMethod::FoceI, Optimizer::Slsqp);
    assert!(
        (focei.ofv - IOV_FOCEI_OFV).abs() < 2.0,
        "pure FOCEI/SLSQP from the cold default start must reach the FOCEI \
         minimum {:.2}, got {:.4}",
        IOV_FOCEI_OFV,
        focei.ofv
    );
    // The fix is specifically about moving the IOV variance off its init: Ω_iov
    // starts at 0.01 and must climb toward ≈0.036.
    let iov = focei.omega_iov.expect("omega_iov present")[(0, 0)];
    assert!(
        iov > 0.02,
        "omega_iov must move off its 0.01 init toward ≈0.036, got {:.4}",
        iov
    );
}
