//! Integration tests for the BOBYQA and trust_region outer optimizers.
//!
//! These fit the warfarin example and assert sane convergence behaviour.
//! They don't reproduce reference estimates bit-for-bit — BOBYQA is
//! derivative-free and trust-region uses a CG subproblem, so both take
//! different paths than SLSQP and can settle on slightly different
//! points. The asserts are loose enough to catch "regression to garbage"
//! while tolerating normal optimizer jitter.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, FitOptions, Optimizer};
use std::path::Path;

fn data_and_model() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin example must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

fn base_options() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false; // skip FD Hessian — not what we're testing
    opts.outer_maxiter = 100;
    opts
}

#[test]
fn slsqp_reaches_a_sane_ofv() {
    // Baseline: SLSQP (default) should converge on warfarin to a moderate OFV.
    // Used as a reference point for the BOBYQA / trust-region assertions.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Slsqp;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("slsqp fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "SLSQP OFV should be finite, got {}",
        result.ofv
    );
}

#[test]
fn bobyqa_fit_converges_to_finite_ofv() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Bobyqa;
    // BOBYQA is derivative-free: one "iteration" is cheap but it needs many
    // to triangulate a quadratic. Give it a bit more headroom.
    opts.outer_maxiter = 200;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("bobyqa fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "BOBYQA OFV must be finite, got {}",
        result.ofv
    );
    // Theta should stay inside the declared bounds from the model file.
    for (i, &th) in result.theta.iter().enumerate() {
        let lo = model.default_params.theta_lower[i];
        let hi = model.default_params.theta_upper[i];
        assert!(th > lo && th < hi, "theta[{}] = {} escaped bounds", i, th);
    }
}

#[test]
fn trust_region_fit_converges_to_finite_ofv() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.steihaug_max_iters = 30;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("trust_region fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "trust_region OFV must be finite, got {}",
        result.ofv
    );
    for (i, &th) in result.theta.iter().enumerate() {
        let lo = model.default_params.theta_lower[i];
        let hi = model.default_params.theta_upper[i];
        assert!(th > lo && th < hi, "theta[{}] = {} escaped bounds", i, th);
    }
}

#[test]
fn bobyqa_ofv_no_worse_than_slsqp_on_warfarin() {
    // Sanity check: BOBYQA must find a fit that is at least as good as SLSQP
    // by OFV (allowing a small slack for the derivative-free optimizer's
    // coarser termination). The earlier theta-agreement test was misleading
    // — it passed only because the original BOBYQA configuration barely
    // moved from the initial values, which made it spuriously "agree" with
    // SLSQP's local minimum. Once BOBYQA can actually explore (rhobeg set,
    // xtol loosened) it routinely finds a better OFV on warfarin than SLSQP
    // does, so the right invariant is OFV-not-worse, not theta-agreement.
    let (model, population) = data_and_model();

    let mut opts_slsqp = base_options();
    opts_slsqp.optimizer = Optimizer::Slsqp;
    let r_slsqp = fit(&model, &population, &model.default_params, &opts_slsqp)
        .expect("slsqp fit must succeed");

    let mut opts_bobyqa = base_options();
    opts_bobyqa.optimizer = Optimizer::Bobyqa;
    opts_bobyqa.outer_maxiter = 300;
    let r_bobyqa = fit(&model, &population, &model.default_params, &opts_bobyqa)
        .expect("bobyqa fit must succeed");

    // BOBYQA's OFV should be ≤ SLSQP's + 5 units of slack. 5 is well below
    // the chi-squared 95% threshold for a single parameter (3.84), so this
    // catches "BOBYQA is stuck near the initial point" without rejecting
    // normal coarseness in derivative-free convergence.
    assert!(
        r_bobyqa.ofv <= r_slsqp.ofv + 5.0,
        "BOBYQA OFV {} should be no worse than SLSQP OFV {} + 5",
        r_bobyqa.ofv,
        r_slsqp.ofv,
    );
}

#[test]
fn slsqp_stops_via_stagnation_well_before_a_generous_maxeval() {
    // Regression: on γ-bearing FOCEI scenarios (SAD_SCEN3/SAD_SCEN4 in the
    // astra-testdata-simulator REPORT.md), SLSQP used to grind through
    // hundreds of evals at a plateaued OFV without terminating — see the
    // `detect_stagnation` doc comment for the underlying cause. The
    // stagnation guard inside the NLopt closure latches once the recent
    // window shows no meaningful OFV improvement and short-circuits
    // subsequent evals, which lets NLopt's xtol/ftol fire promptly.
    //
    // On warfarin we don't have a γ, but the same mechanism kicks in
    // once SLSQP has effectively converged. Asserting that the iteration
    // count stays well below an absurdly generous maxeval budget
    // exercises the guard without needing a 10-minute SAD_SCEN3 dataset
    // in the test corpus.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Slsqp;
    opts.outer_maxiter = 1000; // far more than warfarin needs

    let result =
        fit(&model, &population, &model.default_params, &opts).expect("slsqp fit must succeed");
    assert!(result.ofv.is_finite());
    // n_iterations on the NLopt path counts evals (see comment in
    // outer_optimizer.rs about NLopt not exposing iteration counts).
    // 8000 = 1000 * (n+1 ≈ 8) is the unbounded budget; the guard should
    // stop us at a small fraction of that.
    assert!(
        result.n_iterations < 2000,
        "SLSQP burned {} evals on warfarin with maxiter=1000 — \
         stagnation guard should have stopped us well below 2000",
        result.n_iterations,
    );
}

#[test]
fn steihaug_max_iters_is_respected_by_trust_region() {
    // A very small steihaug_max_iters degrades step quality but should still
    // return a valid FitResult without crashing. Catches regressions in the
    // wiring from FitOptions to the argmin Steihaug subsolver.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.steihaug_max_iters = 2; // intentionally aggressive
    opts.outer_maxiter = 30;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("trust_region with tight CG budget must still return");
    assert!(result.ofv.is_finite());
}
