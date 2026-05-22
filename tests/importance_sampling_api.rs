//! Integration tests for the `[fit_options] methods = [..., imp]` chain stage.
//!
//! Covers validation (Imp must follow another stage; not duplicated) and the
//! happy-path wire-up (the IS result lands on `FitResult.importance_sampling`
//! with a plausible relationship to the FOCEI Laplace OFV on a real model).
//!
//! All fits here cap iterations aggressively — convergence quality is not what
//! we're testing, only the chain integration.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, KappaTreatment};
use std::path::Path;

fn warfarin_setup() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
    FitOptions,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.outer_maxiter = 40;
    opts.is_samples = 200; // fast — accuracy is checked in Tier 3
    opts.is_seed = Some(7);
    (model, population, opts)
}

#[test]
fn imp_only_chain_is_rejected() {
    let (model, population, mut opts) = warfarin_setup();
    opts.methods = vec![EstimationMethod::Imp];
    let err = fit(&model, &population, &model.default_params, &opts)
        .err()
        .expect("methods = [imp] must be rejected");
    assert!(
        err.contains("first stage"),
        "expected `first stage` in error, got: {err}"
    );
}

#[test]
fn imp_first_in_chain_is_rejected() {
    let (model, population, mut opts) = warfarin_setup();
    opts.methods = vec![EstimationMethod::Imp, EstimationMethod::FoceI];
    let err = fit(&model, &population, &model.default_params, &opts)
        .err()
        .expect("methods = [imp, focei] must be rejected");
    assert!(
        err.contains("first stage"),
        "expected `first stage` in error, got: {err}"
    );
}

#[test]
fn imp_non_terminal_in_chain_is_rejected() {
    let (model, population, mut opts) = warfarin_setup();
    opts.methods = vec![
        EstimationMethod::FoceI,
        EstimationMethod::Imp,
        EstimationMethod::FoceI,
    ];
    let err = fit(&model, &population, &model.default_params, &opts)
        .err()
        .expect("non-terminal `imp` must be rejected");
    assert!(
        err.contains("final stage"),
        "expected `final stage` in error, got: {err}"
    );
}

#[test]
fn imp_duplicated_in_chain_is_rejected() {
    let (model, population, mut opts) = warfarin_setup();
    opts.methods = vec![
        EstimationMethod::FoceI,
        EstimationMethod::Imp,
        EstimationMethod::Imp,
    ];
    let err = fit(&model, &population, &model.default_params, &opts)
        .err()
        .expect("duplicated `imp` must be rejected");
    assert!(
        err.contains("at most once"),
        "expected `at most once` in error, got: {err}"
    );
}

#[test]
fn imp_after_focei_populates_field() {
    let (model, population, mut opts) = warfarin_setup();
    opts.methods = vec![EstimationMethod::FoceI, EstimationMethod::Imp];
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("focei → imp chain must produce a fit");
    let imp = result
        .importance_sampling
        .as_ref()
        .expect("importance_sampling field should be populated when imp is in the chain");
    assert!(
        imp.minus2_log_likelihood.is_finite(),
        "-2 LL must be finite, got {}",
        imp.minus2_log_likelihood
    );
    assert!(
        imp.mc_standard_error >= 0.0 && imp.mc_standard_error.is_finite(),
        "MC SE must be finite & non-negative, got {}",
        imp.mc_standard_error
    );
    assert_eq!(imp.n_samples, 200);
    assert_eq!(imp.proposal_df, 5.0);
    assert!(matches!(imp.kappa_treatment, KappaTreatment::NotApplicable));
    // ESS is reported as a fraction of K — must lie in [0, 1].
    assert!(
        (0.0..=1.0).contains(&imp.ess_min),
        "ess_min out of range: {}",
        imp.ess_min
    );
    assert!(
        (0.0..=1.0).contains(&imp.ess_median),
        "ess_median out of range: {}",
        imp.ess_median
    );
    // method_chain preserves the full chain; `method` (the final *estimating*
    // stage) drops the IMP suffix per design.
    assert_eq!(
        result.method_chain,
        vec![EstimationMethod::FoceI, EstimationMethod::Imp]
    );
    assert_eq!(result.method, EstimationMethod::FoceI);
}

#[test]
fn imp_after_focei_handles_fixed_theta_omega_sigma() {
    // `warfarin_fix.ferx` FIXes TVKA (theta), ETA_KA variance (omega), and
    // PROP_ERR (sigma). Exercises that FIX values flow correctly through the
    // IS proposal — Ω⁻¹ uses the FIXED variance, σ used in the obs NLL, θ
    // unchanged across IS samples — without panicking or producing NaN LL.
    let model = parse_model_file(Path::new("examples/warfarin_fix.ferx"))
        .expect("warfarin_fix model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.outer_maxiter = 40;
    opts.is_samples = 200;
    opts.is_seed = Some(11);
    opts.methods = vec![EstimationMethod::FoceI, EstimationMethod::Imp];

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("FIX-model focei → imp must produce a fit");
    let imp = result
        .importance_sampling
        .as_ref()
        .expect("importance_sampling field should be populated");
    assert!(
        imp.minus2_log_likelihood.is_finite(),
        "−2 log L on a FIX model must be finite, got {}",
        imp.minus2_log_likelihood
    );
    // FIXED theta/omega/sigma slots must be preserved in the reported fit —
    // IS doesn't update them, but a wiring bug could surface here.
    assert!(
        model.default_params.theta_fixed.iter().any(|&f| f),
        "test fixture should have at least one FIXED theta"
    );
    for (i, &fixed) in model.default_params.theta_fixed.iter().enumerate() {
        if fixed {
            assert_eq!(
                result.theta[i], model.default_params.theta[i],
                "FIXED theta[{}] must not move",
                i
            );
        }
    }
}

#[test]
fn imp_minus2_ll_is_in_loose_neighbourhood_of_focei_ofv() {
    // Warfarin is well-sampled (≈8 obs/subject) — the Laplace approximation is
    // good here, so the IS and FOCEI likelihoods should be within tens of OFV
    // units of each other. The sparse-data test that demonstrates a larger
    // divergence lives in the Tier 3 slow suite.
    let (model, population, mut opts) = warfarin_setup();
    opts.methods = vec![EstimationMethod::FoceI, EstimationMethod::Imp];
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("focei → imp chain must produce a fit");
    let imp = result.importance_sampling.unwrap();
    let gap = (imp.minus2_log_likelihood - result.ofv).abs();
    assert!(
        gap < 100.0,
        "IS −2LL ({:.2}) and FOCEI OFV ({:.2}) diverge by {:.2}; expected < 100 on \
         well-sampled warfarin",
        imp.minus2_log_likelihood,
        result.ofv,
        gap,
    );
}
