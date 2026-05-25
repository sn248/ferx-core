//! Integration tests for `suggest_start` and `suggest_start_thorough`.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{read_nonmem_csv, suggest_start, suggest_start_thorough};
use std::path::Path;

fn warfarin() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

fn two_cpt_iv() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model = parse_model_file(Path::new("examples/two_cpt_iv.ferx"))
        .expect("two_cpt_iv model must parse");
    let population = read_nonmem_csv(Path::new("data/two_cpt_iv.csv"), None, None)
        .expect("two_cpt_iv data must load");
    (model, population)
}

/// TVCL suggested for warfarin must be closer to the known truth (0.134) than the
/// bad starting value (0.5).  |log(suggested / 0.134)| < |log(0.5 / 0.134)|.
#[test]
fn test_suggest_start_warfarin_direction() {
    let (model, population) = warfarin();
    let result = suggest_start(&model, &population);

    // Find TVCL index.
    let tvcl_idx = result
        .params
        .theta_names
        .iter()
        .position(|n| n == "TVCL")
        .expect("TVCL must be a theta");

    let suggested = result.params.theta[tvcl_idx];
    let truth = 0.134_f64;
    let bad_start = 0.5_f64;

    let err_suggested = (suggested / truth).ln().abs();
    let err_bad = (bad_start / truth).ln().abs();

    assert!(
        err_suggested < err_bad,
        "suggested TVCL ({suggested:.4}) should be closer to truth ({truth}) than bad start ({bad_start}); \
         log-errors: suggested={err_suggested:.3}, bad={err_bad:.3}"
    );
}

/// Suggested params must all be within bounds.
#[test]
fn test_suggest_start_warfarin_within_bounds() {
    let (model, population) = warfarin();
    let result = suggest_start(&model, &population);
    for (i, &theta) in result.params.theta.iter().enumerate() {
        let lo = result.params.theta_lower[i];
        let hi = result.params.theta_upper[i];
        assert!(
            theta >= lo && theta <= hi,
            "theta[{i}] ({name}) = {theta} outside [{lo}, {hi}]",
            name = result.params.theta_names[i]
        );
    }
}

/// For 2-cpt IV, CL and V1 should be within 3× of the model default (sanity,
/// not accuracy — just confirming we get plausible values, not garbage).
#[test]
fn test_suggest_start_two_cpt_iv_sanity() {
    let (model, population) = two_cpt_iv();
    let result = suggest_start(&model, &population);

    let find = |name: &str| -> f64 {
        let idx = result
            .params
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or(0);
        result.params.theta[idx]
    };
    let find_default = |name: &str| -> f64 {
        let idx = model
            .default_params
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or(0);
        model.default_params.theta[idx]
    };

    for param in &["TVCL", "TVV1"] {
        let suggested = find(param);
        let default = find_default(param);
        if default > 0.0 && suggested > 0.0 {
            let ratio = (suggested / default).ln().abs();
            assert!(
                ratio < (3.0_f64).ln(),
                "{param} suggestion ({suggested:.3}) is more than 3× from default ({default:.3})"
            );
        }
    }
}

/// No subjects → returns model defaults without panicking, with a warning.
#[test]
fn test_suggest_start_empty_population() {
    let (model, _) = warfarin();
    let empty = ferx_core::types::Population {
        subjects: vec![],
        covariate_names: vec![],
        dv_column: "DV".into(),
    };
    let result = suggest_start(&model, &empty);
    assert!(!result.warnings.is_empty(), "must warn on empty population");
    assert_eq!(
        result.params.theta, model.default_params.theta,
        "empty population must return model defaults"
    );
}

/// suggest_start_thorough: all thetas within bounds for 2-cpt IV.
#[test]
fn test_suggest_start_thorough_two_cpt_iv_within_bounds() {
    let (model, population) = two_cpt_iv();
    let result = suggest_start_thorough(&model, &population);
    for (i, &theta) in result.params.theta.iter().enumerate() {
        let lo = result.params.theta_lower[i];
        let hi = result.params.theta_upper[i];
        assert!(
            theta >= lo && theta <= hi,
            "thorough: theta[{i}] ({name}) = {theta} outside [{lo}, {hi}]",
            name = result.params.theta_names[i]
        );
    }
}

/// suggest_start_thorough should move unwritten thetas away from model default.
///
/// For 2-cpt IV, CL/V1 are written by NCA; Q/V2 start at model default.
/// After the sweep, Q and/or V2 should have changed (rRMSE found a better value).
#[test]
fn test_suggest_start_thorough_moves_unwritten_thetas() {
    let (model, population) = two_cpt_iv();
    let fast = suggest_start(&model, &population);
    let thorough = suggest_start_thorough(&model, &population);

    // At least one theta should differ between the two results (the sweep did something).
    let any_changed = fast
        .params
        .theta
        .iter()
        .zip(thorough.params.theta.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);

    assert!(
        any_changed,
        "suggest_start_thorough should change at least one theta vs Option A"
    );
}

/// Tier 3 — slow: trust_region should converge from the bad TVCL=0.5 start
/// when auto_start = true corrects the initial values.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn test_trust_region_converges_with_auto_start() {
    use ferx_core::{fit, EstimationMethod, FitOptions, Optimizer};

    let (mut model, population) = warfarin();

    // Deliberately set TVCL to the bad starting value.
    let tvcl_idx = model
        .default_params
        .theta_names
        .iter()
        .position(|n| n == "TVCL")
        .unwrap();
    model.default_params.theta[tvcl_idx] = 0.5;

    let mut options = FitOptions::default();
    options.method = EstimationMethod::FoceI;
    options.outer_maxiter = 300;
    options.auto_start = true;
    options.optimizer = Optimizer::TrustRegion;
    options.run_covariance_step = false;

    let result =
        fit(&model, &population, &model.default_params, &options).expect("fit must not error");
    let ofv = result.ofv;
    assert!(
        ofv < 1000.0,
        "trust_region + auto_start should converge (OFV < 1000), got {ofv:.2}"
    );
}
