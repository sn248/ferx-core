//! End-to-end checks for `covariance_method` (R / S / RSR) — issue #223.
//!
//! The estimator *math* is unit-tested in `outer_optimizer.rs`
//! (`test_combine_covariance_*`). These tests exercise the full `fit()` path:
//! that the per-subject score cross-product `S` assembles, that all three
//! estimators produce finite positive SEs, and that the FOCE guard fires.
//!
//! NONMEM `$COV MATRIX=S` / `MATRIX=RSR` reference values are a follow-up (they
//! need a dedicated NONMEM run); the sanity anchor here is the information-matrix
//! equality — at the MLE of a well-specified model `R ≈ S`, so the `s` and `rsr`
//! SEs land within the same order of magnitude as the `r` SEs.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{
    fit, read_nonmem_csv, CovarianceMethod, CovarianceStatus, EstimationMethod, FitOptions,
};
use std::path::Path;

const WARFARIN_FOCEI: &str = r"
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  mu_referencing = true
";

fn warfarin_focei_opts(method: CovarianceMethod) -> FitOptions {
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.covariance_method = method;
    opts.verbose = false;
    opts
}

/// All SEs (theta, omega, sigma) flattened, in a stable order.
fn all_ses(r: &ferx_core::FitResult) -> Vec<f64> {
    let mut v = Vec::new();
    v.extend(r.se_theta.as_ref().expect("theta SEs").iter().copied());
    v.extend(r.se_omega.as_ref().expect("omega SEs").iter().copied());
    v.extend(r.se_sigma.as_ref().expect("sigma SEs").iter().copied());
    v
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full FOCEI fit ×3 for covariance_method R/S/RSR; opt in with --features slow-tests"
)]
fn covariance_methods_produce_consistent_ses_on_warfarin() {
    let model = parse_model_string(WARFARIN_FOCEI).expect("warfarin model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");

    let run = |m: CovarianceMethod| {
        let r = fit(&model, &pop, &model.default_params, &warfarin_focei_opts(m))
            .unwrap_or_else(|e| panic!("{m:?} fit failed: {e}"));
        assert_eq!(
            r.covariance_status,
            CovarianceStatus::Computed,
            "{m:?}: covariance must be Computed"
        );
        let ses = all_ses(&r);
        assert!(
            ses.iter().all(|s| s.is_finite() && *s > 0.0),
            "{m:?}: all SEs must be finite and positive, got {ses:?}"
        );
        ses
    };

    let se_r = run(CovarianceMethod::Hessian);
    let se_s = run(CovarianceMethod::CrossProduct);
    let se_rsr = run(CovarianceMethod::Sandwich);

    // Information-matrix equality on a well-specified model: S ≈ R, so the S and
    // RSR SEs land within the same order of magnitude as the R SEs. A factor-of-3
    // band catches assembly bugs (transpose, mis-scaling, wrong free-block
    // embedding would be orders off) without being flaky on a 10-subject fit.
    for (name, se_alt) in [("s", &se_s), ("rsr", &se_rsr)] {
        for (k, (&r, &a)) in se_r.iter().zip(se_alt.iter()).enumerate() {
            let ratio = a / r;
            assert!(
                (0.33..=3.0).contains(&ratio),
                "covariance_method={name}: SE[{k}] = {a:.4e} differs from R-matrix {r:.4e} \
                 by ratio {ratio:.2} (outside [0.33, 3.0]) — likely an assembly bug"
            );
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: runs a FOCE fit to exercise the covariance_method guard; opt in with --features slow-tests"
)]
fn covariance_method_rsr_rejects_foce_without_interaction() {
    // Sheiner–Beal FOCE (no interaction): the per-subject score omits the Ω prior,
    // so S would be inconsistent with R. The covariance step must refuse rather
    // than return a wrong matrix.
    let model = parse_model_string(WARFARIN_FOCEI).expect("warfarin model parses");
    let pop =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Foce;
    opts.interaction = false;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.covariance_method = CovarianceMethod::Sandwich;
    opts.verbose = false;

    let r = fit(&model, &pop, &model.default_params, &opts).expect("FOCE fit runs");
    assert_eq!(
        r.covariance_status,
        CovarianceStatus::Failed,
        "rsr under non-interaction FOCE must fail the covariance step, not return a matrix"
    );
    assert!(
        r.warnings.iter().any(|w| w.contains("requires FOCEI")),
        "expected a `requires FOCEI` covariance warning, got: {:?}",
        r.warnings
    );
}
