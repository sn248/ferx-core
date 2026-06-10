//! NONMEM 7.5.1 `$COVARIANCE MATRIX=R` cross-check for the **IOV** FOCEI
//! covariance step (#223, building on the #209/#196/#129 rewrite).
//!
//! The non-IOV covariance path is exercised by `warfarin_covariance_nonmem.rs`.
//! IOV takes a different branch in `compute_covariance`: the κ block has no
//! fixed-EBE analytical gradient, so the Hessian is built from **second
//! differences of the reconverged objective** rather than the central-gradient
//! stencil. This test guards that branch end-to-end against NONMEM.
//!
//! It is also the IOV cross-check for this PR's covariance-robustness changes.
//! Both are inert on this (positive-definite) surface and the SEs must be
//! unchanged from the #209/#234 baseline:
//!   - the SIR fallback (`covariance_fallback`) only fires when the FD Hessian is
//!     negative-semidefinite, which warfarin_iov is not — the covariance step
//!     succeeds and the fallback is never reached;
//!   - the Ω⁻¹/log|Ω| cache reuse lives in the FOCE (non-interaction) arm of the
//!     covariance OFV closure; under FOCEI (`interaction = true`) that arm is not
//!     reached, so the reconverged-OFV second-difference stencil is identical.
//!
//! ## NONMEM reference
//!
//! NONMEM 7.5.1, FOCEI (`$ESTIMATION METHOD=1 INTER`), `$COVARIANCE MATRIX=R`,
//! on `data/warfarin_iov.csv` (1-cpt oral, proportional error, IOV on CL via a
//! single `KAPPA_CL` occasion random effect). The proportional error is the
//! `THETA`-coded SD (`$SIGMA 1 FIX`), directly comparable to ferx's `(sd)` form.
//! SEs are the `.ext` row at `ITERATION = -1000000001`.
//!
//! `#[ignore]`d outside the `slow-tests` feature (runs a fit to convergence).
//! Bands: 20% relative on theta / residual-error / omega (NONMEM-anchored,
//! matching the non-IOV FOCEI test); 40% on the weakly-identified IOV variance
//! `kappa_CL` (one occasion random effect over 10 subjects — the SE itself is
//! noisy in both engines).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const IOV_MODEL_SRC: &str = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  iov_column = OCC
";

/// One parameter's NONMEM reference SE and its acceptance band.
struct SeRef {
    name: &'static str,
    nm: f64,
    tol: f64,
}

const TIGHT: f64 = 0.20; // theta + residual-error + omega: NONMEM-anchored

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored IOV covariance SE cross-check (#223): opt in with --features slow-tests"
)]
fn iov_covariance_se_matches_nonmem() {
    let model = parse_model_string(IOV_MODEL_SRC).expect("warfarin IOV model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("warfarin IOV fit runs");

    // The covariance step must succeed (PD Hessian → no SIR fallback needed).
    assert!(
        result.covariance_matrix.is_some(),
        "IOV covariance step must produce a matrix"
    );
    assert_eq!(
        result.covariance_status,
        ferx_core::CovarianceStatus::Computed,
        "well-identified IOV model must produce a real covariance, not a SIR fallback"
    );

    let se_theta = result.se_theta.as_ref().expect("theta SEs present");
    let se_omega = result.se_omega.as_ref().expect("omega SEs present");
    let se_sigma = result.se_sigma.as_ref().expect("sigma SEs present");
    let se_kappa = result.se_kappa.as_ref().expect("kappa SEs present");

    // NONMEM 7.5.1 FOCEI $COVARIANCE MATRIX=R SEs (.ext, ITER=-1000000001).
    let refs = [
        SeRef {
            name: "TVCL",
            nm: 1.33623e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "TVV",
            nm: 3.41267e-1,
            tol: TIGHT,
        },
        SeRef {
            name: "TVKA",
            nm: 9.32221e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "PROP_ERR",
            nm: 1.0138e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "omega_CL",
            nm: 2.80628e-2,
            tol: TIGHT,
        },
        SeRef {
            name: "omega_V",
            nm: 6.44076e-3,
            tol: TIGHT,
        },
        SeRef {
            name: "omega_KA",
            nm: 2.79920e-2,
            tol: TIGHT,
        },
        // Weakly-identified IOV variance: one occasion effect over 10 subjects.
        SeRef {
            name: "kappa_CL",
            nm: 1.76108e-2,
            tol: 0.40,
        },
    ];
    let ferx = [
        se_theta[0],
        se_theta[1],
        se_theta[2],
        se_sigma[0],
        se_omega[0],
        se_omega[1],
        se_omega[2],
        se_kappa[0],
    ];

    for (r, &ferx_se) in refs.iter().zip(ferx.iter()) {
        let rel = (ferx_se - r.nm).abs() / r.nm;
        assert!(
            ferx_se.is_finite() && rel < r.tol,
            "SE({}) = {ferx_se:.6} vs NONMEM {:.6} — relative diff {:.1}% exceeds {:.0}% band",
            r.name,
            r.nm,
            rel * 100.0,
            r.tol * 100.0
        );
    }
}
